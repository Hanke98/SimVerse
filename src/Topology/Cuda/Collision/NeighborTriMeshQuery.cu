#include "Array/ArrayList.h"
#include "NeighborTriMeshQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "../../../Dynamics/Cuda/RigidBody/RigidBodySystem.h"
#include "Platform.h"
#include "Vector/Vector3D.h"
#include "Timer.h"
#include "NewTimer.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <thrust/sort.h>

namespace dyno
{
	static inline bool NMQ_CheckCuda(const char* tag)
	{
		cudaError_t err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("[NeighborTriMeshQuery][CUDA] %s: %s\n", tag, cudaGetErrorString(err));
			return false;
		}
		return true;
	}

	// Hard cap for triangles per patch in narrow/middle phase temporary buffers.
	// This matches one warp (32 lanes), so one lane can process one triangle candidate.
	static constexpr int NMQ_MaxPatchFaces = 32;
	// Host-side geometric aggregation tolerance for nearly identical contacts.
	static constexpr Real NMQ_ContactMergePosEps = Real(2e-3);
	static constexpr Real NMQ_ContactMergePosEps2 = NMQ_ContactMergePosEps * NMQ_ContactMergePosEps;
	static constexpr Real NMQ_ContactMergeNormalCos = Real(0.999);

	template<typename ContactPairT>
	static inline bool NMQContactPairLess(const ContactPairT& a, const ContactPairT& b)
	{
		if (a.bodyId1 != b.bodyId1) return a.bodyId1 < b.bodyId1;
		if (a.bodyId2 != b.bodyId2) return a.bodyId2 < b.bodyId2;
		if (a.contactType != b.contactType) return static_cast<int>(a.contactType) < static_cast<int>(b.contactType);

		if (a.pos1[0] != b.pos1[0]) return a.pos1[0] < b.pos1[0];
		if (a.pos1[1] != b.pos1[1]) return a.pos1[1] < b.pos1[1];
		if (a.pos1[2] != b.pos1[2]) return a.pos1[2] < b.pos1[2];
		if (a.pos2[0] != b.pos2[0]) return a.pos2[0] < b.pos2[0];
		if (a.pos2[1] != b.pos2[1]) return a.pos2[1] < b.pos2[1];
		if (a.pos2[2] != b.pos2[2]) return a.pos2[2] < b.pos2[2];

		if (a.normal1[0] != b.normal1[0]) return a.normal1[0] < b.normal1[0];
		if (a.normal1[1] != b.normal1[1]) return a.normal1[1] < b.normal1[1];
		if (a.normal1[2] != b.normal1[2]) return a.normal1[2] < b.normal1[2];
		if (a.normal2[0] != b.normal2[0]) return a.normal2[0] < b.normal2[0];
		if (a.normal2[1] != b.normal2[1]) return a.normal2[1] < b.normal2[1];
		if (a.normal2[2] != b.normal2[2]) return a.normal2[2] < b.normal2[2];
		if (a.interpenetration != b.interpenetration) return a.interpenetration < b.interpenetration;
		if (a.localId1 != b.localId1) return a.localId1 < b.localId1;
		if (a.localId2 != b.localId2) return a.localId2 < b.localId2;

		return false;
	}

	static inline Real NMQVectorLen2(const Vector<Real, 3>& v)
	{
		return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	}

	static inline Vector<Real, 3> NMQNormalizeOrFallback(const Vector<Real, 3>& n, const Vector<Real, 3>& fallback)
	{
		Real n2 = NMQVectorLen2(n);
		if (n2 > EPSILON * EPSILON)
			return n / glm::sqrt(n2);
		return fallback;
	}

	template<typename ContactPairT>
	static inline bool NMQContactPairGeomClose(const ContactPairT& a, const ContactPairT& b)
	{
		if (a.bodyId1 != b.bodyId1 || a.bodyId2 != b.bodyId2 || a.contactType != b.contactType)
			return false;

		Vector<Real, 3> dPos1 = a.pos1 - b.pos1;
		Vector<Real, 3> dPos2 = a.pos2 - b.pos2;
		if (NMQVectorLen2(dPos1) > NMQ_ContactMergePosEps2) return false;
		if (NMQVectorLen2(dPos2) > NMQ_ContactMergePosEps2) return false;

		Real n1aLen2 = NMQVectorLen2(a.normal1);
		Real n1bLen2 = NMQVectorLen2(b.normal1);
		Real n2aLen2 = NMQVectorLen2(a.normal2);
		Real n2bLen2 = NMQVectorLen2(b.normal2);
		if (n1aLen2 > EPSILON * EPSILON && n1bLen2 > EPSILON * EPSILON)
		{
			Real cosN1 = a.normal1.dot(b.normal1) / glm::sqrt(n1aLen2 * n1bLen2);
			if (cosN1 < NMQ_ContactMergeNormalCos) return false;
		}
		if (n2aLen2 > EPSILON * EPSILON && n2bLen2 > EPSILON * EPSILON)
		{
			Real cosN2 = a.normal2.dot(b.normal2) / glm::sqrt(n2aLen2 * n2bLen2);
			if (cosN2 < NMQ_ContactMergeNormalCos) return false;
		}
		return true;
	}

	template<typename ContactPairT>
	static inline void NMQStableSortAndAggregateContacts(CArray<ContactPairT>& contacts)
	{
		if (contacts.size() <= 1) return;

		std::stable_sort(contacts.begin(), contacts.begin() + contacts.size(), NMQContactPairLess<ContactPairT>);

		CArray<ContactPairT> aggregatedContacts;
		aggregatedContacts.resize(0);

		ContactPairT clusterRef = contacts[0];
		Real sumPos1[3] = { clusterRef.pos1[0], clusterRef.pos1[1], clusterRef.pos1[2] };
		Real sumPos2[3] = { clusterRef.pos2[0], clusterRef.pos2[1], clusterRef.pos2[2] };
		Real sumN1[3] = { clusterRef.normal1[0], clusterRef.normal1[1], clusterRef.normal1[2] };
		Real sumN2[3] = { clusterRef.normal2[0], clusterRef.normal2[1], clusterRef.normal2[2] };
		Real sumPen = clusterRef.interpenetration;
		uint clusterCount = 1;

		auto flushCluster = [&]()
		{
			const Real invCount = Real(1) / Real(clusterCount);
			clusterRef.pos1 = Vector<Real, 3>(sumPos1[0] * invCount, sumPos1[1] * invCount, sumPos1[2] * invCount);
			clusterRef.pos2 = Vector<Real, 3>(sumPos2[0] * invCount, sumPos2[1] * invCount, sumPos2[2] * invCount);
			clusterRef.normal1 = NMQNormalizeOrFallback(
				Vector<Real, 3>(sumN1[0] * invCount, sumN1[1] * invCount, sumN1[2] * invCount),
				clusterRef.normal1);
			clusterRef.normal2 = NMQNormalizeOrFallback(
				Vector<Real, 3>(sumN2[0] * invCount, sumN2[1] * invCount, sumN2[2] * invCount),
				clusterRef.normal2);
			clusterRef.interpenetration = sumPen * invCount;
			aggregatedContacts.pushBack(clusterRef);
		};

		for (uint i = 1; i < contacts.size(); ++i)
		{
			const ContactPairT& c = contacts[i];
			if (!NMQContactPairGeomClose(clusterRef, c))
			{
				flushCluster();
				clusterRef = c;
				sumPos1[0] = c.pos1[0]; sumPos1[1] = c.pos1[1]; sumPos1[2] = c.pos1[2];
				sumPos2[0] = c.pos2[0]; sumPos2[1] = c.pos2[1]; sumPos2[2] = c.pos2[2];
				sumN1[0] = c.normal1[0]; sumN1[1] = c.normal1[1]; sumN1[2] = c.normal1[2];
				sumN2[0] = c.normal2[0]; sumN2[1] = c.normal2[1]; sumN2[2] = c.normal2[2];
				sumPen = c.interpenetration;
				clusterCount = 1;
				continue;
			}

			sumPos1[0] += c.pos1[0]; sumPos1[1] += c.pos1[1]; sumPos1[2] += c.pos1[2];
			sumPos2[0] += c.pos2[0]; sumPos2[1] += c.pos2[1]; sumPos2[2] += c.pos2[2];
			sumN1[0] += c.normal1[0]; sumN1[1] += c.normal1[1]; sumN1[2] += c.normal1[2];
			sumN2[0] += c.normal2[0]; sumN2[1] += c.normal2[1]; sumN2[2] += c.normal2[2];
			sumPen += c.interpenetration;
			++clusterCount;
		}
		flushCluster();

		contacts.assign(aggregatedContacts);
	}

	static inline Real NMQCosOrOne(const Vector<Real, 3>& a, const Vector<Real, 3>& b)
	{
		const Real a2 = NMQVectorLen2(a);
		const Real b2 = NMQVectorLen2(b);
		if (a2 <= EPSILON * EPSILON || b2 <= EPSILON * EPSILON)
			return Real(1);

		const Real denom = glm::sqrt(a2 * b2);
		if (denom <= EPSILON)
			return Real(1);

		Real c = a.dot(b) / denom;
		if (c < Real(-1)) c = Real(-1);
		if (c > Real(1)) c = Real(1);
		return c;
	}

	template<typename ContactPairT>
	static inline void NMQPrintNearCoincidentContactsEvidence(const CArray<ContactPairT>& contacts, uint64_t frameId)
	{
		const int total = (int)contacts.size();
		if (total < 2)
		{
			printf("[NMQ_CP_NEAR] frame=%llu total=%d scanned=%d near_pairs=0\n",
				(unsigned long long)frameId, total, total);
			return;
		}

		const int scanLimit = total > 256 ? 256 : total;
		const int maxPrintPairs = 24;
		int nearPairs = 0;
		int printed = 0;

		for (int i = 0; i < scanLimit; ++i)
		{
			for (int j = i + 1; j < scanLimit; ++j)
			{
				const ContactPairT& a = contacts[(uint)i];
				const ContactPairT& b = contacts[(uint)j];

				if (!NMQContactPairGeomClose(a, b))
					continue;

				nearPairs++;

				if (printed >= maxPrintPairs)
					continue;

				const Vector<Real, 3> dPos1 = a.pos1 - b.pos1;
				const Vector<Real, 3> dPos2 = a.pos2 - b.pos2;
				const Real dPos1Len = glm::sqrt(NMQVectorLen2(dPos1));
				const Real dPos2Len = glm::sqrt(NMQVectorLen2(dPos2));
				const Real cosN1 = NMQCosOrOne(a.normal1, b.normal1);
				const Real cosN2 = NMQCosOrOne(a.normal2, b.normal2);
				const Real dPen = abs(a.interpenetration - b.interpenetration);

				printf("[NMQ_CP_NEAR] frame=%llu i=%d j=%d body=(%d,%d) triA=(%d,%d) triB=(%d,%d) "
					"|dPos1|=%.9g |dPos2|=%.9g cosN1=%.9g cosN2=%.9g dPen=%.9g\n",
					(unsigned long long)frameId,
					i, j,
					(int)a.bodyId1, (int)a.bodyId2,
					(int)a.localId1, (int)a.localId2,
					(int)b.localId1, (int)b.localId2,
					(double)dPos1Len, (double)dPos2Len,
					(double)cosN1, (double)cosN2, (double)dPen);
				printed++;
			}
		}

		printf("[NMQ_CP_NEAR_SUM] frame=%llu total=%d scanned=%d near_pairs=%d printed=%d pos_eps=%.9g normal_cos=%.9g\n",
			(unsigned long long)frameId,
			total,
			scanLimit,
			nearPairs,
			printed,
			(double)NMQ_ContactMergePosEps,
			(double)NMQ_ContactMergeNormalCos);
		if (scanLimit < total)
		{
			printf("[NMQ_CP_NEAR_SUM] frame=%llu note=scanned_first_%d_contacts_only\n",
				(unsigned long long)frameId, scanLimit);
		}
	}

	IMPLEMENT_TCLASS(NeighborTriMeshQuery, TDataType)

	__device__ inline int NLQ_ClampInt(int v, int lo, int hi)
	{
		return v < lo ? lo : (v > hi ? hi : v);
	}

	__device__ inline bool NLQ_IsAdjacent(List<int>& adj, int other)
	{
		for (int i = 0; i < adj.size(); ++i)
		{
			if (adj[i] == other)
				return true;
		}
		return false;
	}

	__device__ inline int NLQ_WarpReduceSum(int v)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
			v += __shfl_down_sync(0xffffffff, v, offset);
		return v;
	}

	__device__ inline Vec3f NLQ_WarpReduceMinVec3(Vec3f v)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
		{
			Real other = __shfl_down_sync(0xffffffff, v[0], offset);
			v[0] = v[0] < other ? v[0] : other;
			other = __shfl_down_sync(0xffffffff, v[1], offset);
			v[1] = v[1] < other ? v[1] : other;
			other = __shfl_down_sync(0xffffffff, v[2], offset);
			v[2] = v[2] < other ? v[2] : other;
		}
		return v;
	}

	__device__ inline Vec3f NLQ_WarpReduceMaxVec3(Vec3f v)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
		{
			Real other = __shfl_down_sync(0xffffffff, v[0], offset);
			v[0] = v[0] > other ? v[0] : other;
			other = __shfl_down_sync(0xffffffff, v[1], offset);
			v[1] = v[1] > other ? v[1] : other;
			other = __shfl_down_sync(0xffffffff, v[2], offset);
			v[2] = v[2] > other ? v[2] : other;
		}
		return v;
	}

	__device__ inline int NLQ_WarpExclusivePrefix(int v, int lane)
	{
		int sum = v;
		for (int offset = 1; offset < 32; offset <<= 1)
		{
			int n = __shfl_up_sync(0xffffffff, sum, offset);
			if (lane >= offset)
				sum += n;
		}
		return sum - v;
	}

	// Transform a point from current world space to target rest-world space.
	// p_rest = RRest^T * (p_world - tRest)
	__device__ inline Vec3f NLQ_TransformWorldPointToRest(
		const Vec3f& pWorld,
		const Mat3f& RRest,
		const Vec3f& tRest)
	{
		return RRest.transpose() * (pWorld - tRest);
	}

	// inline bool NLQ_BuildShape2RigidBodyIds(
	// 	const DArray<Pair<uint, uint>>& mapping,
	// 	int shapeCount,
	// 	std::vector<int>& shape2RigidBodyIds)
	// {
	// 	CArray<Pair<uint, uint>> hostMapping;
	// 	hostMapping.assign(mapping);
	// 	if (hostMapping.size() == 0)
	// 		return false;

	// 	// shape2RigidBodyIds.assign(shapeCount, -1);
	// 	shape2RigidBodyIds.assign(hostMapping.size(), -1);
	// 	for (uint i = 0; i < hostMapping.size(); ++i)
	// 	{
	// 		uint shapeId = hostMapping[i].first;
	// 		if (shapeId < shapeCount)
	// 			shape2RigidBodyIds[shapeId] = static_cast<int>(hostMapping[i].second);
	// 	}

	// 	return true;
	// }

	// template<typename Real, typename Coord, typename Matrix>
	__device__ inline void NLQ_GetRelativeTransform(
		int shapeId,
		const DArray<int>& shape2RigidBodyIds,
		const DArray<Vec3f>& centers,
		const DArray<Mat3f>& rotations,
		const DArray<Vec3f>& restShapeCenters,
		const DArray<Mat3f>& restShapeRotations,
		Mat3f& RRel,
		Vec3f& tRel,
		int& bodyId)
	{
		bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0 && shapeId >= 0 && shapeId < shape2RigidBodyIds.size())
			bodyId = shape2RigidBodyIds[shapeId];

		if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
		{
			RRel = Mat3f::identityMatrix();
			tRel = Vec3f(Real(0));
			return;
		}

		Vec3f tCurr = centers[bodyId];
		Mat3f RCurr = rotations[bodyId];

		Vec3f tRest = Vec3f(Real(0));
		Mat3f RRest = Mat3f::identityMatrix();
		if (shapeId >= 0 && shapeId < restShapeCenters.size())
			tRest = restShapeCenters[shapeId];
		if (shapeId >= 0 && shapeId < restShapeRotations.size())
			RRest = restShapeRotations[shapeId];

		RRel = RCurr * RRest.transpose();
		tRel = tCurr - RRel * tRest;
	}

	__global__ void NLQ_ComputeShapeRestTransforms(
		DArray<Mat3f> outR,
		DArray<Vec3f> outT,
		DArray<Vec3f> centers,
		DArray<Mat3f> rotations,
		DArray<Vec3f> restShapeCenters,
		DArray<Mat3f> restShapeRotations,
		DArray<int> shape2RigidBodyIds)
	{
		int shapeId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (shapeId >= outR.size() || shapeId >= outT.size()) return;

		Mat3f RRel = Mat3f::identityMatrix();
		Vec3f tRel = Vec3f(Real(0));
		int bodyId = shapeId;
		NLQ_GetRelativeTransform(
			shapeId,
			shape2RigidBodyIds,
			centers,
			rotations,
			restShapeCenters,
			restShapeRotations,
			RRel,
			tRel,
			bodyId);

		outR[shapeId] = RRel;
		outT[shapeId] = tRel;
	}

	__global__ void NLQ_CountGroup2PatchCounts(
		DArray<int> counts,
		DArray<int> groupedSources,
		DArray<int> shape2PatchOffsets,
		int patchCount)
	{
		int idx = threadIdx.x + (blockIdx.x * blockDim.x);
		if (idx >= groupedSources.size() || idx >= counts.size()) return;
		int shapeId = groupedSources[idx];
		if (shapeId < 0 || shapeId + 1 >= shape2PatchOffsets.size())
		{
			counts[idx] = 0;
			return;
		}
		int sBegin = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, patchCount);
		int sEnd = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, patchCount);
		int sCount = sEnd - sBegin;
		counts[idx] = sCount > 0 ? sCount : 0;
	}

	__global__ void NLQ_BuildTarget2SourceCountsAndGroupOffsets(
		DArray<int> target2SourceCounts,
		DArray<int> group2PatchOffsets,
		DArray<int> group2PatchCounts,
		DArray<int> group2TargetIds,
		DArray<int> target2GroupOffsets,
		DArray<int> target2GroupCounts,
		int shapeCount)
	{
		int target = threadIdx.x + (blockIdx.x * blockDim.x);
		if (target >= shapeCount) return;
		if (target >= target2SourceCounts.size()) return;

		int groupStart = target < target2GroupOffsets.size() ? target2GroupOffsets[target] : -1;
		int groupCount = target < target2GroupCounts.size() ? target2GroupCounts[target] : 0;
		if (groupStart < 0 || groupCount <= 0)
		{
			target2SourceCounts[target] = 0;
			return;
		}

		int sum = 0;
		for (int i = 0; i < groupCount; ++i)
		{
			int g = groupStart + i;
			if (g < 0 || g >= group2PatchCounts.size())
				continue;
			group2PatchOffsets[g] = sum;
			group2TargetIds[g] = target;
			int c = group2PatchCounts[g];
			if (c > 0)
				sum += c;
		}
		target2SourceCounts[target] = sum;
	}

	__global__ void NLQ_BuildGroup2GlobalOffsets(
		DArray<int> group2GlobalOffsets,
		DArray<int> group2PatchOffsets,
		DArray<int> group2TargetIds,
		DArray<int> target2SourceOffsets)
	{
		int g = threadIdx.x + (blockIdx.x * blockDim.x);
		if (g >= group2GlobalOffsets.size() || g >= group2PatchOffsets.size() || g >= group2TargetIds.size())
			return;
		int target = group2TargetIds[g];
		if (target < 0 || target >= target2SourceOffsets.size())
		{
			group2GlobalOffsets[g] = 0;
			return;
		}
		group2GlobalOffsets[g] = target2SourceOffsets[target] + group2PatchOffsets[g];
	}

	__global__ void NLQ_FillGroup2PatchDataByGroup(
		DArray<AABB> outAabbs,
		DArray<uint> outIds,
		DArray<int> outTargetIds,
		DArray<AABB> patchAabbsWorld,
		DArray<int> groupedSources,
		DArray<int> shape2PatchOffsets,
		DArray<int> group2GlobalOffsets,
		DArray<int> group2PatchCounts,
		DArray<int> group2TargetIds,
		int patchCount)
	{
		int g = threadIdx.x + (blockIdx.x * blockDim.x);
		if (g >= groupedSources.size() || g >= group2GlobalOffsets.size() || g >= group2PatchCounts.size() || g >= group2TargetIds.size())
			return;

		int target = group2TargetIds[g];
		int shapeId = groupedSources[g];
		if (shapeId < 0 || shapeId + 1 >= shape2PatchOffsets.size())
			return;

		int sBegin = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, patchCount);
		int sEnd = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, patchCount);
		int sCount = sEnd - sBegin;
		if (sCount <= 0)
			return;

		int outBegin = group2GlobalOffsets[g];
		int outCount = group2PatchCounts[g];
		if (outBegin < 0 || outCount <= 0)
			return;

		int copyCount = sCount < outCount ? sCount : outCount;
		for (int localIdx = 0; localIdx < copyCount; ++localIdx)
		{
			int outIdx = outBegin + localIdx;
			int srcIdx = sBegin + localIdx;
			if (outIdx >= outAabbs.size() || outIdx >= outIds.size() || outIdx >= outTargetIds.size() || srcIdx < 0 || srcIdx >= patchAabbsWorld.size())
				break;

			outAabbs[outIdx] = patchAabbsWorld[srcIdx];
			outIds[outIdx] = (uint)srcIdx;
			outTargetIds[outIdx] = target;
		}
	}

		// Warp-per-patch: build source patch AABBs directly from triangles (current world) in target rest-world space.
		template<typename Coord, typename Triangle>
		__global__ void NLQ_UpdateSourcePatchAabbsFromTrianglesWarp(
			DArray<AABB> outAabbs,
			DArray<uint> source2PatchIds,
			DArray<int> source2TargetIds,
			DArray<int> patch2TriOffsets,
			DArray<int> patch2TriIndices,
			DArray<Triangle> triangles,
			DArray<Coord> vertices,
			DArray<uint> patch2Shape,
			DArray<Mat3f> shapeRestR,
			DArray<Vec3f> shapeRestT,
			int patchTriCount,
			int triCount,
			int vertexCount,
			bool verticesInRestWorld)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			int warpId = tId / 32;
			int lane = tId % 32; 

		if (warpId >= outAabbs.size())
			return;

			int patchId = (int)source2PatchIds[warpId];
			if (patchId < 0 || patchId + 1 >= patch2TriOffsets.size())
				return;

			int targetId = source2TargetIds[warpId];
			if (targetId < 0 || targetId >= shapeRestR.size() || targetId >= shapeRestT.size())
				return;

			int sourceShapeId = patchId < patch2Shape.size() ? (int)patch2Shape[patchId] : -1;
			if (verticesInRestWorld)
			{
				if (sourceShapeId < 0 || sourceShapeId >= shapeRestR.size() || sourceShapeId >= shapeRestT.size())
					return;
			}

			int start = NLQ_ClampInt(patch2TriOffsets[patchId], 0, patchTriCount);
			int end = NLQ_ClampInt(patch2TriOffsets[patchId + 1], 0, patchTriCount);
			int triCountLocal = end - start;
			if (triCountLocal <= 0)
			{
				if (lane == 0)
				{
					AABB box;
					auto zero = Vec3f(Real(0));
					box.v0 = zero;
					box.v1 = zero;
					outAabbs[warpId] = box;
				}
				return;
			}

			Mat3f RTarget = shapeRestR[targetId];
			Vec3f tTarget = shapeRestT[targetId];
			Mat3f RTargetT = RTarget.transpose();

			Mat3f RSource = Mat3f::identityMatrix();
			Vec3f tSource = Vec3f(Real(0));
			if (verticesInRestWorld)
			{
				RSource = shapeRestR[sourceShapeId];
				tSource = shapeRestT[sourceShapeId];
			}

			Vec3f localMin(REAL_MAX);
			Vec3f localMax(-REAL_MAX);
			int hasLocal = 0;

			// One warp handles one patch; precondition from host check: triCountLocal <= 32.
			if (lane < triCountLocal)
			{
				int triId = patch2TriIndices[start + lane];
				bool triValid = (triId >= 0 && triId < triCount);
				if (triValid)
				{
					Triangle tri = triangles[triId];
					int v0 = tri[0];
					int v1 = tri[1];
					int v2 = tri[2];

					bool vValid = (v0 >= 0 && v0 < vertexCount
						&& v1 >= 0 && v1 < vertexCount
						&& v2 >= 0 && v2 < vertexCount);
					if (vValid)
					{
						// get triangle vertices
						Vec3f p0c = vertices[v0];
						Vec3f p1c = vertices[v1];
						Vec3f p2c = vertices[v2];

						// vertices:
						// - current world (legacy): directly transform to target rest
						// - rest world (static): transform to current world by source shape, then to target rest by target shape
						if (verticesInRestWorld)
						{
							p0c = RSource * p0c + tSource;
							p1c = RSource * p1c + tSource;
							p2c = RSource * p2c + tSource;
						}

						Vec3f p0 = RTargetT * (p0c - tTarget);
						Vec3f p1 = RTargetT * (p1c - tTarget);
						Vec3f p2 = RTargetT * (p2c - tTarget);

						localMin = p0.minimum(p1).minimum(p2);
						localMax = p0.maximum(p1).maximum(p2);
						hasLocal = 1;
					}
				}
			}

			Vec3f warpMin = NLQ_WarpReduceMinVec3(localMin);
			Vec3f warpMax = NLQ_WarpReduceMaxVec3(localMax);
			int validCount = NLQ_WarpReduceSum(hasLocal);

			if (lane == 0)
			{
				AABB box;
				if (validCount <= 0)
				{
					auto zero = Vec3f(Real(0));
					box.v0 = zero;
					box.v1 = zero;
				}
				else
				{
					box.v0 = warpMin;
					box.v1 = warpMax;
				}
				outAabbs[warpId] = box;
			}
		}

	// template<typename Real, typename Coord, typename Matrix, typename AABB>
	// __global__ void NLQ_UpdateShapeAabbs(
	// 	DArray<AABB> worldAabbs,
	// 	DArray<AABB> localAabbs,
	// 	DArray<Vec3f> centers,
	// 	DArray<Mat3f> rotations,
	// 	DArray<int> shape2RigidBodyIds)
	// {
	// 	int shapeId = threadIdx.x + (blockIdx.x * blockDim.x);
	// 	if (shapeId >= localAabbs.size() || shapeId >= worldAabbs.size())
	// 		return;

	// 	int bodyId = shapeId;
	// 	if (shape2RigidBodyIds.size() > 0)
	// 	{
	// 		if (shapeId < 0 || shapeId >= shape2RigidBodyIds.size())
	// 		{
	// 			worldAabbs[shapeId] = localAabbs[shapeId];
	// 			return;
	// 		}
	// 		bodyId = shape2RigidBodyIds[shapeId];
	// 	}

	// 	if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
	// 	{
	// 		worldAabbs[shapeId] = localAabbs[shapeId];
	// 		return;
	// 	}

	// 	worldAabbs[shapeId] = NLQ_TransformLocalAabbToWorld(
	// 		localAabbs[shapeId],
	// 		rotations[bodyId],
	// 		centers[bodyId]);
	// }

	template<typename TDataType, typename AABB>
	__global__ void NLQ_RequestIntersectionNumberBVH(
		DArray<uint> count,
		DArray<AABB> sourceAabbs,
		DArray<int> source2TargetIds,
		DArray<LinearBVH<TDataType>> targetBVHs,
		DArray<int> targetBVHValid,
		DArray<int> shape2PatchOffsets,
		DArray<AABB> patchAabbs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sourceAabbs.size() || tId >= source2TargetIds.size() || tId >= count.size())
			return;

		int target = source2TargetIds[tId];
		if (target < 0 || target + 1 >= shape2PatchOffsets.size())
		{
			count[tId] = 0;
			return;
		}

		int tBegin = NLQ_ClampInt(shape2PatchOffsets[target], 0, patchAabbs.size());
		int tEnd = NLQ_ClampInt(shape2PatchOffsets[target + 1], 0, patchAabbs.size());
		int tCount = tEnd - tBegin;
		if (tCount <= 0)
		{
			count[tId] = 0;
			return;
		}

		if (tCount == 1)
		{
			if (tBegin >= 0 && tBegin < patchAabbs.size())
				count[tId] = sourceAabbs[tId].checkOverlap(patchAabbs[tBegin]) ? 1 : 0;
			else
				count[tId] = 0;
			return;
		}

		if (target < 0 || target >= targetBVHs.size() || target >= targetBVHValid.size())
		{
			count[tId] = 0;
			return;
		}
		if (targetBVHValid[target] == 0)
		{
			count[tId] = 0;
			return;
		}

		count[tId] = targetBVHs[target].requestIntersectionNumber(sourceAabbs[tId]);
	}

	template<typename TDataType, typename AABB>
	__global__ void NLQ_RequestIntersectionIdsBVH(
		DArrayList<int> idLists,
		DArray<AABB> sourceAabbs,
		DArray<int> source2TargetIds,
		DArray<LinearBVH<TDataType>> targetBVHs,
		DArray<int> targetBVHValid,
		DArray<int> shape2PatchOffsets,
		DArray<AABB> patchAabbs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sourceAabbs.size() || tId >= source2TargetIds.size() || tId >= idLists.size())
			return;

		int target = source2TargetIds[tId];
		if (target < 0 || target + 1 >= shape2PatchOffsets.size())
			return;

		int tBegin = NLQ_ClampInt(shape2PatchOffsets[target], 0, patchAabbs.size());
		int tEnd = NLQ_ClampInt(shape2PatchOffsets[target + 1], 0, patchAabbs.size());
		int tCount = tEnd - tBegin;
		if (tCount <= 0)
			return;

		if (tCount == 1)
		{
			if (tBegin >= 0 && tBegin < patchAabbs.size())
			{
				if (sourceAabbs[tId].checkOverlap(patchAabbs[tBegin]))
					idLists[tId].insert(0);
			}
			return;
		}

		if (target < 0 || target >= targetBVHs.size() || target >= targetBVHValid.size())
			return;
		if (targetBVHValid[target] == 0)
			return;

		targetBVHs[target].requestIntersectionIds(idLists[tId], sourceAabbs[tId]);
	}

	template<typename Box3D>
	__global__ void NTQ_SetupAABBFromElementIds(
		DArray<AABB> boundingBox,
		DArray<int> shape2ElementIds,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset,
		Real boundary_expand)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shape2ElementIds.size()) return;

		int elementId = shape2ElementIds[tId];
		if (elementId < 0)
		{
			AABB box;
			auto zero = decltype(box.v0)(Real(0));
			box.v0 = zero;
			box.v1 = zero;
			boundingBox[tId] = box;
			return;
		}

		ElementType eleType = elementOffset.checkElementType((uint)elementId);

		AABB box;
		auto zero = decltype(box.v0)(Real(0));
		box.v0 = zero;
		box.v1 = zero;
		bool valid = true;
		switch (eleType)
		{
		case ET_SPHERE:
		{
			int localId = elementId - elementOffset.sphereIndex();
			if (localId < 0 || localId >= (int)spheres.size())
			{
				valid = false;
				break;
			}
			box = spheres[localId].aabb();
			break;
		}
		case ET_BOX:
		{
			int localId = elementId - elementOffset.boxIndex();
			if (localId < 0 || localId >= (int)boxes.size())
			{
				valid = false;
				break;
			}
			box = boxes[localId].aabb();
			break;
		}
		case ET_TET:
		{
			int localId = elementId - elementOffset.tetIndex();
			if (localId < 0 || localId >= (int)tets.size())
			{
				valid = false;
				break;
			}
			box = tets[localId].aabb();
			break;
		}
		case ET_CAPSULE:
		{
			int localId = elementId - elementOffset.capsuleIndex();
			if (localId < 0 || localId >= (int)caps.size())
			{
				valid = false;
				break;
			}
			box = caps[localId].aabb();
			break;
		}
		case ET_TRI:
		{
			int localId = elementId - elementOffset.triangleIndex();
			if (localId < 0 || localId >= (int)tris.size())
			{
				valid = false;
				break;
			}
			boundary_expand = 0.01;
			box = tris[localId].aabb();
			break;
		}
		default:
			valid = false;
			break;
		}

		if (!valid)
		{
			boundingBox[tId] = box;
			return;
		}

		box.v0 -= boundary_expand;
		box.v1 += boundary_expand;

		boundingBox[tId] = box;
	}

	__global__ void NLQ_CountShapePairs(
		DArray<int> counts,
		DArrayList<int> contactList,
		DArrayList<int> adjacentShapes,
		bool enableAdjacentFilter,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		List<int>& list_i = contactList[tId];
		int cnt = 0;
		bool useAdj = enableAdjacentFilter && tId < adjacentShapes.size();

		for (int j = 0; j < list_i.size(); j++)
		{
			int nb = list_i[j];
			if (nb <= tId || nb < 0 || nb >= shapeCount)
				continue;

			if (useAdj && NLQ_IsAdjacent(adjacentShapes[tId], nb))
				continue;

			cnt++;
		}

		counts[tId] = cnt;
	}

	__global__ void NLQ_SetShapePairs(
		DArray<Pair<uint, uint>> shapePairs,
		DArrayList<int> contactList,
		DArray<int> prefix,
		DArray<int> counts,
		DArrayList<int> adjacentShapes,
		bool enableAdjacentFilter,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		List<int>& list_i = contactList[tId];
		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;
		bool useAdj = enableAdjacentFilter && tId < adjacentShapes.size();

		for (int j = 0; j < list_i.size(); j++)
		{
			int nb = list_i[j];
			if (nb <= tId || nb < 0 || nb >= shapeCount)
				continue;

			if (useAdj && NLQ_IsAdjacent(adjacentShapes[tId], nb))
				continue;

			if (write < size && (offset + write) < shapePairs.size())
			{
				shapePairs[offset + write] = Pair<uint, uint>((uint)tId, (uint)nb);
				write++;
			}
		}
	}

	// Count how many times a shape is a target from shapePairs
	__global__ void NLQ_CountTargetShapes(
		DArray<int> targetCounts,
		DArray<Pair<uint, uint>> shapePairs,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> lp = shapePairs[tId];
		int target = (int)lp.second;
		if (target < 0 || target >= shapeCount)
			return;

		atomicAdd(&targetCounts[target], 1);
	}

	// Group shape pairs by target shape
	// That is, reorder the source shapes so that they are grouped by target shape
	__global__ void NLQ_GroupShapePairsByTarget(
		DArray<int> groupedSources,
		DArray<int> targetOffsets,
		DArray<int> targetWrite,
		DArray<Pair<uint, uint>> shapePairs,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> lp = shapePairs[tId];
		int source = (int)lp.first;
		int target = (int)lp.second;
		if (source < 0 || source >= shapeCount || target < 0 || target >= shapeCount)
			return;

		int local = atomicAdd(&targetWrite[target], 1);
		int out = targetOffsets[target] + local;
		if (out >= 0 && out < groupedSources.size())
			groupedSources[out] = source;
	}

	__global__ void NLQ_ExtractSourceTargetFromShapePairs(
		DArray<int> outTargets,
		DArray<int> outSources,
		DArray<Pair<uint, uint>> shapePairs,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size() || tId >= outTargets.size() || tId >= outSources.size())
			return;

		Pair<uint, uint> pair = shapePairs[tId];
		int source = (int)pair.first;
		int target = (int)pair.second;
		if (source < 0 || source >= shapeCount || target < 0 || target >= shapeCount)
		{
			outTargets[tId] = shapeCount;
			outSources[tId] = -1;
			return;
		}

		outTargets[tId] = target;
		outSources[tId] = source;
	}

	__global__ void NLQ_CountTargetShapesFromSortedKeys(
		DArray<int> targetCounts,
		DArray<int> sortedTargets,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sortedTargets.size())
			return;

		int target = sortedTargets[tId];
		if (target < 0 || target >= shapeCount)
			return;

		int prev = (tId > 0) ? sortedTargets[tId - 1] : -1;
		if (tId > 0 && prev == target)
			return;

		int end = tId + 1;
		int N = (int)sortedTargets.size();
		while (end < N && sortedTargets[end] == target)
			++end;

		targetCounts[target] = end - tId;
	}

	__global__ void NLQ_CopyCountU2I(
		DArray<int> dst,
		DArray<uint> src)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= dst.size() || tId >= src.size())
			return;
		dst[tId] = (int)src[tId];
	}

	__global__ void NLQ_SetPatchPairsFromContactList(
		DArray<Pair<uint, uint>> patchPairs,
		DArrayList<int> contactList,
		DArray<int> prefix,
		DArray<int> counts,
		DArray<uint> source2PatchIds,
		DArray<int> source2TargetIds,
		DArray<int> shape2PatchOffsets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size() || tId >= source2PatchIds.size() || tId >= source2TargetIds.size())
			return;

		int target = source2TargetIds[tId];
		if (target < 0 || target + 1 >= shape2PatchOffsets.size())
			return;

		int tBegin = shape2PatchOffsets[target];
		int tEnd = shape2PatchOffsets[target + 1];
		int tCount = tEnd - tBegin;
		if (tCount <= 0)
			return;

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;
		uint srcId = source2PatchIds[tId];

		List<int>& list_i = contactList[tId];
		for (int j = 0; j < list_i.size(); j++)
		{
			int targetIdx = list_i[j];
			if (targetIdx < 0 || targetIdx >= tCount)
				continue;

			if (write < size && (offset + write) < patchPairs.size())
			{
				patchPairs[offset + write] = Pair<uint, uint>(srcId, (uint)(tBegin + targetIdx));
				write++;
			}
		}
	}

	__global__ void NTQ_CountPatchPairsAll(
		DArray<int> counts,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		int patchCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> lp = shapePairs[tId];
		int shape0 = (int)lp.first;
		int shape1 = (int)lp.second;

		if (shape0 + 1 >= shape2PatchOffsets.size() || shape1 + 1 >= shape2PatchOffsets.size())
		{
			counts[tId] = 0;
			return;
		}

		int start0 = NLQ_ClampInt(shape2PatchOffsets[shape0], 0, patchCount);
		int end0 = NLQ_ClampInt(shape2PatchOffsets[shape0 + 1], 0, patchCount);
		int start1 = NLQ_ClampInt(shape2PatchOffsets[shape1], 0, patchCount);
		int end1 = NLQ_ClampInt(shape2PatchOffsets[shape1 + 1], 0, patchCount);

		int count0 = end0 - start0;
		int count1 = end1 - start1;
		if (count0 <= 0 || count1 <= 0)
		{
			counts[tId] = 0;
			return;
		}

		counts[tId] = count0 * count1;
	}

	__global__ void NTQ_SetPatchPairsAll(
		DArray<Pair<uint, uint>> patchPairs,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		DArray<int> prefix,
		DArray<int> counts,
		int patchCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> lp = shapePairs[tId];
		int shape0 = (int)lp.first;
		int shape1 = (int)lp.second;

		if (shape0 + 1 >= shape2PatchOffsets.size() || shape1 + 1 >= shape2PatchOffsets.size())
			return;

		int start0 = NLQ_ClampInt(shape2PatchOffsets[shape0], 0, patchCount);
		int end0 = NLQ_ClampInt(shape2PatchOffsets[shape0 + 1], 0, patchCount);
		int start1 = NLQ_ClampInt(shape2PatchOffsets[shape1], 0, patchCount);
		int end1 = NLQ_ClampInt(shape2PatchOffsets[shape1 + 1], 0, patchCount);

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;

		for (int p0 = start0; p0 < end0; ++p0)
		{
			for (int p1 = start1; p1 < end1; ++p1)
			{
				if (write < size && (offset + write) < patchPairs.size())
				{
					patchPairs[offset + write] = Pair<uint, uint>((uint)p0, (uint)p1);
					write++;
				}
			}
		}
	}

	// Calculate the number of triangle-lists to be generated for each patch pair.
	// For a pair of patches (P0, P1) with N0 and N1 triangles respectively:
	// We generate N0 lists (one for each triangle in P0 checking against all in P1)
	// and N1 lists (one for each triangle in P1 checking against all in P0).
	// Total size = N0 + N1.
	__global__ void NLQ_Narrow_BuildTriListCounts(
		DArray<int> listSizes,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patch2TriOffsets,
		int patchCount,
		int patchTriCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= patchPairs.size()) return;

		Pair<uint, uint> pp = patchPairs[tId];
		int patch0 = (int)pp.first;
		int patch1 = (int)pp.second;

		if (patch0 < 0 || patch0 >= patchCount || patch1 < 0 || patch1 >= patchCount)
		{
			listSizes[tId] = 0;
			return;
		}

		if (patch0 + 1 >= patch2TriOffsets.size() || patch1 + 1 >= patch2TriOffsets.size())
		{
			listSizes[tId] = 0;
			return;
		}

		// Calculate number of triangles in each patch
		int start0 = NLQ_ClampInt(patch2TriOffsets[patch0], 0, patchTriCount);
		int end0 = NLQ_ClampInt(patch2TriOffsets[patch0 + 1], 0, patchTriCount);
		int start1 = NLQ_ClampInt(patch2TriOffsets[patch1], 0, patchTriCount);
		int end1 = NLQ_ClampInt(patch2TriOffsets[patch1 + 1], 0, patchTriCount);

		int count0 = end0 - start0;
		int count1 = end1 - start1;

		// TODO: count may be zero for discrete elements
		if (count0 <= 0 || count1 <= 0)
		{
			listSizes[tId] = 0;
			return;
		}

		// Total work items for this patch pair
		listSizes[tId] = count0 + count1;
	}

	// Build contact lists for narrow phase.
	// For each patch pair (P0, P1), we create a list for every triangle in P0 and every triangle in P1.
	// To avoid duplicate checks (checking (A,B) and (B,A)), we enforce an ordering based on triangle IDs.
	// A pair (T_a, T_b) is only added to the list of T_a if T_a < T_b.
	__global__ void NLQ_Narrow_BuildTriContactLists(
		DArrayList<int> triContactList,
		DArray<int> triListTriIds,
		DArray<int> triListPairIds,
		DArray<int> triListSide,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patch2TriOffsets,
		DArray<int> patch2TriIndices,
		DArray<int> listOffsets,
		int patchCount,
		int triCount,
		int patchTriCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= patchPairs.size()) return;

		Pair<uint, uint> pp = patchPairs[tId];
		int patch0 = (int)pp.first;
		int patch1 = (int)pp.second;

		if (patch0 < 0 || patch0 >= patchCount || patch1 < 0 || patch1 >= patchCount)
			return;

		if (patch0 + 1 >= patch2TriOffsets.size() || patch1 + 1 >= patch2TriOffsets.size())
			return;

		int start0 = NLQ_ClampInt(patch2TriOffsets[patch0], 0, patchTriCount);
		int end0 = NLQ_ClampInt(patch2TriOffsets[patch0 + 1], 0, patchTriCount);
		int start1 = NLQ_ClampInt(patch2TriOffsets[patch1], 0, patchTriCount);
		int end1 = NLQ_ClampInt(patch2TriOffsets[patch1 + 1], 0, patchTriCount);

		int count0 = end0 - start0;
		int count1 = end1 - start1;
		if (count0 <= 0 || count1 <= 0)
			return;

		int base = listOffsets[tId];

		// Process triangles in Patch 0
		for (int i = 0; i < count0; ++i)
		{
			int listId = base + i;
			if (listId >= triContactList.size())
				break;

			int triId0 = patch2TriIndices[start0 + i];
			triListTriIds[listId] = triId0;
			triListPairIds[listId] = tId;
			triListSide[listId] = 0;

			List<int>& list = triContactList[listId];
			list.clear();

			if (triId0 < 0 || triId0 >= triCount)
				continue;

			for (int j = 0; j < count1; ++j)
			{
				int triId1 = patch2TriIndices[start1 + j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;
				// Only add if triId1 > triId0 to avoid duplicates and self-checks
				if (triId1 <= triId0)
					continue;
				// Precondition: host-side validation guarantees patch face count <= 32.
				list.insert(triId1);
			}
		}

		// Process triangles in Patch 1
		for (int j = 0; j < count1; ++j)
		{
			int listId = base + count0 + j;
			if (listId >= triContactList.size())
				break;

			int triId1 = patch2TriIndices[start1 + j];
			triListTriIds[listId] = triId1;
			triListPairIds[listId] = tId;
			triListSide[listId] = 1;

			List<int>& list = triContactList[listId];
			list.clear();

			if (triId1 < 0 || triId1 >= triCount)
				continue;

			for (int i = 0; i < count0; ++i)
			{
				int triId0 = patch2TriIndices[start0 + i];
				if (triId0 < 0 || triId0 >= triCount)
					continue;
				// Only add if triId0 > triId1 to avoid duplicates and self-checks
				if (triId0 <= triId1)
					continue;
				// Precondition: host-side validation guarantees patch face count <= 32.
				list.insert(triId0);
			}
		}
	}

		template<typename Real, typename Coord, typename Triangle>
		__global__ void NLQ_Narrow_WarpCount(
			DArray<int> counts,
			DArrayList<int> triContactList,
			DArray<int> triListTriIds,
			DArray<int> triListPairIds,
			DArray<int> triListSide,
			DArray<Pair<uint, uint>> patchPairs,
			DArray<uint> patch2Shape,
			DArray<Coord> vertices,
			DArray<Triangle> triangles,
			DArray<Mat3f> shapeRestR,
			DArray<Vec3f> shapeRestT,
			Real dHat,
			int triCount,
			bool verticesInRestWorld)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			int warpId = tId / 32; // representing the current triangle list index
			int lane = tId % 32;   // representing the candidate triangle list index
			if (warpId >= triContactList.size()) return;

			int triIdCurrent = triListTriIds[warpId];

			int side = triListSide[warpId];
			int pairId = triListPairIds[warpId];
			if (triIdCurrent < 0 || triIdCurrent >= triCount)
			{
				if (lane == 0)
					counts[warpId] = 0;
				return;
			}
			if (pairId < 0 || pairId >= patchPairs.size())
			{
				if (lane == 0)
					counts[warpId] = 0;
				return;
			}

			Pair<uint, uint> pp = patchPairs[pairId];
			int patch0 = (int)pp.first;
			int patch1 = (int)pp.second;
			int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
			int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

			Mat3f R0 = Mat3f::identityMatrix();
			Vec3f tRel0 = Vec3f(Real(0));
			Mat3f R1 = Mat3f::identityMatrix();
			Vec3f tRel1 = Vec3f(Real(0));
			if (verticesInRestWorld)
			{
				if (shape0 < 0 || shape1 < 0
					|| shape0 >= shapeRestR.size() || shape0 >= shapeRestT.size()
					|| shape1 >= shapeRestR.size() || shape1 >= shapeRestT.size())
				{
					if (lane == 0)
						counts[warpId] = 0;
					return;
				}
				R0 = shapeRestR[shape0];
				tRel0 = shapeRestT[shape0];
				R1 = shapeRestR[shape1];
				tRel1 = shapeRestT[shape1];
			}

			List<int>& list = triContactList[warpId];
			int candSize = (int)list.size();

		int laneCount = 0;
		if (lane < candSize)
		{
			int triIdCandidate = list[lane];
			if (triIdCandidate >= 0 && triIdCandidate < triCount)
			{
					int triId0 = side == 0 ? triIdCurrent : triIdCandidate;
					int triId1 = side == 0 ? triIdCandidate : triIdCurrent;

					Triangle tri0 = triangles[triId0];
					Coord p00 = vertices[tri0[0]];
					Coord p01 = vertices[tri0[1]];
					Coord p02 = vertices[tri0[2]];
					if (verticesInRestWorld)
					{
						p00 = R0 * p00 + tRel0;
						p01 = R0 * p01 + tRel0;
						p02 = R0 * p02 + tRel0;
					}
					TTriangle3D<Real> t0(p00, p01, p02);

					Triangle tri1 = triangles[triId1];
					Coord p10 = vertices[tri1[0]];
					Coord p11 = vertices[tri1[1]];
					Coord p12 = vertices[tri1[2]];
					if (verticesInRestWorld)
					{
						p10 = R1 * p10 + tRel1;
						p11 = R1 * p11 + tRel1;
						p12 = R1 * p12 + tRel1;
					}
					TTriangle3D<Real> t1(p10, p11, p12);

					TManifold<Real> manifold;
					CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);
					// CollisionDetection<Real>::request(manifold, t0, t1, 0.0f, 0.0f);
				laneCount = manifold.contactCount;
			}
		}

		int total = NLQ_WarpReduceSum(laneCount);
		if (lane == 0)
			counts[warpId] = total;
	}

		template<typename Real, typename Coord, typename Triangle, typename ContactPair>
		__global__ void NLQ_Narrow_WarpSet(
			DArray<ContactPair> contacts,
			DArrayList<int> triContactList,
			DArray<int> triListTriIds,
			DArray<int> triListPairIds,
			DArray<int> triListSide,
			DArray<Pair<uint, uint>> patchPairs,
			DArray<uint> patch2Shape,
			DArray<Coord> vertices,
			DArray<Triangle> triangles,
			DArray<Mat3f> shapeRestR,
			DArray<Vec3f> shapeRestT,
			DArray<int> prefix,
			DArray<int> counts,
			DArray<int> shape2RigidBody,
			Real dHat,
			int triCount,
			bool verticesInRestWorld)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			int warpId = tId / 32; // representing 
			int lane = tId % 32;   // representing 
			if (warpId >= triContactList.size()) return;

		int triIdCurrent = triListTriIds[warpId];
		int side = triListSide[warpId];
		int pairId = triListPairIds[warpId];
		if (triIdCurrent < 0 || triIdCurrent >= triCount)
			return;
		if (pairId < 0 || pairId >= patchPairs.size())
			return;

		Pair<uint, uint> pp = patchPairs[pairId];
		int patch0 = (int)pp.first;
		int patch1 = (int)pp.second;

			int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
			int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

			int bodyId0 = shape2RigidBody[shape0];
			int bodyId1 = shape2RigidBody[shape1];

			Mat3f R0 = Mat3f::identityMatrix();
			Vec3f tRel0 = Vec3f(Real(0));
			Mat3f R1 = Mat3f::identityMatrix();
			Vec3f tRel1 = Vec3f(Real(0));
			if (verticesInRestWorld)
			{
				if (shape0 < 0 || shape1 < 0
					|| shape0 >= shapeRestR.size() || shape0 >= shapeRestT.size()
					|| shape1 >= shapeRestR.size() || shape1 >= shapeRestT.size())
				{
					return;
				}
				R0 = shapeRestR[shape0];
				tRel0 = shapeRestT[shape0];
				R1 = shapeRestR[shape1];
				tRel1 = shapeRestT[shape1];
			}

			List<int>& list = triContactList[warpId];
			int candSize = (int)list.size();

		int triIdCandidate = -1;
		int laneCount = 0;
		TManifold<Real> manifold;

		if (lane < candSize)
		{
			triIdCandidate = list[lane];
			if (triIdCandidate >= 0 && triIdCandidate < triCount)
			{
					int triId0 = side == 0 ? triIdCurrent : triIdCandidate;
					int triId1 = side == 0 ? triIdCandidate : triIdCurrent;

					Triangle tri0 = triangles[triId0];
					Coord p00 = vertices[tri0[0]];
					Coord p01 = vertices[tri0[1]];
					Coord p02 = vertices[tri0[2]];
					if (verticesInRestWorld)
					{
						p00 = R0 * p00 + tRel0;
						p01 = R0 * p01 + tRel0;
						p02 = R0 * p02 + tRel0;
					}
					TTriangle3D<Real> t0(p00, p01, p02);

					Triangle tri1 = triangles[triId1];
					Coord p10 = vertices[tri1[0]];
					Coord p11 = vertices[tri1[1]];
					Coord p12 = vertices[tri1[2]];
					if (verticesInRestWorld)
					{
						p10 = R1 * p10 + tRel1;
						p11 = R1 * p11 + tRel1;
						p12 = R1 * p12 + tRel1;
					}
					TTriangle3D<Real> t1(p10, p11, p12);

					CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);
					laneCount = manifold.contactCount;
				}
		}

		int laneOffset = NLQ_WarpExclusivePrefix(laneCount, lane);
		int writeBase = prefix[warpId] + laneOffset;
		int writeLimit = prefix[warpId] + counts[warpId];

		if (lane < candSize && laneCount > 0)
		{
			int triId0 = side == 0 ? triIdCurrent : triIdCandidate;
			int triId1 = side == 0 ? triIdCandidate : triIdCurrent;

			for (int n = 0; n < laneCount; ++n)
			{
				int outIdx = writeBase + n;
				if (outIdx >= writeLimit || outIdx >= contacts.size())
					break;

				ContactPair cp;
				cp.bodyId1 = bodyId0;
				cp.bodyId2 = bodyId1;
				cp.localId1 = triId0;
				cp.localId2 = triId1;
				cp.pos1 = manifold.contacts[n].position;
				cp.pos2 = manifold.contacts[n].position;
				// cp.pos1 = manifold.contacts[n].position + dHat * manifold.normal;
				// cp.pos2 = manifold.contacts[n].position + dHat * manifold.normal;
				cp.normal1 = -manifold.normal;
				cp.normal2 = manifold.normal;
				cp.contactType = ContactType::CT_NONPENETRATION;
				cp.interpenetration = -manifold.contacts[n].penetration;
				// cp.interpenetration = -manifold.contacts[n].penetration - 2 * dHat;


				contacts[outIdx] = cp;
			}
		}
	}

	template<typename TDataType>
	NeighborTriMeshQuery<TDataType>::NeighborTriMeshQuery()
		: NeighborElementQuery<TDataType>()
	{
		this->inAdjacentShapes()->tagOptional(true);
		// this->inShape2PatchCounts()->tagOptional(true);
		// this->inShape2RigidBodyIds()->tagOptional(true);
		this->inShape2ElementIds()->tagOptional(true);
		this->inShapeBVHs()->tagOptional(true);

		this->varGridSizeLimit()->setValue(Real(0.01));
		this->varDHead()->setValue(Real(0.001));
	}

	template<typename TDataType>
	NeighborTriMeshQuery<TDataType>::~NeighborTriMeshQuery()
	{
		auto& shapeBVHs = this->inShapeBVHs()->constDataPtr();
		if (shapeBVHs != nullptr)
		{
			for (const auto& bvh : *shapeBVHs)
			{
				if (bvh)
				{
					bvh->release();
				}
			}
		}

		clearWorkspace();
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::setStaticShape2PatchOffsets(const std::vector<int>& offsets)
	{
		mShape2PatchOffsets.clear();
		mStaticShape2PatchOffsetsReady = false;
		mPatch2Shape.clear();
		mStaticPatch2ShapeReady = false;
		mTargetBVHs.clear();
		mTargetBVHValid.clear();
		mStaticTargetBVHCacheReady = false;

		if (offsets.size() < 2)
		{
			printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets: size=%zu (<2).\n", offsets.size());
			return false;
		}
		if (offsets[0] != 0)
		{
			printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets: offsets[0]=%d (expected 0).\n", offsets[0]);
			return false;
		}
		for (size_t i = 0; i < offsets.size(); ++i)
		{
			if (offsets[i] < 0)
			{
				printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets: offsets[%zu]=%d (<0).\n", i, offsets[i]);
				return false;
			}
			if (i > 0 && offsets[i] < offsets[i - 1])
			{
				printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets: non-monotonic at [%zu]=%d < [%zu]=%d.\n",
					i,
					offsets[i],
					i - 1,
					offsets[i - 1]);
				return false;
			}
		}

		mShape2PatchOffsets.assign(offsets);
		mStaticShape2PatchOffsetsReady = true;
		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::setStaticPatch2Shape(const std::vector<uint>& patch2Shape)
	{
		mPatch2Shape.clear();
		mStaticPatch2ShapeReady = false;

		if (!mStaticShape2PatchOffsetsReady || mShape2PatchOffsets.size() < 2)
		{
			printf("[NeighborTriMeshQuery] Invalid setStaticPatch2Shape call: Shape2PatchOffsets not ready.\n");
			return false;
		}

		const int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if (shapeCount <= 0)
		{
			printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets for Patch2Shape: shapeCount=%d.\n", shapeCount);
			return false;
		}

		CArray<int> hShape2PatchOffsets;
		hShape2PatchOffsets.assign(mShape2PatchOffsets);
		const int expectedPatchCount = hShape2PatchOffsets[hShape2PatchOffsets.size() - 1];
		if (expectedPatchCount < 0)
		{
			printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets for Patch2Shape: back=%d (<0).\n",
				expectedPatchCount);
			return false;
		}

		if ((int)patch2Shape.size() != expectedPatchCount)
		{
			printf("[NeighborTriMeshQuery] Invalid Patch2Shape size: expected=%d, actual=%zu.\n",
				expectedPatchCount,
				patch2Shape.size());
			return false;
		}

		for (int patchId = 0; patchId < expectedPatchCount; ++patchId)
		{
			const uint shapeId = patch2Shape[patchId];
			if (shapeId >= (uint)shapeCount)
			{
				printf("[NeighborTriMeshQuery] Invalid Patch2Shape[%d]=%u (shapeCount=%d).\n",
					patchId,
					(unsigned int)shapeId,
					shapeCount);
				return false;
			}
		}

		mPatch2Shape.assign(patch2Shape);
		mStaticPatch2ShapeReady = true;
		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::setStaticTargetBVHCache(const ShapeBVHList& shapeBVHs)
	{
		mTargetBVHs.clear();
		mTargetBVHValid.clear();
		mStaticTargetBVHCacheReady = false;

		if (!mStaticShape2PatchOffsetsReady || mShape2PatchOffsets.size() < 2)
		{
			printf("[NeighborTriMeshQuery] Invalid setStaticTargetBVHCache call: Shape2PatchOffsets not ready.\n");
			return false;
		}

		const int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if (shapeCount <= 0)
		{
			printf("[NeighborTriMeshQuery] Invalid Shape2PatchOffsets for TargetBVH cache: shapeCount=%d.\n", shapeCount);
			return false;
		}

		CArray<LinearBVH<TDataType>> hTargetBVHs;
		hTargetBVHs.resize(shapeCount);
		CArray<int> hTargetBVHValid;
		hTargetBVHValid.assign((uint)shapeCount, 0);

		int copyCount = shapeCount;
		if (copyCount > (int)shapeBVHs.size())
			copyCount = (int)shapeBVHs.size();

		for (int i = 0; i < copyCount; ++i)
		{
			auto& bvh = shapeBVHs[i];
			if (!bvh)
				continue;

			hTargetBVHs[i] = *bvh;
			int nodeCount = (int)bvh->getSortedAABBs().size();
			hTargetBVHValid[i] = (nodeCount > 0 && (nodeCount % 2) == 1) ? 1 : 0;
		}

		if (mTargetBVHs.size() != (uint)shapeCount)
			mTargetBVHs.resize(shapeCount);
		if (mTargetBVHValid.size() != (uint)shapeCount)
			mTargetBVHValid.resize(shapeCount);

		mTargetBVHs.assign(hTargetBVHs);
		mTargetBVHValid.assign(hTargetBVHValid);
		mStaticTargetBVHCacheReady = true;
		return true;
	}

	// template<typename TDataType>
	// bool NeighborTriMeshQuery<TDataType>::updateShape2RigidBodyIds(int shapeCount)
	// {
	// 	if (!this->inShape2RigidBodyIds()->isEmpty())
	// 	{
	// 		auto& ids = this->inShape2RigidBodyIds()->getData();
	// 		if ((int)ids.size() == shapeCount)
	// 		{
	// 			mShape2RigidBodyIds.assign(ids);
	// 			if (!mMappingReady)
	// 			{
	// 				printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping ready (shapeCount=%d, source=input).\n", shapeCount);
	// 			}
	// 			mMappingReady = true;
	// 			mWarnedEmptyMapping = false;
	// 			return true;
	// 		}
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyIds size mismatch (shapeCount=%d, inputSize=%u), fallback to topology mapping.\n",
	// 				shapeCount,
	// 				(unsigned int)ids.size());
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 	}

	// 	auto topo = this->inDiscreteElements()->getDataPtr();
	// 	if (topo == nullptr)
	// 	{
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (topology unavailable, shapeCount=%d, mappingSize=0), skip this frame.\n", shapeCount);
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 		mMappingReady = false;
	// 		return false;
	// 	}

	// 	uint totalSize = topo->totalSize();
	// 	if ((uint)shapeCount != totalSize)
	// 	{
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, totalSize=%u), skip this frame.\n",
	// 				shapeCount,
	// 				totalSize);
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 		mMappingReady = false;
	// 		return false;
	// 	}

	// 	auto& mapping = topo->shape2RigidBodyMapping();
	// 	uint mappingSize = mapping.size();
	// 	if (mappingSize == 0)
	// 	{
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
	// 				shapeCount,
	// 				mappingSize);
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 		mMappingReady = false;
	// 		return false;
	// 	}
	// 	if (mappingSize < totalSize)
	// 	{
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, totalSize=%u, mappingSize=%u), skip this frame.\n",
	// 				shapeCount,
	// 				totalSize,
	// 				mappingSize);
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 		mMappingReady = false;
	// 		return false;
	// 	}
	// 	// Move ouside now
	// 	std::vector<int> shape2RigidBodyIds;
	// 	if (!NLQ_BuildShape2RigidBodyIds(mapping, shapeCount, shape2RigidBodyIds))
	// 	{
	// 		if (!mWarnedEmptyMapping)
	// 		{
	// 			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
	// 				shapeCount,
	// 				mappingSize);
	// 			mWarnedEmptyMapping = true;
	// 		}
	// 		mMappingReady = false;
	// 		return false;
	// 	}

	// 	mShape2RigidBodyIds.assign(shape2RigidBodyIds);

	// 	if (!mMappingReady)
	// 	{
	// 		printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping ready (shapeCount=%d, mappingSize=%u).\n",
	// 			shapeCount,
	// 			mappingSize);
	// 	}
	// 	mMappingReady = true;
	// 	mWarnedEmptyMapping = false;

	// 	return true;
	// }

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::buildPatchPairsFromContactList(int shapeCount, int patchCount)
	{
		auto& shapePairs = this->outPotentialShapePairs()->getData();
		if (shapePairs.size() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (shapeCount <= 0 || patchCount <= 0 || mShape2PatchOffsets.size() != (uint)(shapeCount + 1))
		{
			if (!mWarnedEmptyPatchMapping)
			{
				printf("[NeighborTriMeshQuery] Patch mapping not ready yet (shapeCount=%d, patchCount=%d), skip this frame.\n",
					shapeCount,
					patchCount);
				mWarnedEmptyPatchMapping = true;
			}
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		mWarnedEmptyPatchMapping = false;

		DArray<int> pairCount;
		pairCount.resize(shapePairs.size());
		pairCount.reset();

		cuExecute(shapePairs.size(),
			NTQ_CountPatchPairsAll,
			pairCount,
			shapePairs,
			mShape2PatchOffsets,
			patchCount);

		int total = mReduce.accumulate(pairCount.begin(), pairCount.size());
		if (total <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			pairCount.clear();
			return false;
		}

		DArray<int> pairCountCpy;
		pairCountCpy.assign(pairCount);
		mScan.exclusive(pairCount, true);

		this->outPotentialPatchPairs()->resize(total);

		cuExecute(shapePairs.size(),
			NTQ_SetPatchPairsAll,
			this->outPotentialPatchPairs()->getData(),
			shapePairs,
			mShape2PatchOffsets,
			pairCount,
			pairCountCpy,
			patchCount);

		pairCountCpy.clear();
		pairCount.clear();

		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::updatePatchFaceLimitState(int patchCount)
	{
		// Host-side guard for kernels that assume "one patch <= one warp".
		// If a patch has more than NMQ_MaxPatchFaces triangles, we skip this frame safely.
		auto& patch2TriOffsets = this->inPatch2TriOffsets()->getData();
		auto& patch2TriIndices = this->inPatch2TriIndices()->getData();

		const uint offsetsSize = patch2TriOffsets.size();
		const uint indicesSize = patch2TriIndices.size();

		// Cache by sizes/counts to avoid CPU readback every frame when inputs are unchanged.
		const bool needRebuild = (!mPatchFaceLimitReady)
			|| (mCachedPatchCount != patchCount)
			|| (mCachedPatch2TriOffsetsSize != offsetsSize)
			|| (mCachedPatch2TriIndicesSize != indicesSize);

		if (!needRebuild)
			return mPatchFaceLimitValid;

		mPatchFaceLimitReady = true;
		mCachedPatchCount = patchCount;
		mCachedPatch2TriOffsetsSize = offsetsSize;
		mCachedPatch2TriIndicesSize = indicesSize;
		mCachedMaxPatchFaces = 0;
		mPatchFaceLimitValid = false;

		if (patchCount <= 0 || offsetsSize < (uint)(patchCount + 1))
			return false;

		CArray<int> hPatch2TriOffsets;
		hPatch2TriOffsets.assign(patch2TriOffsets);
		if (hPatch2TriOffsets.size() < (uint)(patchCount + 1))
			return false;

		int maxPatchFaces = 0;
		const int patchTriCount = (int)indicesSize;
		auto clampHost = [](int v, int lo, int hi) {
			return v < lo ? lo : (v > hi ? hi : v);
		};
		for (int p = 0; p < patchCount; ++p)
		{
			int begin = hPatch2TriOffsets[p];
			int end = hPatch2TriOffsets[p + 1];
			begin = clampHost(begin, 0, patchTriCount);
			end = clampHost(end, 0, patchTriCount);
			if (end < begin)
			{
				int tmp = begin;
				begin = end;
				end = tmp;
			}
			int count = end - begin;
			if (count > maxPatchFaces)
				maxPatchFaces = count;
		}

		mCachedMaxPatchFaces = maxPatchFaces;
		mPatchFaceLimitValid = (maxPatchFaces <= NMQ_MaxPatchFaces);
		if (!mPatchFaceLimitValid && !mWarnedPatchFaceLimit)
		{
			printf("[NeighborTriMeshQuery] patch face count overflow (max=%d, limit=%d), skip this frame.\n",
				mCachedMaxPatchFaces,
				NMQ_MaxPatchFaces);
			mWarnedPatchFaceLimit = true;
		}

		return mPatchFaceLimitValid;
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::ensureMiddleWorkspace(int totalSource)
	{
		// Allocate per-source-patch temporaries used by middle phase BVH query and compaction.
		if (totalSource <= 0)
			return;

		if (mMiddleLocalBroadPhaseCounter.size() != (uint)totalSource)
			mMiddleLocalBroadPhaseCounter.resize(totalSource);
		if (mMiddleContactCount.size() != (uint)totalSource)
			mMiddleContactCount.resize(totalSource);
		if (mMiddleContactCountCpy.size() != (uint)totalSource)
			mMiddleContactCountCpy.resize(totalSource);
		if (mMiddleContactList.size() != (uint)totalSource)
			mMiddleContactList.resize((uint)totalSource);
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::ensureNarrowWorkspace(int totalTriLists)
	{
		// Allocate per-triangle-list temporaries used by narrow phase.
		if (totalTriLists <= 0)
			return;

		if (mNarrowTriListOffsets.size() != (uint)totalTriLists)
			mNarrowTriListOffsets.resize(totalTriLists);
		if (mNarrowTriListTriIds.size() != (uint)totalTriLists)
			mNarrowTriListTriIds.resize(totalTriLists);
		if (mNarrowTriListPairIds.size() != (uint)totalTriLists)
			mNarrowTriListPairIds.resize(totalTriLists);
		if (mNarrowTriListSide.size() != (uint)totalTriLists)
			mNarrowTriListSide.resize(totalTriLists);
		if (mNarrowTriContactCounts.size() != (uint)totalTriLists)
			mNarrowTriContactCounts.resize(totalTriLists);
		if (mNarrowTriContactOffsets.size() != (uint)totalTriLists)
			mNarrowTriContactOffsets.resize(totalTriLists);
		if (mNarrowTriContactList.size() != (uint)totalTriLists)
			// Each list stores candidate triangles from the opposite patch; bounded by warp-sized patch face limit.
			mNarrowTriContactList.resize((uint)totalTriLists, NMQ_MaxPatchFaces);
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::clearWorkspace()
	{
		mMiddleLocalBroadPhaseCounter.clear();
		mMiddleContactList.clear();
		mMiddleContactCount.clear();
		mMiddleContactCountCpy.clear();

		mNarrowTriListSizes.clear();
		mNarrowTriListOffsets.clear();
		mNarrowTriContactList.clear();
		mNarrowTriListTriIds.clear();
		mNarrowTriListPairIds.clear();
		mNarrowTriListSide.clear();
		mNarrowTriContactCounts.clear();
		mNarrowTriContactOffsets.clear();
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::compute()
	{
		// Pipeline overview:
		// 1) broadPhase  : shape-level culling
		// 2) middlePhase : patch-level culling
		// 3) narrowPhase : triangle-level contact generation
		CTimer timer;
		timer.start();

		auto finishTiming = [&]() {
			timer.stop();
			std::cout << "[NeighborTriMeshQuery] compute time: " << timer.getElapsedTime() << " ms" << std::endl;
		};

		mUseBroadPhasePatchPairs = false;
		// Ensure output buffers are available before early exits.
		if (this->outPotentialShapePairs()->isEmpty())
			this->outPotentialShapePairs()->allocate();
		if (this->outPotentialPatchPairs()->isEmpty())
			this->outPotentialPatchPairs()->allocate();
		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();
		if (this->outPotentialTriSet()->isEmpty())
			this->outPotentialTriSet()->allocate();

		// Validate all required inputs up front; on failure clear outputs and return early.
		if (this->inPatchAABBs()->isEmpty()
			|| this->inPatch2TriOffsets()->isEmpty()
			|| this->inPatch2TriIndices()->isEmpty()
			|| this->inCenter()->isEmpty()
			|| this->inRotationMatrix()->isEmpty()
			|| this->inRestShapeCenter()->isEmpty()
			|| this->inRestShapeRotation()->isEmpty()
			|| this->inTriangleSet()->isEmpty()
			|| this->inShape2ElementIdsDense()->size() <= 0
			|| this->inShape2RigidBodyIds()->size() <= 0
			|| !mStaticShape2PatchOffsetsReady
			|| !mStaticPatch2ShapeReady)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] Missing input data.\n");
			finishTiming();
			return;
		}

		int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if ((int)this->inShape2ElementIdsDense()->size() != shapeCount
			|| (int)this->inShape2RigidBodyIds()->size() != shapeCount)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] Mapping size mismatch (shapeCount=%d, dense=%u, rigid=%u).\n",
				shapeCount,
				(unsigned int)this->inShape2ElementIdsDense()->size(),
				(unsigned int)this->inShape2RigidBodyIds()->size());
			finishTiming();
			return;
		}

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patchCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			finishTiming();
			return;
		}

		if (mPatch2Shape.size() != (uint)patchCount)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] Patch2Shape size mismatch (patchCount=%d, cached=%u).\n",
				patchCount,
				(unsigned int)mPatch2Shape.size());
			finishTiming();
			return;
		}

		if (!mStaticTargetBVHCacheReady
			|| mTargetBVHs.size() != (uint)shapeCount
			|| mTargetBVHValid.size() != (uint)shapeCount)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] TargetBVH cache mismatch (ready=%d, shapeCount=%d, bvh=%u, valid=%u).\n",
				mStaticTargetBVHCacheReady ? 1 : 0,
				shapeCount,
				(unsigned int)mTargetBVHs.size(),
				(unsigned int)mTargetBVHValid.size());
			finishTiming();
			return;
		}

		// BroadPhase: shape AABB overlap -> unique (i < j) shape pairs.
		if (!broadPhase())
		{
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			// printf("[NeighborTriMeshQuery] BroadPhase failed.\n");
			finishTiming();
			return;
		}

		// MiddlePhase: shape pairs + patch CSR -> candidate patch pairs.
		if (!middlePhase())
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			// printf("[NeighborTriMeshQuery] MiddlePhase failed.\n");
			finishTiming();
			return;
		}

		// NarrowPhase: patch pairs + triangle CSR -> final contact manifolds.
		narrowPhase();
		
		finishTiming();
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::broadPhase()
	{
		// printf("[NeighborTriMeshQuery] BroadPhase started.\n");
		// Build per-shape world AABBs, query broad-phase accelerator, then compact valid shape pairs.
		NewTimer broadTimer;
		broadTimer.start();

		NewTimer broadTimer1;
		broadTimer1.start();

		NewTimer broadTimer2;
		broadTimer2.start();

		NewTimer broadTimer3;
		broadTimer3.start();


		auto inTopo = this->inDiscreteElements()->getDataPtr();
		if (inTopo == nullptr)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] DiscreteElements missing.\n");
			return false;
		}

		int shapeCount = (int)mShape2PatchOffsets.size() - 1;

		auto& shape2ElementIds = this->inShape2ElementIdsDense()->getData();
		if (shape2ElementIds.size() != (uint)shapeCount)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] Shape2ElementIdsDense size mismatch.\n");
			return false;
		}

		int elementCount = inTopo->totalSize();
		if (elementCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] DiscreteElements size mismatch.\n");
			return false;
		}

		if (this->mQueriedAABB.size() != (uint)shapeCount)
			this->mQueriedAABB.resize(shapeCount);

		if (this->mQueryAABB.size() != (uint)shapeCount)
			this->mQueryAABB.resize(shapeCount);
		ElementOffset elementOffset = inTopo->calculateElementOffset();
		Real dHat = this->varDHead()->getValue();

		auto& boxInGlobal = inTopo->boxesInGlobal();
		auto& sphereInGlobal = inTopo->spheresInGlobal();
		auto& tetInGlobal = inTopo->tetsInGlobal();
		auto& capsuleInGlobal = inTopo->capsulesInGlobal();
		auto& triangleInGlobal = inTopo->trianglesInGlobal();

		// Convert shape -> element id to a conservative AABB used for broad-phase culling.
		cuExecute((uint)shapeCount,
			NTQ_SetupAABBFromElementIds,
			this->mQueriedAABB,
			shape2ElementIds,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capsuleInGlobal,
			triangleInGlobal,
			elementOffset,
			dHat);
		if (!NMQ_CheckCuda("NTQ_SetupAABBFromElementIds"))
			return false;

		broadTimer1.stop();
		// std::cout << "[NeighborTriMeshQuery] compute broad phase time 1: " << broadTimer1.elapsedMilliseconds() << " ms" << std::endl;

		this->mQueryAABB.assign(this->mQueriedAABB);

		this->mBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
		this->mBroadPhaseCD->varSelfCollision()->setValue(true);

		this->mBroadPhaseCD->inSource()->assign(this->mQueryAABB);
		this->mBroadPhaseCD->inTarget()->assign(this->mQueriedAABB);
		auto type = this->varSpatial()->getDataPtr()->currentKey();
		switch (type)
		{
		case Spatial::BVH:
			this->mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
			// printf("[NeighborTriMeshQuery] BroadPhase acceleration: BVH\n");
			break;
		case Spatial::OCTREE:
			this->mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
			printf("[NeighborTriMeshQuery] BroadPhase acceleration: Octree\n");
			break;
		default:
			printf("[NeighborTriMeshQuery] BroadPhase acceleration: Unknown key=%d\n", (int)type);
			break;
		}

		this->mBroadPhaseCD->update();

		auto& contactList = this->mBroadPhaseCD->outContactList()->getData();
		if (contactList.elementSize() == 0)
		{
			this->outPotentialShapePairs()->resize(0);
			return false;
		}

		DArray<int> pairCount;
		pairCount.resize(contactList.size());
		pairCount.reset();

		DArray<int> pairCountCpy;

		bool useAdj = this->varEnableAdjacentFilter()->getValue() && !this->inAdjacentShapes()->isEmpty();
		DArrayList<int> dummyAdj;
		auto& adj = useAdj ? this->inAdjacentShapes()->getData() : dummyAdj;

		// Count valid neighbors per shape, then use scan+scatter to emit compact (shapeA, shapeB) pairs.
		cuExecute(contactList.size(),
			NLQ_CountShapePairs,
			pairCount,
			contactList,
			adj,
			useAdj,
			shapeCount);

		broadTimer2.stop();
		// std::cout << "[NeighborTriMeshQuery] compute broad phase time 1: " << broadTimer2.elapsedMilliseconds() << " ms" << std::endl;

		int total = mReduce.accumulate(pairCount.begin(), pairCount.size());
		if (total <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			pairCount.clear();
			pairCountCpy.clear();
			return false;
		}

		pairCountCpy.assign(pairCount);
		mScan.exclusive(pairCount, true);

		this->outPotentialShapePairs()->resize(total);

		cuExecute(contactList.size(),
			NLQ_SetShapePairs,
			this->outPotentialShapePairs()->getData(),
			contactList,
			pairCount,       // prefix sum of pair count
			pairCountCpy,    // pair count of every shape's contact
			adj,
			useAdj,
			shapeCount);

		broadTimer3.stop();
		// std::cout << "[NeighborTriMeshQuery] compute broad phase time 2: " << broadTimer3.elapsedMilliseconds() << " ms" << std::endl;

		pairCountCpy.clear();
		pairCount.clear();

		// std::cout << "[NeighborTriMeshQuery] broadPhase found " << total << " shape pairs." << std::endl;
		broadTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute broad phase time: " << broadTimer.elapsedMilliseconds() << " ms" << std::endl;
		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::middlePhase()
	{
		// printf("[NeighborTriMeshQuery] MiddlePhase started.\n");
		// Convert shape pairs to patch pairs:
		// - group by target shape
		// - flatten source shape patches
		// - query target patch BVH in target-rest space
		// - compact overlap pairs
		NewTimer middleTimer;
		middleTimer.start();

		// Get potential shape pairs from broad phase
		auto& shapePairs = this->outPotentialShapePairs()->getData();
		if (shapePairs.size() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		// Get all Patch AABBs in rest-world space
		auto& patchAabbs = this->inPatchAABBs()->getData();
		int patchCount = (int)patchAabbs.size();
		if (patchCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		// Patch -> triangle mapping and triangle set (current world coordinates, same as narrowPhase)
		auto& patch2TriOffsets = this->inPatch2TriOffsets()->getData();
		auto& patch2TriIndices = this->inPatch2TriIndices()->getData();
		if (patch2TriOffsets.size() < (uint)(patchCount + 1) || patch2TriIndices.size() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}
		auto& vertices = ts->getPoints();
		auto& triIndices = ts->triangleIndices();
		int triCount = (int)triIndices.size();
		int vertexCount = (int)vertices.size();
		if (triCount <= 0 || vertexCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}
		int patchTriCount = (int)patch2TriIndices.size();
		if (!updatePatchFaceLimitState(patchCount))
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if (shapeCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		// Compute per-shape AABBs relative transforms from rest-world space to current-world space 
		if (mShapeRestR.size() != (uint)shapeCount)
			mShapeRestR.resize(shapeCount);
		if (mShapeRestT.size() != (uint)shapeCount)
			mShapeRestT.resize(shapeCount);
		// Launch kernel to compute per-shape rest transforms.
		cuExecute((uint)shapeCount,
			NLQ_ComputeShapeRestTransforms,
			mShapeRestR,
			mShapeRestT,
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData(),
			this->inRestShapeCenter()->getData(),
			this->inRestShapeRotation()->getData(),
			this->inShape2RigidBodyIds()->getData());

		// Patch AABBs updated
		// if (mTouchedShapeFlags.size() != (uint)shapeCount)
		// 	mTouchedShapeFlags.resize(shapeCount);
		// mTouchedShapeFlags.reset();

		// // Calculate how many times each shape is touched
		// cuExecute(shapePairs.size(),
		// 	NLQ_MarkTouchedShapesFromPairs,
		// 	mTouchedShapeFlags,
		// 	shapePairs,
		// 	shapeCount);
		// // Accumulate touched shape count
		// int touchedShapeCount = mReduce.accumulate(mTouchedShapeFlags.begin(), mTouchedShapeFlags.size());
		// if (touchedShapeCount > 0)
		// {
		// 	if (mTouchedShapeOffsets.size() != (uint)shapeCount)
		// 		mTouchedShapeOffsets.resize(shapeCount);
		// 	// Exclusive scan to build touched shape offsets
		// 	mTouchedShapeOffsets.assign(mTouchedShapeFlags);
		// 	mScan.exclusive(mTouchedShapeOffsets, true);

		// 	if (mTouchedShapeIds.size() != (uint)touchedShapeCount)
		// 		mTouchedShapeIds.resize(touchedShapeCount);
		// 	// Compact touched shapes
		// 	cuExecute((uint)shapeCount,
		// 		NLQ_CompactTouchedShapes,
		// 		mTouchedShapeIds,
		// 		mTouchedShapeFlags,
		// 		mTouchedShapeOffsets,
		// 		shapeCount);
		// 	// Update only touched shapes' patch AABBs
		// 	cuExecute((uint)touchedShapeCount,
		// 		NLQ_UpdatePatchAabbsForTouchedShapes,
		// 		mPatchAabbsWorld,
		// 		patchAabbs,
		// 		mShape2PatchOffsets,
		// 		mTouchedShapeIds,
		// 		this->inCenter()->getData(),
		// 		this->inRotationMatrix()->getData(),
		// 		mShape2RigidBodyIds);
		// }

		if (mTargetShapeCounts.size() != (uint)shapeCount)
			mTargetShapeCounts.resize(shapeCount);
		mTargetShapeCounts.reset();

		int shapePairCount = (int)shapePairs.size();
		if (shapePairCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (mSortedPairTargets.size() != (uint)shapePairCount)
			mSortedPairTargets.resize(shapePairCount);
		if (mTarget2SourceShapes.size() != (uint)shapePairCount)
			mTarget2SourceShapes.resize(shapePairCount);

		cuExecute((uint)shapePairCount,
			NLQ_ExtractSourceTargetFromShapePairs,
			mSortedPairTargets,
			mTarget2SourceShapes,
			shapePairs,
			shapeCount);
		// if (!NMQ_CheckCuda("NLQ_ExtractSourceTargetFromShapePairs"))
		// 	return false;

		thrust::sort_by_key(
			thrust::device,
			mSortedPairTargets.begin(),
			mSortedPairTargets.begin() + mSortedPairTargets.size(),
			mTarget2SourceShapes.begin());
		// if (!NMQ_CheckCuda("sort_by_target"))
		// 	return false;

		cuExecute((uint)shapePairCount,
			NLQ_CountTargetShapesFromSortedKeys,
			mTargetShapeCounts,
			mSortedPairTargets,
			shapeCount);
		// if (!NMQ_CheckCuda("NLQ_CountTargetShapesFromSortedKeys"))
		// 	return false;
		
		// Target shape counts computed
		// Exclusive scan to build target shape offsets
		if (mTargetShapeOffsets.size() != (uint)shapeCount)
			mTargetShapeOffsets.resize(shapeCount);
		mTargetShapeOffsets.assign(mTargetShapeCounts);
		mScan.exclusive(mTargetShapeOffsets, true);

		int groupCountAll = mReduce.accumulate(mTargetShapeCounts.begin(), mTargetShapeCounts.size()); // number of valid (target, source-shape) groups
		if (groupCountAll <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (mGroup2PatchCounts.size() != (uint)groupCountAll)
			mGroup2PatchCounts.resize(groupCountAll);
		mGroup2PatchCounts.reset();

		// Flatten (target, sourceShape) groups into a dense (target, sourcePatch) stream.

		// Launch kernel to count patches for each source shape in each target group.
		cuExecute((uint)groupCountAll,
			NLQ_CountGroup2PatchCounts,
			mGroup2PatchCounts,
			mTarget2SourceShapes,
			mShape2PatchOffsets,
			patchCount);

		if (mGroup2PatchOffsets.size() != (uint)groupCountAll)
			mGroup2PatchOffsets.resize(groupCountAll);
		if (mGroup2TargetIds.size() != (uint)groupCountAll)
			mGroup2TargetIds.resize(groupCountAll);
		if (mTarget2SourceCounts.size() != (uint)shapeCount)
			mTarget2SourceCounts.resize(shapeCount);
		mTarget2SourceCounts.reset();

		// Launch kernel to build per-target source counts and per-source offsets.
		// mTarget2SourceCounts, mGroup2PatchOffsets and mGroup2TargetIds are computed here.
		// mTarget2SourceCounts: number of source patches considered for each target shape.
		// mGroup2PatchOffsets: offsets of patches for each source shape in a target shape
		// mGroup2TargetIds: target shape id for each patch of source shape
		// mGroup2PatchOffsets[g]: patch start offset of group g within its target.
		// mGroup2TargetIds[g]: owning target shape id of group g.
		cuExecute((uint)shapeCount,
			NLQ_BuildTarget2SourceCountsAndGroupOffsets,
			mTarget2SourceCounts,
			mGroup2PatchOffsets,
			mGroup2PatchCounts,
			mGroup2TargetIds,
			mTargetShapeOffsets,
			mTargetShapeCounts,
			shapeCount);

		int totalSource = mReduce.accumulate(mTarget2SourceCounts.begin(), mTarget2SourceCounts.size());
		if (totalSource <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (mTarget2SourceOffsets.size() != (uint)shapeCount)
			mTarget2SourceOffsets.resize(shapeCount);
		mTarget2SourceOffsets.assign(mTarget2SourceCounts);
		mScan.exclusive(mTarget2SourceOffsets, true);

		if (mGroup2GlobalOffsets.size() != (uint)groupCountAll)
			mGroup2GlobalOffsets.resize(groupCountAll);
		// Launch kernel to compute global source offsets per group.
		cuExecute((uint)groupCountAll,
			NLQ_BuildGroup2GlobalOffsets,
			mGroup2GlobalOffsets,
			mGroup2PatchOffsets,
			mGroup2TargetIds,
			mTarget2SourceOffsets);

		if (mSourcePatchAabbs.size() != (uint)totalSource)
			mSourcePatchAabbs.resize(totalSource);
		if (mSource2PatchIds.size() != (uint)totalSource)
			mSource2PatchIds.resize(totalSource);
		if (mSource2TargetIds.size() != (uint)totalSource)
			mSource2TargetIds.resize(totalSource);

		// Launch kernel to fill global source patch arrays (patch-level).
		cuExecute((uint)groupCountAll,
			NLQ_FillGroup2PatchDataByGroup,
			mSourcePatchAabbs,
			mSource2PatchIds,
			mSource2TargetIds,
			patchAabbs,
			mTarget2SourceShapes,
			mShape2PatchOffsets,
			mGroup2GlobalOffsets,
			mGroup2PatchCounts,
			mGroup2TargetIds,
			patchCount);

		if (totalSource > 0)
		{
			// Legacy: transform AABBs into target rest space by transforming AABB corners.
			// Kept for comparison/backward reference (do not delete).
			/*
			// Launch kernel to transform source patch AABBs into target rest space.
			cuExecute((uint)totalSource,
				NLQ_TransformPatchAabbsToTargetRest,
				mSourcePatchAabbs,
				mSource2TargetIds,
				mShapeRestR,
				mShapeRestT);
			*/

			// Build source patch AABBs directly from triangles in the target rest frame.
			// One warp handles one source patch (validated by NMQ_MaxPatchFaces guard).
			// p_world comes from triangleSet; transform to target rest with RRest^T * (p_world - tRest).
			uint totalThreads = (uint)totalSource * 32;
			cuExecute(totalThreads,
				NLQ_UpdateSourcePatchAabbsFromTrianglesWarp,
				mSourcePatchAabbs,
				mSource2PatchIds,
				mSource2TargetIds,
				patch2TriOffsets,
				patch2TriIndices,
				triIndices,
				vertices,
				mPatch2Shape,
				mShapeRestR,
				mShapeRestT,
				patchTriCount,
				triCount,
				vertexCount,
				this->varInputVerticesInRestWorld()->getValue());
			// if (!NMQ_CheckCuda("NLQ_UpdateSourcePatchAabbsFromTrianglesWarp"))
			// 	return false;
			}

		ensureMiddleWorkspace(totalSource);
		auto& localBroadPhaseCounter = mMiddleLocalBroadPhaseCounter;
		localBroadPhaseCounter.reset();

		// Launch kernel to count BVH intersections for each source patch.
		cuExecute((uint)totalSource,
			NLQ_RequestIntersectionNumberBVH,
			localBroadPhaseCounter,
			mSourcePatchAabbs,
			mSource2TargetIds,
			mTargetBVHs,
			mTargetBVHValid,
			mShape2PatchOffsets,
			patchAabbs);
		// if (!NMQ_CheckCuda("NLQ_RequestIntersectionNumberBVH"))
		// 	return false;
		// Launch kernel to fetch BVH intersection ids for each source patch.
		auto& contactList = mMiddleContactList;

		contactList.resize(localBroadPhaseCounter);

		cuExecute((uint)totalSource,
			NLQ_RequestIntersectionIdsBVH,
			contactList,
			mSourcePatchAabbs,
			mSource2TargetIds,
			mTargetBVHs,
			mTargetBVHValid,
			mShape2PatchOffsets,
			patchAabbs);
		// if (!NMQ_CheckCuda("NLQ_RequestIntersectionIdsBVH"))
		// 	return false;

		// printf("[NeighborTriMeshQuery] middle phase contacts=%u (lists=%u)\n",
			// (unsigned int)contactList.elementSize(),
			// (unsigned int)contactList.size());
		if (contactList.elementSize() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		auto& contactCount = mMiddleContactCount;
		cuExecute((uint)totalSource,
			NLQ_CopyCountU2I,
			contactCount,
			localBroadPhaseCounter);

		int totalPairs = mReduce.accumulate(contactCount.begin(), contactCount.size());
		if (totalPairs <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		auto& contactCountCpy = mMiddleContactCountCpy;
		contactCountCpy.assign(contactCount);
		mScan.exclusive(contactCount, true);

		this->outPotentialPatchPairs()->resize(totalPairs);
		// Launch kernel to write patch pairs from contact lists.
		cuExecute(contactList.size(),
			NLQ_SetPatchPairsFromContactList,
			this->outPotentialPatchPairs()->getData(),
			contactList,
			contactCount,
			contactCountCpy,
			mSource2PatchIds,
			mSource2TargetIds,
			mShape2PatchOffsets);

		// std::cout << "[NeighborTriMeshQuery] middlePhase found " << totalPairs << " patch pairs." << std::endl;
		// printf("[NeighborTriMeshQuery] MiddlePhase completed.\n");
		middleTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute middle phase time: " << middleTimer.elapsedMilliseconds() << " ms" << std::endl;
		
		return true;
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::narrowPhase()
	{
		// For each candidate patch pair:
		// 1) build candidate triangle lists
		// 2) count contacts (warp kernel)
		// 3) scan+scatter final ContactPair output
		NewTimer narrowTimer;
		narrowTimer.start();
		// printf("[NeighborTriMeshQuery] NarrowPhase started.\n");
		auto& patchPairs = this->outPotentialPatchPairs()->getData();
		if (patchPairs.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto& patch2TriOffsets = this->inPatch2TriOffsets()->getData();
		auto& patch2TriIndices = this->inPatch2TriIndices()->getData();

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patch2TriOffsets.size() < (uint)(patchCount + 1) || patch2TriIndices.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}
		if (!updatePatchFaceLimitState(patchCount))
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		auto& vertices = ts->getPoints();
		auto& triIndices = ts->triangleIndices();

		int triCount = (int)triIndices.size();
		if (triCount <= 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		int patchTriCount = (int)patch2TriIndices.size();

		if (mNarrowTriListSizes.size() != patchPairs.size())
			mNarrowTriListSizes.resize((uint)patchPairs.size());
		auto& triListSizes = mNarrowTriListSizes;
		triListSizes.reset();

		Real dHat = this->varDHead()->getValue();

		// Count the length of potential triangle contact lists for each patch pair
		cuExecute(patchPairs.size(),
			NLQ_Narrow_BuildTriListCounts,
			triListSizes,
			patchPairs,
			patch2TriOffsets,
			patchCount,
			patchTriCount);
		int totalTriLists = mReduce.accumulate(triListSizes.begin(), triListSizes.size());
		if (totalTriLists <= 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		ensureNarrowWorkspace(totalTriLists);
		auto& triListOffsets = mNarrowTriListOffsets;
		auto& triContactList = mNarrowTriContactList;
		auto& triListTriIds = mNarrowTriListTriIds;
		auto& triListPairIds = mNarrowTriListPairIds;
		auto& triListSide = mNarrowTriListSide;

		triListOffsets.assign(triListSizes);
		// Calculate offsets for each patch pair's triangle list using exclusive scan
		mScan.exclusive(triListOffsets, true);

		cuExecute(patchPairs.size(),
			NLQ_Narrow_BuildTriContactLists,
			triContactList,
			triListTriIds,
			triListPairIds,
			triListSide,
			patchPairs,
			patch2TriOffsets,
			patch2TriIndices,
			triListOffsets,
			patchCount,
			triCount,
			patchTriCount);

		/* Warp-level narrow phase version (current path). */
		auto& triContactCounts = mNarrowTriContactCounts;
		triContactCounts.reset();

		uint totalThreads = (uint)totalTriLists * 32;
		cuExecute(totalThreads,
			NLQ_Narrow_WarpCount,
			triContactCounts,
			triContactList,
			triListTriIds,
			triListPairIds,
			triListSide,
			patchPairs,
			mPatch2Shape,
			vertices,
			triIndices,
			mShapeRestR,
			mShapeRestT,
			dHat,
			triCount,
			this->varInputVerticesInRestWorld()->getValue());
		// printf("[NeighborTriMeshQuery] NarrowPhase contact count computed.\n");

		int total = mReduce.accumulate(triContactCounts.begin(), triContactCounts.size());
		if (total <= 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto& triContactOffsets = mNarrowTriContactOffsets;
		triContactOffsets.assign(triContactCounts);
		mScan.exclusive(triContactOffsets, true);

		this->outContacts()->resize(total);

		cuExecute(totalThreads,
			NLQ_Narrow_WarpSet,
			this->outContacts()->getData(),
			triContactList,
			triListTriIds,
			triListPairIds,
			triListSide,
			patchPairs,
			mPatch2Shape,
			vertices,
			triIndices,
			mShapeRestR,
			mShapeRestT,
			triContactOffsets,
			triContactCounts,
			this->inShape2RigidBodyIds()->getData(),
			dHat,
			triCount,
			this->varInputVerticesInRestWorld()->getValue());
		// printf("[NeighborTriMeshQuery] NarrowPhase contacts generated: %d contacts found.\n", total);

		static uint64_t sNarrowFrame = 0;
		const uint64_t frameId = sNarrowFrame++;

		if (this->outContacts()->size() > 1)
		{
			CArray<ContactPair> hContacts;
			hContacts.assign(this->outContacts()->getData());

			// NMQPrintNearCoincidentContactsEvidence<ContactPair>(hContacts, frameId);

			// const uint preAggCount = hContacts.size();

			// Keep deterministic order and geometrically aggregate near-identical contacts before solver.
			NMQStableSortAndAggregateContacts<ContactPair>(hContacts);

			// const uint postAggCount = hContacts.size();

			// printf("[NMQ_CP_AGG] frame=%llu pre=%u post=%u removed=%d\n",
			// 	(unsigned long long)frameId,
			// 	(unsigned int)preAggCount,
			// 	(unsigned int)postAggCount,
			// 	(int)preAggCount - (int)postAggCount);

			this->outContacts()->resize(hContacts.size());
			if (hContacts.size() > 0)
				this->outContacts()->getData().assign(hContacts);
		}

		{

			bool enabled = false;
			if (enabled)
			{
				int maxPrint = 16;

				CArray<ContactPair> hContacts;
				hContacts.assign(this->outContacts()->getData());

				int printCount = (int)hContacts.size();
				if (printCount > maxPrint)
					printCount = maxPrint;

				printf("[NMQ_CP] frame=%llu total=%u print=%d\n",
					(unsigned long long)frameId,
					(unsigned int)hContacts.size(),
					printCount);

				for (int i = 0; i < printCount; ++i)
				{
					const ContactPair& cp = hContacts[(uint)i];
					printf("  cp[%d] body=(%d,%d) tri=(%d,%d) pen=%.9f\n",
						i,
						(int)cp.bodyId1, (int)cp.bodyId2,
						(int)cp.localId1, (int)cp.localId2,
						(double)cp.interpenetration);
					printf("         pos1=(%.9f %.9f %.9f) pos2=(%.9f %.9f %.9f)\n",
						(double)cp.pos1[0], (double)cp.pos1[1], (double)cp.pos1[2],
						(double)cp.pos2[0], (double)cp.pos2[1], (double)cp.pos2[2]);
					printf("         n1=(%.9f %.9f %.9f) n2=(%.9f %.9f %.9f)\n",
						(double)cp.normal1[0], (double)cp.normal1[1], (double)cp.normal1[2],
						(double)cp.normal2[0], (double)cp.normal2[1], (double)cp.normal2[2]);
				}
			}
		}
		
		/* thread-level narrow phase version
		DArray<int> triPairSizes;
		triPairSizes.resize(totalTriLists);
		triPairSizes.reset();

		// Count the number of potential triangle contact pairs for each triangle list
		cuExecute(totalTriLists,
			NLQ_Narrow_CountTriContactSizes,
			triPairSizes,
			triContactList);
		cuSynchronize();
		printf("[NeighborTriMeshQuery] NarrowPhase contact count computed.\n");

		int totalPairs = mReduce.accumulate(triPairSizes.begin(), triPairSizes.size());
		if (totalPairs <= 0)
		{
			this->outContacts()->resize(0);
			triPairSizes.clear();
			triListSide.clear();
			triListPairIds.clear();
			triListTriIds.clear();
			triContactList.clear();
			triListOffsets.clear();
			triListSizes.clear();
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		DArray<int> triPairOffsets;
		triPairOffsets.assign(triPairSizes);
		mScan.exclusive(triPairOffsets, true);

		DArray<int> pairListIds;
		DArray<int> pairCandidateOffsets;
		pairListIds.resize(totalPairs);
		pairCandidateOffsets.resize(totalPairs);

		cuExecute(totalTriLists,
			NLQ_Narrow_FlattenTriContactList,
			pairListIds,
			pairCandidateOffsets,
			triContactList,
			triPairOffsets,
			triPairSizes);
		cuSynchronize();

		DArray<int> pairCounts;
		pairCounts.resize(totalPairs);
		pairCounts.reset();

		cuExecute(totalPairs,
			NLQ_Narrow_ThreadCount,
			pairCounts,
			triContactList,
			triListTriIds,
			triListSide,
			pairListIds,
			pairCandidateOffsets,
			vertices,
			triIndices,
			dHat,
			triCount);
		cuSynchronize();

		int total = mReduce.accumulate(pairCounts.begin(), pairCounts.size());
		if (total <= 0)
		{
			this->outContacts()->resize(0);
			pairCounts.clear();
			pairCandidateOffsets.clear();
			pairListIds.clear();
			triPairOffsets.clear();
			triPairSizes.clear();
			triListSide.clear();
			triListPairIds.clear();
			triListTriIds.clear();
			triContactList.clear();
			triListOffsets.clear();
			triListSizes.clear();
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		DArray<int> pairOffsets;
		pairOffsets.assign(pairCounts);
		mScan.exclusive(pairOffsets, true);

		this->outContacts()->resize(total);

		cuExecute(totalPairs,
			NLQ_Narrow_ThreadSet,
			this->outContacts()->getData(),
			triContactList,
			triListTriIds,
			triListPairIds,
			triListSide,
			pairListIds,
			pairCandidateOffsets,
			patchPairs,
			mPatch2Shape,
			vertices,
			triIndices,
			pairOffsets,
			pairCounts,
			dHat,
			triCount);
		cuSynchronize();
		
		printf("[NeighborTriMeshQuery] NarrowPhase contacts generated: %d contacts found.\n", total);
		*/
		
		// Build a TriangleSet for collided triangles in world space
		if (this->inEnableVisualizeCollisionTriSet()->getValue())
		{
			CArray<ContactPair> hContacts;
			hContacts.assign(this->outContacts()->getData());

			CArray<Coord> hVertices;
			hVertices.assign(vertices);
			CArray<Triangle> hTriangles;
			hTriangles.assign(triIndices);

			CArray<int> hPatch2TriOffsets;
			hPatch2TriOffsets.assign(patch2TriOffsets);
			CArray<int> hPatch2TriIndices;
			hPatch2TriIndices.assign(patch2TriIndices);
			CArray<uint> hPatch2Shape;
			hPatch2Shape.assign(mPatch2Shape);

			CArray<Coord> hCenters;
			hCenters.assign(this->inCenter()->getData());
			CArray<Matrix> hRotations;
			hRotations.assign(this->inRotationMatrix()->getData());
			CArray<Coord> hRestCenters;
			hRestCenters.assign(this->inRestShapeCenter()->getData());
			CArray<Matrix> hRestRotations;
			hRestRotations.assign(this->inRestShapeRotation()->getData());
			CArray<int> hShape2Rigid;
			hShape2Rigid.assign(this->inShape2RigidBodyIds()->getData());

			std::vector<int> triIdToShape(triCount, -1);
			for (int patchId = 0; patchId < patchCount; ++patchId)
			{
				int shapeId = patchId < (int)hPatch2Shape.size() ? (int)hPatch2Shape[patchId] : -1;
				if (shapeId < 0)
					continue;

				if (patchId + 1 >= (int)hPatch2TriOffsets.size())
					continue;
				int tStart = hPatch2TriOffsets[patchId];
				int tEnd = hPatch2TriOffsets[patchId + 1];
				if (tStart < 0) tStart = 0;
				if (tEnd > patchTriCount) tEnd = patchTriCount;
				if (tEnd <= tStart)
					continue;
				for (int t = tStart; t < tEnd; ++t)
				{
					int triId = hPatch2TriIndices[t];
					if (triId < 0 || triId >= triCount)
						continue;
					if (triIdToShape[triId] < 0)
						triIdToShape[triId] = shapeId;
				}
			}

			std::vector<Coord> contactVertices;
			std::vector<Triangle> contactTriangles;
			contactVertices.reserve(hContacts.size() * 6);
			contactTriangles.reserve(hContacts.size() * 2);

			for (uint i = 0; i < hContacts.size(); ++i)
			{
				int triId0 = hContacts[i].localId1;
				int triId1 = hContacts[i].localId2;
				if (triId0 < 0 || triId0 >= triCount || triId1 < 0 || triId1 >= triCount)
					continue;

				int shape0 = triIdToShape[triId0];
				int shape1 = triIdToShape[triId1];
				if (shape0 < 0 || shape1 < 0)
					continue;

				int bodyId0 = shape0;
				int bodyId1 = shape1;

				Triangle tri0 = hTriangles[triId0];
				Coord p00 = hVertices[tri0[0]];
				Coord p01 = hVertices[tri0[1]];
				Coord p02 = hVertices[tri0[2]];
				int base = (int)contactVertices.size();
				contactVertices.push_back(p00);
				contactVertices.push_back(p01);
				contactVertices.push_back(p02);
				contactTriangles.push_back(Triangle(base, base + 1, base + 2));

				Triangle tri1 = hTriangles[triId1];
				Coord p10 = hVertices[tri1[0]];
				Coord p11 = hVertices[tri1[1]];
				Coord p12 = hVertices[tri1[2]];
				base = (int)contactVertices.size();
				contactVertices.push_back(p10);
				contactVertices.push_back(p11);
				contactVertices.push_back(p12);
				contactTriangles.push_back(Triangle(base, base + 1, base + 2));
			}

			if (contactTriangles.empty())
			{
				this->triSet->clear();
				this->outPotentialTriSet()->setDataPtr(this->triSet);
			}
			else
			{
				this->triSet->setPoints(contactVertices);
				this->triSet->setTriangles(contactTriangles);
				this->triSet->update();
				this->outPotentialTriSet()->setDataPtr(this->triSet);
			}

			/* thread-level narrow phase version
			pairOffsets.clear();
			pairCounts.clear();
			pairCandidateOffsets.clear();
			pairListIds.clear();
			triPairOffsets.clear();
			triPairSizes.clear();
			triListSide.clear();
			triListPairIds.clear();
			triListTriIds.clear();
			*/
		}
		narrowTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute narrow phase time: " << narrowTimer.elapsedMilliseconds() << " ms" << std::endl;
		// printf("[NeighborTriMeshQuery] NarrowPhase completed.\n");
	}

	DEFINE_CLASS(NeighborTriMeshQuery);
}
