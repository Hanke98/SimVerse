#include "Array/ArrayList.h"
#include "NeighborTriMeshQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "../../../Dynamics/Cuda/RigidBody/RigidBodySystem.h"
#include "Vector/Vector3D.h"
#include "Timer.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace dyno
{
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

	__device__ inline int NLQ_FindGroupByOffset(
		int idx,
		const DArray<int>& group2GlobalOffsets,
		const DArray<int>& group2PatchCounts)
	{
		int lo = 0;
		int hi = (int)group2GlobalOffsets.size() - 1;
		int res = -1;
		while (lo <= hi)
		{
			int mid = (lo + hi) >> 1;
			int start = group2GlobalOffsets[mid];
			if (start <= idx)
			{
				res = mid;
				lo = mid + 1;
			}
			else
			{
				hi = mid - 1;
			}
		}
		if (res < 0)
			return -1;
		int start = group2GlobalOffsets[res];
		int count = group2PatchCounts[res];
		if (idx >= start && idx < (start + count))
			return res;
		return -1;
	}

	__global__ void NLQ_FillGroup2PatchData(
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
		int outIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		if (outIdx >= outAabbs.size() || outIdx >= outIds.size() || outIdx >= outTargetIds.size())
			return;

		int g = NLQ_FindGroupByOffset(outIdx, group2GlobalOffsets, group2PatchCounts);
		if (g < 0 || g >= groupedSources.size() || g >= group2TargetIds.size())
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

		int localIdx = outIdx - group2GlobalOffsets[g];
		if (localIdx < 0 || localIdx >= sCount)
			return;

		int srcIdx = sBegin + localIdx;
		if (srcIdx < 0 || srcIdx >= patchAabbsWorld.size())
			return;

		outAabbs[outIdx] = patchAabbsWorld[srcIdx];
		outIds[outIdx] = (uint)srcIdx;
		outTargetIds[outIdx] = target;
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
		DArray<Mat3f> targetRestR,
		DArray<Vec3f> targetRestT,
		int patchTriCount,
		int triCount,
		int vertexCount)
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
		if (targetId < 0 || targetId >= targetRestR.size() || targetId >= targetRestT.size())
			return;

		int start = NLQ_ClampInt(patch2TriOffsets[patchId], 0, patchTriCount);
		int end = NLQ_ClampInt(patch2TriOffsets[patchId + 1], 0, patchTriCount);
		int triCountLocal = end - start;
		if (triCountLocal <= 0 || triCountLocal > 32)
		{
			if (lane == 0)
			{
				AABB box;
				auto zero = Vec3f(Real(0));
				box.v0 = zero;
				box.v1 = zero;
				outAabbs[warpId] = box;
			}
			printf("[NeighborTriMeshQuery] Invalid triCountLocal: %d\n", 
				triCountLocal);
			return;
		}

		Mat3f RRest = targetRestR[targetId];
		Vec3f tRest = targetRestT[targetId];

		Vec3f localMin(REAL_MAX);
		Vec3f localMax(-REAL_MAX);

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

					Vec3f p0 = NLQ_TransformWorldPointToRest(p0c, RRest, tRest);
					Vec3f p1 = NLQ_TransformWorldPointToRest(p1c, RRest, tRest);
					Vec3f p2 = NLQ_TransformWorldPointToRest(p2c, RRest, tRest);

					localMin = p0.minimum(p1).minimum(p2);
					localMax = p0.maximum(p1).maximum(p2);
				}
			}
		}

		Vec3f warpMin = NLQ_WarpReduceMinVec3(localMin);
		Vec3f warpMax = NLQ_WarpReduceMaxVec3(localMax);

		if (lane == 0)
		{
			AABB box;
			box.v0 = warpMin;
			box.v1 = warpMax;
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
		switch (eleType)
		{
		case ET_SPHERE:
		{
			box = spheres[elementId - elementOffset.sphereIndex()].aabb();
			break;
		}
		case ET_BOX:
		{
			box = boxes[elementId - elementOffset.boxIndex()].aabb();
			break;
		}
		case ET_TET:
		{
			box = tets[elementId - elementOffset.tetIndex()].aabb();
			break;
		}
		case ET_CAPSULE:
		{
			box = caps[elementId - elementOffset.capsuleIndex()].aabb();
			break;
		}
		case ET_TRI:
		{
			boundary_expand = 0.01;
			box = tris[elementId - elementOffset.triangleIndex()].aabb();
			break;
		}
		default:
			break;
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

	__global__ void NLQ_CountContactList(
		DArray<int> counts,
		DArrayList<int> contactList)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		counts[tId] = contactList[tId].size();
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

	__global__ void NLQ_BuildPatch2Shape(
		DArray<uint> patch2Shape,
		DArray<int> shape2PatchOffsets,
		int patchCount)
	{
		int shapeId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (shapeId + 1 >= shape2PatchOffsets.size()) return;

		int start = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, patchCount);
		int end = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, patchCount);

		for (int p = start; p < end; ++p)
		{
			patch2Shape[p] = (uint)shapeId;
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
				// Note: If list size reaches 32, subsequent insertions are ignored (capacity limit).
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
				// Note: If list size reaches 32, subsequent insertions are ignored (capacity limit).
				list.insert(triId0);
			}
		}
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void NLQ_Narrow_WarpCount(
		DArray<int> counts,
		DArrayList<int> triContactList,
		DArray<int> triListTriIds,
		DArray<int> triListSide,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		Real dHat,
		int triCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		int warpId = tId / 32; // representing the current triangle list index
		int lane = tId % 32;   // representing the candidate triangle list index
		if (warpId >= triContactList.size()) return;

		int triIdCurrent = triListTriIds[warpId];

		int side = triListSide[warpId];
		if (triIdCurrent < 0 || triIdCurrent >= triCount)
		{
			if (lane == 0)
				counts[warpId] = 0;
			return;
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
				TTriangle3D<Real> t0(p00, p01, p02);

				Triangle tri1 = triangles[triId1];
				Coord p10 = vertices[tri1[0]];
				Coord p11 = vertices[tri1[1]];
				Coord p12 = vertices[tri1[2]];
				TTriangle3D<Real> t1(p10, p11, p12);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);
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
		DArray<int> prefix,
		DArray<int> counts,
		Real dHat,
		int triCount)
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

		int bodyId0 = shape0;
		int bodyId1 = shape1;

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
				TTriangle3D<Real> t0(p00, p01, p02);

				Triangle tri1 = triangles[triId1];
				Coord p10 = vertices[tri1[0]];
				Coord p11 = vertices[tri1[1]];
				Coord p12 = vertices[tri1[2]];
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
				cp.normal1 = -manifold.normal;
				cp.normal2 = manifold.normal;
				cp.contactType = ContactType::CT_NONPENETRATION;
				cp.interpenetration = -manifold.contacts[n].penetration;

				contacts[outIdx] = cp;
			}
		}
	}

	template<typename TDataType>
	NeighborTriMeshQuery<TDataType>::NeighborTriMeshQuery()
		: NeighborElementQuery<TDataType>()
	// NeighborTriMeshQuery<TDataType>::NeighborTriMeshQuery()
	// 	: NeighborElementQuery<TDataType>()
	{
		this->inAdjacentShapes()->tagOptional(true);
		// this->inShape2PatchCounts()->tagOptional(true);
		// this->inShape2RigidBodyIds()->tagOptional(true);
		this->inShape2ElementIds()->tagOptional(true);
		this->inShapeBVHs()->tagOptional(true);

		this->varGridSizeLimit()->setValue(Real(0.01));
		this->varDHead()->setValue(Real(0));
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
	bool NeighborTriMeshQuery<TDataType>::updateShape2ElementIds(int shapeCount)
	{
		if (shapeCount <= 0)
			return false;

		// const uint invalidElementId = static_cast<uint>(-1);

		// if (!this->inShape2ElementIds()->isEmpty())
		// {
		// 	auto& pairs = this->inShape2ElementIds()->getData();
		// 	if (pairs.size() != (uint)shapeCount)
		// 	{
		// 		if (!mWarnedEmptyElementMapping)
		// 		{
		// 			printf("[NeighborTriMeshQuery] Shape2ElementIds size mismatch (shapeCount=%d, pairCount=%u), skip this frame.\n",
		// 				shapeCount,
		// 				(unsigned int)pairs.size());
		// 			mWarnedEmptyElementMapping = true;
		// 		}
		// 		return false;
		// 	}

		// 	CArray<Pair<uint, uint>> hostPairs;
		// 	hostPairs.assign(pairs);

		// 	std::vector<int> shape2ElementIds(shapeCount, -1);
		// 	bool warnedDuplicate = false;
		// 	bool warnedOutOfRange = false;
		// 	for (uint i = 0; i < hostPairs.size(); ++i)
		// 	{
		// 		uint shapeId = hostPairs[i].first;
		// 		uint elementId = hostPairs[i].second;

		// 		if (elementId == invalidElementId)
		// 			continue;
		// 		if (shapeId >= (uint)shapeCount)
		// 		{
		// 			if (!warnedOutOfRange)
		// 			{
		// 				printf("[NeighborTriMeshQuery] Shape2ElementPairs has out-of-range shapeId=%u (shapeCount=%d), skipping.\n",
		// 					shapeId, shapeCount);
		// 				warnedOutOfRange = true;
		// 			}
		// 			continue;
		// 		}

		// 		if (shape2ElementIds[shapeId] >= 0 && !warnedDuplicate)
		// 		{
		// 			printf("[NeighborTriMeshQuery] Shape2ElementPairs has duplicate shapeId=%u, overwriting.\n", shapeId);
		// 			warnedDuplicate = true;
		// 		}
		// 		shape2ElementIds[shapeId] = (int)elementId;
		// 	}

		// 	bool allReady = true;
		// 	for (int i = 0; i < shapeCount; ++i)
		// 	{
		// 		if (shape2ElementIds[i] < 0)
		// 			allReady = false;
		// 	}

		// 	if (!allReady)
		// 	{
		// 		if (!mWarnedEmptyElementMapping)
		// 		{
		// 			printf("[NeighborTriMeshQuery] Shape2ElementMapping incomplete (shapeCount=%d), skip this frame.\n", shapeCount);
		// 			mWarnedEmptyElementMapping = true;
		// 		}
		// 		return false;
		// 	}

			// auto topo = this->inDiscreteElements()->getDataPtr();
			// if (topo == nullptr)
			// {
			// 	if (!mWarnedEmptyElementMapping)
			// 	{
			// 		printf("[NeighborTriMeshQuery] Shape2ElementMapping not ready yet (topology unavailable, shapeCount=%d), skip this frame.\n",
			// 			shapeCount);
			// 		mWarnedEmptyElementMapping = true;
			// 	}
			// 	return false;
			// }

			// auto& mapping = topo->shape2RigidBodyMapping();
			// if (mapping.size() == 0)
			// {
			// 	if (!mWarnedEmptyMapping)
			// 	{
			// 		printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
			// 			shapeCount,
			// 			(unsigned int)mapping.size());
			// 		mWarnedEmptyMapping = true;
			// 	}
			// 	return false;
			// }

			// CArray<Pair<uint, uint>> hostMapping;
			// hostMapping.assign(mapping);

			// uint totalSize = topo->totalSize();
			// if (totalSize == 0)
			// {
			// 	if (!mWarnedEmptyMapping)
			// 	{
			// 		printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (totalSize=0), skip this frame.\n");
			// 		mWarnedEmptyMapping = true;
			// 	}
			// 	return false;
			// }

			// std::vector<int> element2Rigid(totalSize, -1);
			// for (uint i = 0; i < hostMapping.size(); ++i)
			// {
			// 	uint elementId = hostMapping[i].first;
			// 	if (elementId < totalSize)
			// 		element2Rigid[elementId] = (int)hostMapping[i].second;
			// }

			// std::vector<int> shape2RigidBodyIds(shapeCount, -1);
			// bool rigidReady = true;
			// for (int i = 0; i < shapeCount; ++i)
			// {
			// 	int elementId = shape2ElementIds[i];
			// 	if (elementId < 0 || (uint)elementId >= totalSize)
			// 	{
			// 		rigidReady = false;
			// 		continue;
			// 	}
			// 	int bodyId = element2Rigid[elementId];
			// 	if (bodyId < 0)
			// 	{
			// 		rigidReady = false;
			// 		continue;
			// 	}
			// 	shape2RigidBodyIds[i] = bodyId;
			// }

			// if (!rigidReady)
			// {
			// 	if (!mWarnedEmptyMapping)
			// 	{
			// 		printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping incomplete (shapeCount=%d), skip this frame.\n", shapeCount);
			// 		mWarnedEmptyMapping = true;
			// 	}
			// 	return false;
			// }

			// mShape2ElementIds.assign(shape2ElementIds);
			// mShape2RigidBodyIds.assign(shape2RigidBodyIds);
		// 	mWarnedEmptyElementMapping = false;
		// 	mWarnedEmptyMapping = false;
		// 	mMappingReady = true;

		// 	return true;
		// } else {
		// 	printf("[NeighborTriMeshQuery] Shape2ElementIds input is empty, skip this frame.\n");
		// 	return false;
		// }

		// Move to BatchRigidBodySystem.cpp
		mShape2ElementIds.assign(this->inShape2ElementIdsDense()->getData());
		mShape2RigidBodyIds.assign(this->inShape2RigidBodyIds()->getData());
		return true;
	}

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
	void NeighborTriMeshQuery<TDataType>::compute()
	{
		CTimer timer;
		timer.start();

		auto finishTiming = [&]() {
			timer.stop();
			std::cout << "[NeighborTriMeshQuery] compute time: " << timer.getElapsedTime() << " ms" << std::endl;
		};

		mUseBroadPhasePatchPairs = false;
		if (this->outPotentialShapePairs()->isEmpty())
			this->outPotentialShapePairs()->allocate();
		if (this->outPotentialPatchPairs()->isEmpty())
			this->outPotentialPatchPairs()->allocate();
		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();
		if (this->outPotentialTriSet()->isEmpty())
			this->outPotentialTriSet()->allocate();

		// examine inputs
		if (this->inPatchAABBs()->isEmpty()
			|| this->inShape2PatchOffsets()->isEmpty()
			|| this->inPatch2TriOffsets()->isEmpty()
			|| this->inPatch2TriIndices()->isEmpty()
			|| this->inCenter()->isEmpty()
			|| this->inRotationMatrix()->isEmpty()
			|| this->inRestShapeCenter()->isEmpty()
			|| this->inRestShapeRotation()->isEmpty()
			|| this->inTriangleSet()->isEmpty())
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

		int shapeCount = this->inShape2ElementIds()->size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] Shape2ElementIds is empty.\n");
			finishTiming();
			return;
		}

		// examine inputs of shape transforms
		if (this->inRestShapeCenter()->isEmpty() ) // && (int)this->inRestShapeCenter()->size() != shapeCount
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] RestShapeCenter missing.\n");
			finishTiming();
			return;
		}

		if (this->inRestShapeRotation()->isEmpty() ) // && (int)this->inRestShapeRotation()->size() != shapeCount
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] RestShapeRotation missing.\n");
			finishTiming();
			return;
		}

		if (!updateShape2ElementIds(shapeCount))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			finishTiming();
			return;
		}

		if (!this->inShape2PatchOffsets()->isEmpty())
		{
			mShape2PatchOffsets.assign(this->inShape2PatchOffsets()->getData());
			if (mShape2PatchOffsets.size() != (uint)(shapeCount + 1))
			{
				this->outPotentialShapePairs()->resize(0);
				this->outPotentialPatchPairs()->resize(0);
				this->outContacts()->resize(0);
				this->triSet->clear();
				this->outPotentialTriSet()->setDataPtr(this->triSet);
				printf("[NeighborTriMeshQuery] Shape2PatchOffsets size mismatch.\n");
				finishTiming();
				return;
			}
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
			mPatch2Shape.resize(patchCount);

		// Build patch2shape mapping
		mPatch2Shape.reset();
		cuExecute(shapeCount,
			NLQ_BuildPatch2Shape,
			mPatch2Shape,
			mShape2PatchOffsets,
			patchCount);

		if (mPatch2Shape.size() != (uint)patchCount)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] Patch2Shape size mismatch.\n");
			finishTiming();
			return;
		}

		// BroadPhase: shape AABB overlap -> i<j shape pairs
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

		// MiddlePhase: shape pairs + patch CSR -> patch pairs 
		if (!middlePhase())
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			// printf("[NeighborTriMeshQuery] MiddlePhase failed.\n");
			finishTiming();
			return;
		}

		// NarrowPhase: patch pairs + triangle CSR -> contacts
		narrowPhase();
		finishTiming();
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::broadPhase()
	{
		// printf("[NeighborTriMeshQuery] BroadPhase started.\n");
		CTimer broadTimer;
		broadTimer.start();

		auto inTopo = this->inDiscreteElements()->getDataPtr();
		if (inTopo == nullptr)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] DiscreteElements missing.\n");
			return false;
		}

		int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] Shape2PatchOffsets size mismatch.\n");
			return false;
		}

		if (mShape2ElementIds.size() != (uint)shapeCount)
		{
			if (!updateShape2ElementIds(shapeCount))
			{
				this->outPotentialShapePairs()->resize(0);
				printf("[NeighborTriMeshQuery] Shape2ElementIds size mismatch.\n");
				return false;
			}
		}
		if (mShape2ElementIds.size() != (uint)shapeCount)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborTriMeshQuery] Shape2ElementIds size mismatch.\n");
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

		// TODO: not shapeCount but actual number of elements
		cuExecute((uint)shapeCount,
			NTQ_SetupAABBFromElementIds,
			this->mQueriedAABB,
			mShape2ElementIds,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capsuleInGlobal,
			triangleInGlobal,
			elementOffset,
			dHat);

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
			break;
		case Spatial::OCTREE:
			this->mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
			break;
		default:
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

		cuExecute(contactList.size(),
			NLQ_CountShapePairs,
			pairCount,
			contactList,
			adj,
			useAdj,
			shapeCount);

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

		pairCountCpy.clear();
		pairCount.clear();

		// std::cout << "[NeighborTriMeshQuery] broadPhase found " << total << " shape pairs." << std::endl;
		broadTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute broad phase time: " << broadTimer.getElapsedTime() << " ms" << std::endl;
		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::middlePhase()
	{
		// printf("[NeighborTriMeshQuery] MiddlePhase started.\n");
		CTimer middleTimer;
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
			mShape2RigidBodyIds);
		cuSynchronize();

		// Legacy: update patch AABBs by transforming rest-world AABB corners.
		// Kept for comparison/backward reference (do not delete).
		/*
		// Ensure mPatchAabbsWorld is properly sized before the kernel writes to it
		if (mPatchAabbsWorld.size() != patchCount)
			mPatchAabbsWorld.resize(patchCount);

		// Update patch AABBs in world space (current pose) from rest-world patch AABBs.
		// Notd: Is full update is necessary?
		// Launch kernel to update patch AABBs to world space.
		cuExecute((uint)patchCount,
			NLQ_UpdatePatchAabbsFromRestWorld,
			mPatchAabbsWorld,
			patchAabbs,
			mPatch2Shape,
			mShapeRestR,
			mShapeRestT);
		cuSynchronize();
		*/


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

		// Count how many souce shapes each target shape has from shapePairs
		// Launch kernel to count target shapes per shape pair.
		cuExecute(shapePairs.size(),
			NLQ_CountTargetShapes,
			mTargetShapeCounts,
			shapePairs,
			shapeCount);
		// Target shape counts computed
		// Exclusive scan to build target shape offsets
		if (mTargetShapeOffsets.size() != (uint)shapeCount)
			mTargetShapeOffsets.resize(shapeCount);
		mTargetShapeOffsets.assign(mTargetShapeCounts);
		mScan.exclusive(mTargetShapeOffsets, true);

		// Build target shape write flags and target->source shape mapping
		if (mTargetShapeWrite.size() != (uint)shapeCount)
			mTargetShapeWrite.resize(shapeCount);
		mTargetShapeWrite.reset(); 

		if (mTarget2SourceShapes.size() != shapePairs.size())
			mTarget2SourceShapes.resize(shapePairs.size());
		mTarget2SourceShapes.reset();

		// Group source shapes by target shapes from shapePairs
		// Launch kernel to group source shapes by target.
		cuExecute(shapePairs.size(),
			NLQ_GroupShapePairsByTarget,
			mTarget2SourceShapes,
			mTargetShapeOffsets,
			mTargetShapeWrite,
			shapePairs,
			shapeCount);
		// Target to source shape mapping built

		// Active target compaction is no longer needed in GPU-parallel middle phase.
		// Build GPU-accessible BVH table
		auto& shapeBVHs = this->inShapeBVHs()->constDataPtr();
		if (mTargetBVHs.size() != (uint)shapeCount)
			mTargetBVHs.resize(shapeCount);
		if (mTargetBVHValid.size() != (uint)shapeCount)
			mTargetBVHValid.resize(shapeCount);

		CArray<LinearBVH<TDataType>> hTargetBVHs;
		hTargetBVHs.resize(shapeCount);
		CArray<int> hTargetBVHValid;
		hTargetBVHValid.assign((uint)shapeCount, 0);

		// TODO: move ouside
		if (shapeBVHs != nullptr)
		{
			for (int i = 0; i < shapeCount; ++i)
			{
				if (i >= (int)shapeBVHs->size())
					continue;
				auto& bvh = (*shapeBVHs)[i];
				if (!bvh)
					continue;
				hTargetBVHs[i] = *bvh;
				int nodeCount = (int)bvh->getSortedAABBs().size();
				hTargetBVHValid[i] = (nodeCount > 0 && (nodeCount % 2) == 1) ? 1 : 0;
			}
		}

		mTargetBVHs.assign(hTargetBVHs);
		mTargetBVHValid.assign(hTargetBVHValid);

		int groupCountAll = (int)mTarget2SourceShapes.size(); // the number of groups
		if (groupCountAll <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (mGroup2PatchCounts.size() != (uint)groupCountAll)
			mGroup2PatchCounts.resize(groupCountAll);
		mGroup2PatchCounts.reset();

		// ======= flatten the source patches =======
		// flatten the (target, sourceShape) to (target, sourcePatch)

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
		// mTarget2SourceCounts：how many patches of source shapes are in each target shape
		// mGroup2PatchOffsets: offsets of patches for each source shape in a target shape
		// mGroup2TargetIds: target shape id for each patch of source shape
		// mGroup2PatchOffsets[g]：group g 在其 target 内部的 patch 起始偏移
		// mGroup2TargetIds[g]：group g 属于哪个 target
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
		cuExecute((uint)totalSource,
			NLQ_FillGroup2PatchData,
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

			// Build patch AABBs directly from triangles (current world space) in target rest-world space.
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
				vertexCount);
		}

		DArray<uint> localBroadPhaseCounter;
		localBroadPhaseCounter.resize(totalSource);

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

		DArrayList<int> contactList;
		contactList.resize(localBroadPhaseCounter);

		// Launch kernel to fetch BVH intersection ids for each source patch.
		cuExecute((uint)totalSource,
			NLQ_RequestIntersectionIdsBVH,
			contactList,
			mSourcePatchAabbs,
			mSource2TargetIds,
			mTargetBVHs,
			mTargetBVHValid,
			mShape2PatchOffsets,
			patchAabbs);

		// printf("[NeighborTriMeshQuery] middle phase contacts=%u (lists=%u)\n",
			// (unsigned int)contactList.elementSize(),
			// (unsigned int)contactList.size());
		if (contactList.elementSize() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		DArray<int> contactCount;
		contactCount.resize(contactList.size());
		contactCount.reset();

		// Launch kernel to count contacts per source patch.
		cuExecute(contactList.size(),
			NLQ_CountContactList,
			contactCount,
			contactList);

		int totalPairs = mReduce.accumulate(contactCount.begin(), contactCount.size());
		if (totalPairs <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			contactCount.clear();
			return false;
		}

		DArray<int> contactCountCpy;
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

		contactCountCpy.clear();
		contactCount.clear();

		// std::cout << "[NeighborTriMeshQuery] middlePhase found " << totalPairs << " patch pairs." << std::endl;
		// printf("[NeighborTriMeshQuery] MiddlePhase completed.\n");
		middleTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute middle phase time: " << middleTimer.getElapsedTime() << " ms" << std::endl;
		
		return true;
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::narrowPhase()
	{
		CTimer narrowTimer;
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

		DArray<int> triListSizes;
		triListSizes.resize(patchPairs.size());
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
		cuSynchronize();

		int totalTriLists = mReduce.accumulate(triListSizes.begin(), triListSizes.size());
		if (totalTriLists <= 0)
		{
			this->outContacts()->resize(0);
			triListSizes.clear();
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		DArray<int> triListOffsets;
		triListOffsets.assign(triListSizes);
		// Calculate offsets for each patch pair's triangle list using exclusive scan
		mScan.exclusive(triListOffsets, true);

		DArrayList<int> triContactList;
		// Resize each list to capacity 32. Unused slots contain undefined values but are ignored by size().
		// TODO: To optimize memory, a CSR approach could be used to build a compact array. However, maybe it will slow down the performance?
		triContactList.resize((uint)totalTriLists, 32);

		DArray<int> triListTriIds;
		DArray<int> triListPairIds;
		DArray<int> triListSide;
		triListTriIds.resize(totalTriLists);
		triListPairIds.resize(totalTriLists);
		triListSide.resize(totalTriLists);

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
		cuSynchronize();

		/* warp-level narrow phase version*/
		DArray<int> triContactCounts;
		triContactCounts.resize(totalTriLists);
		triContactCounts.reset();

		uint totalThreads = (uint)totalTriLists * 32;
		cuExecute(totalThreads,
			NLQ_Narrow_WarpCount,
			triContactCounts,
			triContactList,
			triListTriIds,
			triListSide,
			vertices,
			triIndices,
			dHat,
			triCount);
		cuSynchronize();
		// printf("[NeighborTriMeshQuery] NarrowPhase contact count computed.\n");

		int total = mReduce.accumulate(triContactCounts.begin(), triContactCounts.size());
		if (total <= 0)
		{
			this->outContacts()->resize(0);
			triListSide.clear();
			triListPairIds.clear();
			triListTriIds.clear();
			triContactCounts.clear();
			triContactList.clear();
			triListOffsets.clear();
			triListSizes.clear();
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		DArray<int> triContactOffsets;
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
			triContactOffsets,
			triContactCounts,
			dHat,
			triCount);
		cuSynchronize();
		// printf("[NeighborTriMeshQuery] NarrowPhase contacts generated: %d contacts found.\n", total);
		
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
			hShape2Rigid.assign(mShape2RigidBodyIds);

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
			/* warp-level narrow phase version */
			triContactList.clear();
			triListOffsets.clear();
			triListSizes.clear();
		}
		narrowTimer.stop();
		std::cout << "[NeighborTriMeshQuery] compute narrow phase time: " << narrowTimer.getElapsedTime() << " ms" << std::endl;
		// printf("[NeighborTriMeshQuery] NarrowPhase completed.\n");
	}

	DEFINE_CLASS(NeighborTriMeshQuery);
}

// #ifdef UNIT_TEST
// #include "Topology/TriangleSet.h"

// void NeighborTriMeshQuery_UnitTest()
// {
// 	using namespace dyno;

// 	NeighborTriMeshQuery<DataType3f> query;
// 	CArray<NeighborTriMeshQuery<DataType3f>::AABB> shapeAabbs;
// 	shapeAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
// 	shapeAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
// 	query.inShapeAABBs()->assign(shapeAabbs);

// 	CArray<NeighborTriMeshQuery<DataType3f>::AABB> patchAabbs;
// 	patchAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
// 	patchAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
// 	query.inPatchAABBs()->assign(patchAabbs);

// 	CArray<int> shape2PatchOffsets;
// 	shape2PatchOffsets.pushBack(0);
// 	shape2PatchOffsets.pushBack(1);
// 	shape2PatchOffsets.pushBack(2);
// 	query.inShape2PatchOffsets()->assign(shape2PatchOffsets);

// 	CArray<int> patch2TriOffsets;
// 	patch2TriOffsets.pushBack(0);
// 	patch2TriOffsets.pushBack(1);
// 	patch2TriOffsets.pushBack(2);
// 	query.inPatch2TriOffsets()->assign(patch2TriOffsets);

// 	CArray<int> patch2TriIndices;
// 	patch2TriIndices.pushBack(0);
// 	patch2TriIndices.pushBack(1);
// 	query.inPatch2TriIndices()->assign(patch2TriIndices);

// 	auto triSet = std::make_shared<TriangleSet<DataType3f>>();
// 	CArray<Vec3f> vertices;
// 	vertices.pushBack(Vec3f(0.0f, 0.0f, 0.0f));
// 	vertices.pushBack(Vec3f(1.0f, 0.0f, 0.0f));
// 	vertices.pushBack(Vec3f(0.0f, 1.0f, 0.0f));
// 	vertices.pushBack(Vec3f(1.0f, 1.0f, 0.0f));
// 	triSet->getPoints().assign(vertices);

// 	CArray<TopologyModule::Triangle> triangles;
// 	triangles.pushBack(TopologyModule::Triangle(0, 1, 2));
// 	triangles.pushBack(TopologyModule::Triangle(1, 3, 2));
// 	triSet->triangleIndices().assign(triangles);
// 	query.inTriangleSet()->setDataPtr(triSet);

// 	query.update();
// }
// #endif
