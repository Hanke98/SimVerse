#include "NeighborTriMeshQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "../../../Dynamics/Cuda/RigidBody/RigidBodySystem.h"
#include <cmath>
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

	__device__ inline AABB NLQ_TransformLocalAabbToWorld(const AABB& localAabb, const Mat3f& R, const Vec3f& t)
	{
		Vec3f centerLocal = (localAabb.v0 + localAabb.v1) * Real(0.5);
		Vec3f extentLocal = (localAabb.v1 - localAabb.v0) * Real(0.5);

		Vec3f centerWorld = R * centerLocal + t;

		Vec3f extentWorld;
		extentWorld[0] = fabs(R(0, 0)) * extentLocal[0] + fabs(R(0, 1)) * extentLocal[1] + fabs(R(0, 2)) * extentLocal[2];
		extentWorld[1] = fabs(R(1, 0)) * extentLocal[0] + fabs(R(1, 1)) * extentLocal[1] + fabs(R(1, 2)) * extentLocal[2];
		extentWorld[2] = fabs(R(2, 0)) * extentLocal[0] + fabs(R(2, 1)) * extentLocal[1] + fabs(R(2, 2)) * extentLocal[2];

		AABB worldAabb;
		worldAabb.v0 = centerWorld - extentWorld;
		worldAabb.v1 = centerWorld + extentWorld;
		return worldAabb;
	}

	inline bool NLQ_BuildShape2RigidBodyIds(
		const DArray<int>& mapping,
		int shapeCount,
		std::vector<int>& shape2RigidBodyIds)
	{
		CArray<int> hostMapping;
		hostMapping.assign(mapping);
		if (hostMapping.size() == 0)
			return false;

		shape2RigidBodyIds.assign(shapeCount, -1);
		int copyCount = hostMapping.size() < (uint)shapeCount ? (int)hostMapping.size() : shapeCount;
		for (int i = 0; i < copyCount; ++i)
		{
			shape2RigidBodyIds[i] = hostMapping[i];
		}

		return true;
	}

	inline bool NLQ_BuildShape2RigidBodyIds(
		const DArray<Pair<uint, uint>>& mapping,
		int shapeCount,
		std::vector<int>& shape2RigidBodyIds)
	{
		CArray<Pair<uint, uint>> hostMapping;
		hostMapping.assign(mapping);
		if (hostMapping.size() == 0)
			return false;

		shape2RigidBodyIds.assign(shapeCount, -1);
		for (uint i = 0; i < hostMapping.size(); ++i)
		{
			uint shapeId = hostMapping[i].first;
			if (shapeId < shape2RigidBodyIds.size())
				shape2RigidBodyIds[shapeId] = static_cast<int>(hostMapping[i].second);
		}

		return true;
	}

	template<typename Real, typename Coord, typename Matrix>
	__device__ inline void NLQ_GetRelativeTransform(
		int shapeId,
		const DArray<int>& shape2RigidBodyIds,
		const DArray<Coord>& centers,
		const DArray<Matrix>& rotations,
		const DArray<Coord>& restShapeCenters,
		const DArray<Matrix>& restShapeRotations,
		Matrix& RRel,
		Coord& tRel,
		int& bodyId)
	{
		bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0 && shapeId >= 0 && shapeId < shape2RigidBodyIds.size())
			bodyId = shape2RigidBodyIds[shapeId];

		if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
		{
			RRel = Matrix::identityMatrix();
			tRel = Coord(Real(0));
			return;
		}

		Coord tCurr = centers[bodyId];
		Matrix RCurr = rotations[bodyId];

		Coord tRest = Coord(Real(0));
		Matrix RRest = Matrix::identityMatrix();
		if (shapeId >= 0 && shapeId < restShapeCenters.size())
			tRest = restShapeCenters[shapeId];
		if (shapeId >= 0 && shapeId < restShapeRotations.size())
			RRest = restShapeRotations[shapeId];

		RRel = RCurr * RRest.transpose();
		tRel = tCurr - RRel * tRest;
	}

	// template<typename Real, typename Coord, typename Matrix, typename AABB>
	__global__ void NLQ_UpdateShapeAabbs(
		DArray<AABB> worldAabbs,
		DArray<AABB> localAabbs,
		DArray<Vec3f> centers,
		DArray<Mat3f> rotations,
		DArray<int> shape2RigidBodyIds)
	{
		int shapeId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (shapeId >= localAabbs.size() || shapeId >= worldAabbs.size())
			return;

		int bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0)
		{
			if (shapeId < 0 || shapeId >= shape2RigidBodyIds.size())
			{
				worldAabbs[shapeId] = localAabbs[shapeId];
				return;
			}
			bodyId = shape2RigidBodyIds[shapeId];
		}

		if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
		{
			worldAabbs[shapeId] = localAabbs[shapeId];
			return;
		}

		worldAabbs[shapeId] = NLQ_TransformLocalAabbToWorld(
			localAabbs[shapeId],
			rotations[bodyId],
			centers[bodyId]);
	}

	// template<typename Real, typename Coord, typename Matrix, typename AABB>
	__global__ void NLQ_UpdatePatchAabbs(
		DArray<AABB> worldAabbs,
		DArray<AABB> localAabbs,
		DArray<uint> patch2Shape,
		DArray<Vec3f> centers,
		DArray<Mat3f> rotations,
		DArray<int> shape2RigidBodyIds)
	{
		int patchId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (patchId >= localAabbs.size() || patchId >= worldAabbs.size())
			return;

		int shapeId = patchId < patch2Shape.size() ? (int)patch2Shape[patchId] : -1;
		if (shapeId < 0)
		{
			worldAabbs[patchId] = localAabbs[patchId];
			return;
		}

		int bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0)
		{
			if (shapeId >= shape2RigidBodyIds.size())
			{
				worldAabbs[patchId] = localAabbs[patchId];
				return;
			}
			bodyId = shape2RigidBodyIds[shapeId];
		}

		if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
		{
			worldAabbs[patchId] = localAabbs[patchId];
			return;
		}

		// worldAabbs[patchId] = NLQ_TransformLocalAabbToWorld<Real, Coord, Matrix, AABB>(
		// 	localAabbs[patchId],
		// 	rotations[bodyId],
		// 	centers[bodyId]);
		worldAabbs[patchId] = NLQ_TransformLocalAabbToWorld(
			localAabbs[patchId],
			rotations[bodyId],
			centers[bodyId]);
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

	template<typename AABB>
	__global__ void NLQ_CountPatchPairs(
		DArray<int> counts,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		DArray<AABB> patchAabbs,
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

		if (end0 <= start0 || end1 <= start1)
		{
			counts[tId] = 0;
			return;
		}

		int cnt = 0;
		for (int p0 = start0; p0 < end0; ++p0)
		{
			AABB aabb0 = patchAabbs[p0];
			for (int p1 = start1; p1 < end1; ++p1)
			{
				if (aabb0.checkOverlap(patchAabbs[p1]))
					cnt++;
			}
		}

		counts[tId] = cnt;
	}

	__global__ void NLQ_CountContactList(
		DArray<int> counts,
		DArrayList<int> contactList)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size() || tId >= counts.size()) return;

		counts[tId] = contactList[tId].size();
	}

	__global__ void NLQ_SetPatchPairsFromContactList(
		DArray<Pair<uint, uint>> patchPairs,
		DArrayList<int> contactList,
		DArray<int> prefix,
		DArray<int> counts,
		DArray<uint> source2PatchIds,
		int targetBase,
		int targetCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size() || tId >= counts.size() || tId >= source2PatchIds.size())
			return;

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;
		uint srcId = source2PatchIds[tId];

		List<int>& list_i = contactList[tId];
		for (int j = 0; j < list_i.size(); j++)
		{
			int targetIdx = list_i[j];
			if (targetIdx < 0 || targetIdx >= targetCount)
				continue;

			if (write < size && (offset + write) < patchPairs.size())
			{
				patchPairs[offset + write] = Pair<uint, uint>(srcId, (uint)(targetBase + targetIdx));
				write++;
			}
		}
	}

	template<typename AABB>
	__global__ void NLQ_SetPatchPairs(
		DArray<Pair<uint, uint>> patchPairs,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		DArray<AABB> patchAabbs,
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
			AABB aabb0 = patchAabbs[p0];
			for (int p1 = start1; p1 < end1; ++p1)
			{
				if (!aabb0.checkOverlap(patchAabbs[p1]))
					continue;

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

	__global__ void NLQ_BuildPatchGlobalIds(
		DArray<uint> patchIds)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= patchIds.size()) return;

		patchIds[pId] = (uint)pId;
	}

	template<typename Real, typename Coord, typename Matrix, typename Triangle>
	__global__ void NLQ_Narrow_Count(
		DArray<int> counts,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patch2TriOffsets,
		DArray<int> patch2TriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<uint> patch2Shape,
		DArray<Coord> centers,
		DArray<Matrix> rotations,
		DArray<Coord> restShapeCenters,
		DArray<Matrix> restShapeRotations,
		DArray<int> shape2RigidBodyIds,
		Real dHat,
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
		{
			counts[tId] = 0;
			return;
		}

		if (patch0 + 1 >= patch2TriOffsets.size() || patch1 + 1 >= patch2TriOffsets.size())
		{
			counts[tId] = 0;
			return;
		}

		int start0 = NLQ_ClampInt(patch2TriOffsets[patch0], 0, patchTriCount);
		int end0 = NLQ_ClampInt(patch2TriOffsets[patch0 + 1], 0, patchTriCount);
		int start1 = NLQ_ClampInt(patch2TriOffsets[patch1], 0, patchTriCount);
		int end1 = NLQ_ClampInt(patch2TriOffsets[patch1 + 1], 0, patchTriCount);

		int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
		int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

		Matrix RRel0 = Matrix::identityMatrix();
		Matrix RRel1 = Matrix::identityMatrix();
		Coord tRel0 = Coord(Real(0));
		Coord tRel1 = Coord(Real(0));
		int bodyId0 = shape0;
		int bodyId1 = shape1;

		if (shape0 >= 0)
		{
			NLQ_GetRelativeTransform<Real, Coord, Matrix>(
				shape0,
				shape2RigidBodyIds,
				centers,
				rotations,
				restShapeCenters,
				restShapeRotations,
				RRel0,
				tRel0,
				bodyId0);
		}

		if (shape1 >= 0)
		{
			NLQ_GetRelativeTransform<Real, Coord, Matrix>(
				shape1,
				shape2RigidBodyIds,
				centers,
				rotations,
				restShapeCenters,
				restShapeRotations,
				RRel1,
				tRel1,
				bodyId1);
		}

		int cnt = 0;
		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patch2TriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;

			Triangle tri0 = triangles[triId0];
			Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
			Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
			Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
			TTriangle3D<Real> t0(p00, p01, p02);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patch2TriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				Triangle tri1 = triangles[triId1];
				Coord p10 = RRel1 * vertices[tri1[0]] + tRel1;
				Coord p11 = RRel1 * vertices[tri1[1]] + tRel1;
				Coord p12 = RRel1 * vertices[tri1[2]] + tRel1;
				TTriangle3D<Real> t1(p10, p11, p12);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);

				cnt += manifold.contactCount;
			}
		}

		counts[tId] = cnt;
	}

	template<typename Real, typename Coord, typename Matrix, typename Triangle, typename ContactPair>
	__global__ void NLQ_Narrow_Set(
		DArray<ContactPair> contacts,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patch2TriOffsets,
		DArray<int> patch2TriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<uint> patch2Shape,
		DArray<Coord> centers,
		DArray<Matrix> rotations,
		DArray<Coord> restShapeCenters,
		DArray<Matrix> restShapeRotations,
		DArray<int> shape2RigidBodyIds,
		DArray<int> prefix,
		DArray<int> counts,
		Real dHat,
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

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;

		int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
		int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

		Matrix RRel0 = Matrix::identityMatrix();
		Matrix RRel1 = Matrix::identityMatrix();
		Coord tRel0 = Coord(Real(0));
		Coord tRel1 = Coord(Real(0));
		int bodyId0 = shape0;
		int bodyId1 = shape1;

		if (shape0 >= 0)
		{
			NLQ_GetRelativeTransform<Real, Coord, Matrix>(
				shape0,
				shape2RigidBodyIds,
				centers,
				rotations,
				restShapeCenters,
				restShapeRotations,
				RRel0,
				tRel0,
				bodyId0);
		}

		if (shape1 >= 0)
		{
			NLQ_GetRelativeTransform<Real, Coord, Matrix>(
				shape1,
				shape2RigidBodyIds,
				centers,
				rotations,
				restShapeCenters,
				restShapeRotations,
				RRel1,
				tRel1,
				bodyId1);
		}

		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patch2TriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;

			Triangle tri0 = triangles[triId0];
			Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
			Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
			Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
			TTriangle3D<Real> t0(p00, p01, p02);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patch2TriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				Triangle tri1 = triangles[triId1];
				Coord p10 = RRel1 * vertices[tri1[0]] + tRel1;
				Coord p11 = RRel1 * vertices[tri1[1]] + tRel1;
				Coord p12 = RRel1 * vertices[tri1[2]] + tRel1;
				TTriangle3D<Real> t1(p10, p11, p12);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);

				for (int n = 0; n < manifold.contactCount; ++n)
				{
					if (write >= size || (offset + write) >= contacts.size())
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

					contacts[offset + write] = cp;
					write++;
				}
			}
		}
	}

	template<typename TDataType>
	NeighborTriMeshQuery<TDataType>::NeighborTriMeshQuery()
		: ComputeModule()
	// NeighborTriMeshQuery<TDataType>::NeighborTriMeshQuery()
	// 	: NeighborElementQuery<TDataType>()
	{
		this->inAdjacentShapes()->tagOptional(true);
		this->inShape2PatchCounts()->tagOptional(true);
		this->inShape2RigidBodyIds()->tagOptional(true);

		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();

		this->varGridSizeLimit()->setValue(Real(0.01));
		this->varDHead()->setValue(Real(0));
	}

	template<typename TDataType>
	NeighborTriMeshQuery<TDataType>::~NeighborTriMeshQuery()
	{
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::updateShape2RigidBodyIds(int shapeCount)
	{
		auto parent = this->getParentNode();
		auto rigid = dynamic_cast<RigidBodySystem<TDataType>*>(parent);
		if (rigid == nullptr)
		{
			if (!mWarnedEmptyMapping)
			{
				printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (topology unavailable, shapeCount=%d, mappingSize=0), skip this frame.\n", shapeCount);
				mWarnedEmptyMapping = true;
			}
			mMappingReady = false;
			return false;
		}

		auto topo = rigid->stateTopology()->constDataPtr();
		if (topo == nullptr)
		{
			if (!mWarnedEmptyMapping)
			{
				printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (topology unavailable, shapeCount=%d, mappingSize=0), skip this frame.\n", shapeCount);
				mWarnedEmptyMapping = true;
			}
			mMappingReady = false;
			return false;
		}

		auto& mapping = topo->shape2RigidBodyMapping();
		uint mappingSize = mapping.size();
		if (mappingSize == 0)
		{
			if (!mWarnedEmptyMapping)
			{
				printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
					shapeCount,
					mappingSize);
				mWarnedEmptyMapping = true;
			}
			mMappingReady = false;
			return false;
		}

		std::vector<int> shape2RigidBodyIds;
		if (!NLQ_BuildShape2RigidBodyIds(mapping, shapeCount, shape2RigidBodyIds))
		{
			if (!mWarnedEmptyMapping)
			{
				printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
					shapeCount,
					mappingSize);
				mWarnedEmptyMapping = true;
			}
			mMappingReady = false;
			return false;
		}

		mShape2RigidBodyIds.assign(shape2RigidBodyIds);
		if (!mMappingReady)
		{
			printf("[NeighborTriMeshQuery] Shape2RigidBodyMapping ready (shapeCount=%d, mappingSize=%u).\n",
				shapeCount,
				mappingSize);
		}
		mMappingReady = true;
		mWarnedEmptyMapping = false;

		return true;
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::compute()
	{
		if (this->outPotentialShapePairs()->isEmpty())
			this->outPotentialShapePairs()->allocate();
		if (this->outPotentialPatchPairs()->isEmpty())
			this->outPotentialPatchPairs()->allocate();
		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();

		if (this->inShapeAABBs()->isEmpty()
			|| this->inPatchAABBs()->isEmpty()
			|| (this->inShape2PatchOffsets()->isEmpty() && this->inShape2PatchCounts()->isEmpty())
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
			printf("[NeighborTriMeshQuery] Missing input data.\n");
			return;
		}

		auto& shapeAabbs = this->inShapeAABBs()->getData();
		int shapeCount = (int)shapeAabbs.size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			printf("[NeighborTriMeshQuery] shapeAABBs is empty.\n");
			return;
		}

		if (this->inRestShapeCenter()->isEmpty() ) // && (int)this->inRestShapeCenter()->size() != shapeCount
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			printf("[NeighborTriMeshQuery] RestShapeCenter missing.\n");
			return;
		}

		if (this->inRestShapeRotation()->isEmpty() ) // && (int)this->inRestShapeRotation()->size() != shapeCount
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			printf("[NeighborTriMeshQuery] RestShapeRotation missing.\n");
			return;
		}

		if (!updateShape2RigidBodyIds(shapeCount))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		if (!this->inShape2PatchOffsets()->isEmpty())
		{
			mShape2PatchOffsets.assign(this->inShape2PatchOffsets()->getData());
		}
		else
		{
			auto& counts = this->inShape2PatchCounts()->getData();
			if ((int)counts.size() != shapeCount)
			{
				this->outPotentialShapePairs()->resize(0);
				this->outPotentialPatchPairs()->resize(0);
				this->outContacts()->resize(0);
				printf("[NeighborTriMeshQuery] Shape2PatchCounts size mismatch.\n");
				return;
			}

			mShape2PatchOffsets.resize(counts.size() + 1);
			mShape2PatchOffsets.reset();
			mShape2PatchOffsets.assign(counts, counts.size(), 0, 0);
			mScan.exclusive(mShape2PatchOffsets, true);
		}

		if (mShape2PatchOffsets.size() != (uint)(shapeCount + 1))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			printf("[NeighborTriMeshQuery] Shape2PatchOffsets size mismatch.\n");
			return;
		}

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patchCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		if (mPatch2Shape.size() != (uint)patchCount)
			mPatch2Shape.resize(patchCount);
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
			printf("[NeighborTriMeshQuery] Patch2Shape size mismatch.\n");
			return;
		}

		// BroadPhase: shape AABB overlap -> i<j shape pairs
		if (!broadPhase())
		{
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		// MiddlePhase: shape pairs + patch CSR -> patch pairs 
		if (!middlePhase())
		{
			this->outContacts()->resize(0);
			return;
		}

		// NarrowPhase: patch pairs + triangle CSR -> contacts
		narrowPhase();
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::broadPhase()
	{
		auto& shapeAabbs = this->inShapeAABBs()->getData();
		int shapeCount = (int)shapeAabbs.size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			return false;
		}

		if (mShapeAabbsWorld.size() != (uint)shapeCount)
			mShapeAabbsWorld.resize(shapeCount);

		// NeighborTriMeshQuery runs before TJConstraintSolver in the animation pipeline,
		// so inCenter/inRotationMatrix represent the previous-frame pose.
		cuExecute((uint)shapeCount,
			NLQ_UpdateShapeAabbs,
			mShapeAabbsWorld,
			shapeAabbs,
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData(),
			mShape2RigidBodyIds);


		mBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
		mBroadPhaseCD->varSelfCollision()->setValue(true);

		mBroadPhaseCD->inSource()->assign(mShapeAabbsWorld);
		mBroadPhaseCD->inTarget()->assign(mShapeAabbsWorld);

		auto type = this->varSpatial()->getDataPtr()->currentKey();
		switch (type)
		{
		case Spatial::BVH:
			mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
			break;
		case Spatial::OCTREE:
			mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
			break;
		default:
			break;
		}

		mBroadPhaseCD->update();

		auto& contactList = mBroadPhaseCD->outContactList()->getData();
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

		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::middlePhase()
	{
		auto& shapePairs = this->outPotentialShapePairs()->getData();
		if (shapePairs.size() == 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		auto& patchAabbs = this->inPatchAABBs()->getData();
		int patchCount = (int)patchAabbs.size();
		if (patchCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		if (mPatchAabbsWorld.size() != (uint)patchCount)
			mPatchAabbsWorld.resize(patchCount);

		{
			cuExecute((uint)patchCount,
				NLQ_UpdatePatchAabbs,
				mPatchAabbsWorld,
				patchAabbs,
				mPatch2Shape,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				mShape2RigidBodyIds);
		}

// // #if !defined(NLQ_USE_LEGACY_MIDDLEPHASE)
		if (mPatch2GlobalIds.size() != (uint)patchCount)
		{
			mPatch2GlobalIds.resize(patchCount);
			cuExecute((uint)patchCount, NLQ_BuildPatchGlobalIds, mPatch2GlobalIds);
		}

		int shapeCount = (int)mShape2PatchOffsets.size() - 1;
		if (shapeCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

#ifndef NDEBUG
		printf("[NeighborTriMeshQuery] middlePhase shapePairs=%u\n", (uint)shapePairs.size());
#endif

		if (mTargetShapeCounts.size() != (uint)shapeCount)
			mTargetShapeCounts.resize(shapeCount);
		mTargetShapeCounts.reset();

		cuExecute(shapePairs.size(),
			NLQ_CountTargetShapes,
			mTargetShapeCounts,
			shapePairs,
			shapeCount);

		if (mTargetShapeOffsets.size() != (uint)shapeCount)
			mTargetShapeOffsets.resize(shapeCount);
		mTargetShapeOffsets.assign(mTargetShapeCounts);
		mScan.exclusive(mTargetShapeOffsets, true);

		if (mTargetShapeWrite.size() != (uint)shapeCount)
			mTargetShapeWrite.resize(shapeCount);
		mTargetShapeWrite.reset();

		if (mTarget2SourceShapes.size() != shapePairs.size())
			mTarget2SourceShapes.resize(shapePairs.size());
		mTarget2SourceShapes.reset();

		cuExecute(shapePairs.size(),
			NLQ_GroupShapePairsByTarget,
			mTarget2SourceShapes,
			mTargetShapeOffsets,
			mTargetShapeWrite,
			shapePairs,
			shapeCount);

		CArray<int> hTargetCounts;
		CArray<int> hTargetOffsets;
		CArray<int> hGroupedSources;
		CArray<int> hShape2PatchOffsets;
		hTargetCounts.assign(mTargetShapeCounts);
		hTargetOffsets.assign(mTargetShapeOffsets);
		hGroupedSources.assign(mTarget2SourceShapes);
		hShape2PatchOffsets.assign(mShape2PatchOffsets);

		CArray<int> hTargetPairCounts;
		hTargetPairCounts.assign((uint)shapeCount, 0);

		std::vector<std::unique_ptr<DArray<PairUU>>> targetPairs;
		targetPairs.resize(shapeCount);

		auto clampInt = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

		// NOTE: CollisionDetectionBroadPhase uses internal buffers and is not thread-safe for
		// concurrent update() calls, so targetShape groups are processed sequentially here.
		// The target->sources grouping keeps the layout ready for batched parallelization.
		for (int target = 0; target < shapeCount; ++target)
		{
			int groupCount = hTargetCounts[target];
			if (groupCount <= 0)
				continue;

			int groupStart = hTargetOffsets[target];
			if (groupStart < 0 || groupStart >= (int)hGroupedSources.size())
				continue;

			int groupEnd = groupStart + groupCount;
			if (groupEnd > (int)hGroupedSources.size())
				groupEnd = (int)hGroupedSources.size();

			int tBegin = clampInt(hShape2PatchOffsets[target], 0, patchCount);
			int tEnd = clampInt(hShape2PatchOffsets[target + 1], 0, patchCount);
			int tCount = tEnd - tBegin;
			if (tCount <= 0)
				continue;

			if (mTargetPatchAabbs.size() != (uint)tCount)
				mTargetPatchAabbs.resize(tCount);
			mTargetPatchAabbs.assign(mPatchAabbsWorld, tCount, 0, tBegin);

			int sourceTotal = 0;
			for (int i = groupStart; i < groupEnd; ++i)
			{
				int sourceShape = hGroupedSources[i];
				if (sourceShape < 0 || sourceShape + 1 >= (int)hShape2PatchOffsets.size())
					continue;

				int sBegin = clampInt(hShape2PatchOffsets[sourceShape], 0, patchCount);
				int sEnd = clampInt(hShape2PatchOffsets[sourceShape + 1], 0, patchCount);
				if (sEnd > sBegin)
					sourceTotal += (sEnd - sBegin);
			}

			if (sourceTotal <= 0)
				continue;

			if (mSourcePatchAabbs.size() != (uint)sourceTotal)
				mSourcePatchAabbs.resize(sourceTotal);
			if (mSource2PatchIds.size() != (uint)sourceTotal)
				mSource2PatchIds.resize(sourceTotal);

			int dstOffset = 0;
			for (int i = groupStart; i < groupEnd; ++i)
			{
				int sourceShape = hGroupedSources[i];
				if (sourceShape < 0 || sourceShape + 1 >= (int)hShape2PatchOffsets.size())
					continue;

				int sBegin = clampInt(hShape2PatchOffsets[sourceShape], 0, patchCount);
				int sEnd = clampInt(hShape2PatchOffsets[sourceShape + 1], 0, patchCount);
				int sCount = sEnd - sBegin;
				if (sCount <= 0)
					continue;

				if (dstOffset + sCount > sourceTotal)
					sCount = sourceTotal - dstOffset;
				if (sCount <= 0)
					break;

				mSourcePatchAabbs.assign(mPatchAabbsWorld, sCount, (uint)dstOffset, (uint)sBegin);
				mSource2PatchIds.assign(mPatch2GlobalIds, sCount, (uint)dstOffset, (uint)sBegin);
				dstOffset += sCount;
			}

			if (dstOffset <= 0)
				continue;

			mBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
			mBroadPhaseCD->varSelfCollision()->setValue(false);
			mBroadPhaseCD->inSource()->assign(mSourcePatchAabbs);
			mBroadPhaseCD->inTarget()->assign(mTargetPatchAabbs);

			auto type = this->varSpatial()->getDataPtr()->currentKey();
			switch (type)
			{
			case Spatial::BVH:
				mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
				break;
			case Spatial::OCTREE:
				mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
				break;
			default:
				break;
			}

			mBroadPhaseCD->update();

			auto& contactList = mBroadPhaseCD->outContactList()->getData();
			if (contactList.elementSize() == 0)
				continue;

			DArray<int> contactCount;
			contactCount.resize(contactList.size());
			contactCount.reset();

			cuExecute(contactList.size(),
				NLQ_CountContactList,
				contactCount,
				contactList);

			int total = mReduce.accumulate(contactCount.begin(), contactCount.size());
			hTargetPairCounts[target] = total;
			if (total <= 0)
			{
				contactCount.clear();
				continue;
			}

			DArray<int> contactCountCpy;
			contactCountCpy.assign(contactCount);
			mScan.exclusive(contactCount, true);

			auto pairs = std::make_unique<DArray<PairUU>>();
			pairs->resize(total);

			cuExecute(contactList.size(),
				NLQ_SetPatchPairsFromContactList,
				*pairs,
				contactList,
				contactCount,
				contactCountCpy,
				mSource2PatchIds,
				tBegin,
				tCount);

			targetPairs[target] = std::move(pairs);

			contactCountCpy.clear();
			contactCount.clear();
		}

		int totalPairs = 0;
		CArray<int> hTargetPairOffsets;
		hTargetPairOffsets.resize(shapeCount);
		for (int i = 0; i < shapeCount; ++i)
		{
			hTargetPairOffsets[i] = totalPairs;
			int count = hTargetPairCounts[i];
			if (count > 0)
				totalPairs += count;
		}

		if (totalPairs <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}

		this->outPotentialPatchPairs()->resize(totalPairs);
		auto& patchPairs = this->outPotentialPatchPairs()->getData();

		for (int i = 0; i < shapeCount; ++i)
		{
			int count = hTargetPairCounts[i];
			if (count <= 0 || !targetPairs[i])
				continue;

			int offset = hTargetPairOffsets[i];
			patchPairs.assign(*targetPairs[i], (uint)count, (uint)offset, 0);
			targetPairs[i]->clear();
			targetPairs[i].reset();
		}

#ifndef NDEBUG
		printf("[NeighborTriMeshQuery] middlePhase patchPairs=%d\n", totalPairs);
#endif

		return true;

// #else
		// // Legacy O(n^2) middlePhase. Kept for reference/regression fallback.
		// DArray<int> pairCount;
		// pairCount.resize(shapePairs.size());
		// pairCount.reset();

		// DArray<int> pairCountCpy;

		// cuExecute(shapePairs.size(),
		// 	NLQ_CountPatchPairs,
		// 	pairCount,
		// 	shapePairs,
		// 	mShape2PatchOffsets,
		// 	mPatchAabbsWorld,
		// 	patchCount);

		// int total = mReduce.accumulate(pairCount.begin(), pairCount.size());
		// if (total <= 0)
		// {
		// 	this->outPotentialPatchPairs()->resize(0);
		// 	pairCount.clear();
		// 	pairCountCpy.clear();
		// 	return false;
		// }

		// pairCountCpy.assign(pairCount);
		// mScan.exclusive(pairCount, true);

		// this->outPotentialPatchPairs()->resize(total);

		// cuExecute(shapePairs.size(),
		// 	NLQ_SetPatchPairs,
		// 	this->outPotentialPatchPairs()->getData(),
		// 	shapePairs,
		// 	mShape2PatchOffsets,
		// 	mPatchAabbsWorld,
		// 	pairCount,
		// 	pairCountCpy,
		// 	patchCount);

		// pairCountCpy.clear();
		// pairCount.clear();

		// return true;
// #endif
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::narrowPhase()
	{
		auto& patchPairs = this->outPotentialPatchPairs()->getData();
		if (patchPairs.size() == 0)
		{
			this->outContacts()->resize(0);
			return;
		}

		auto& patch2TriOffsets = this->inPatch2TriOffsets()->getData();
		auto& patch2TriIndices = this->inPatch2TriIndices()->getData();

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patch2TriOffsets.size() < (uint)(patchCount + 1) || patch2TriIndices.size() == 0)
		{
			this->outContacts()->resize(0);
			return;
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		auto& vertices = ts->getPoints();
		auto& triIndices = ts->triangleIndices();

		int triCount = (int)triIndices.size();
		if (triCount <= 0)
		{
			this->outContacts()->resize(0);
			return;
		}

		int patchTriCount = (int)patch2TriIndices.size();

		DArray<int> contactCount;
		contactCount.resize(patchPairs.size());
		contactCount.reset();

		DArray<int> contactCountCpy;

		Real dHat = this->varDHead()->getValue();

		cuExecute(patchPairs.size(),
			NLQ_Narrow_Count,
			contactCount,
			patchPairs,
			patch2TriOffsets,
			patch2TriIndices,
			vertices,
			triIndices,
			mPatch2Shape,
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData(),
			this->inRestShapeCenter()->getData(),
			this->inRestShapeRotation()->getData(),
			mShape2RigidBodyIds,
			dHat,
			patchCount,
			triCount,
			patchTriCount);

		int total = mReduce.accumulate(contactCount.begin(), contactCount.size());
		if (total <= 0)
		{
			this->outContacts()->resize(0);
			contactCount.clear();
			contactCountCpy.clear();
			return;
		}

		contactCountCpy.assign(contactCount);
		mScan.exclusive(contactCount, true);

		this->outContacts()->resize(total);

		cuExecute(patchPairs.size(),
			NLQ_Narrow_Set,
			this->outContacts()->getData(),
			patchPairs,
			patch2TriOffsets,
			patch2TriIndices,
			vertices,
			triIndices,
			mPatch2Shape,
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData(),
			this->inRestShapeCenter()->getData(),
			this->inRestShapeRotation()->getData(),
			mShape2RigidBodyIds,
			contactCount,
			contactCountCpy,
			dHat,
			patchCount,
			triCount,
			patchTriCount);

		contactCountCpy.clear();
		contactCount.clear();
	}

	DEFINE_CLASS(NeighborTriMeshQuery);
}

#ifdef UNIT_TEST
#include "Topology/TriangleSet.h"

void NeighborTriMeshQuery_UnitTest()
{
	using namespace dyno;

	NeighborTriMeshQuery<DataType3f> query;
	CArray<NeighborTriMeshQuery<DataType3f>::AABB> shapeAabbs;
	shapeAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
	shapeAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
	query.inShapeAABBs()->assign(shapeAabbs);

	CArray<NeighborTriMeshQuery<DataType3f>::AABB> patchAabbs;
	patchAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
	patchAabbs.pushBack(NeighborTriMeshQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
	query.inPatchAABBs()->assign(patchAabbs);

	CArray<int> shape2PatchOffsets;
	shape2PatchOffsets.pushBack(0);
	shape2PatchOffsets.pushBack(1);
	shape2PatchOffsets.pushBack(2);
	query.inShape2PatchOffsets()->assign(shape2PatchOffsets);

	CArray<int> patch2TriOffsets;
	patch2TriOffsets.pushBack(0);
	patch2TriOffsets.pushBack(1);
	patch2TriOffsets.pushBack(2);
	query.inPatch2TriOffsets()->assign(patch2TriOffsets);

	CArray<int> patch2TriIndices;
	patch2TriIndices.pushBack(0);
	patch2TriIndices.pushBack(1);
	query.inPatch2TriIndices()->assign(patch2TriIndices);

	auto triSet = std::make_shared<TriangleSet<DataType3f>>();
	CArray<Vec3f> vertices;
	vertices.pushBack(Vec3f(0.0f, 0.0f, 0.0f));
	vertices.pushBack(Vec3f(1.0f, 0.0f, 0.0f));
	vertices.pushBack(Vec3f(0.0f, 1.0f, 0.0f));
	vertices.pushBack(Vec3f(1.0f, 1.0f, 0.0f));
	triSet->getPoints().assign(vertices);

	CArray<TopologyModule::Triangle> triangles;
	triangles.pushBack(TopologyModule::Triangle(0, 1, 2));
	triangles.pushBack(TopologyModule::Triangle(1, 3, 2));
	triSet->triangleIndices().assign(triangles);
	query.inTriangleSet()->setDataPtr(triSet);

	query.update();
}
#endif
