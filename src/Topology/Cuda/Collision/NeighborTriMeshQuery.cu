#include "Array/ArrayList.h"
#include "NeighborTriMeshQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "../../../Dynamics/Cuda/RigidBody/RigidBodySystem.h"
#include "Vector/Vector3D.h"
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

	__device__ inline int NLQ_WarpReduceSum(int v)
	{
		for (int offset = 16; offset > 0; offset >>= 1)
			v += __shfl_down_sync(0xffffffff, v, offset);
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

	// NOTE: localAabb is actually the patch AABB in rest world space (restWorldAabb).
	// R represents the rotation from rest to current (or an equivalent world rotation).
	// t represents the current world-space center of the patch AABB.
	// x_1 = R \cdot (x_0 - c_0) + t
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

	// Transform a world-space AABB into a target rest-pose coordinate system.
	__device__ inline AABB NLQ_TransformWorldAabbToRest(const AABB& worldAabb, const Mat3f& RRest, const Vec3f& tRest)
	{
		Vec3f centerWorld = (worldAabb.v0 + worldAabb.v1) * Real(0.5);
		Vec3f extentWorld = (worldAabb.v1 - worldAabb.v0) * Real(0.5);

		Vec3f centerLocal = RRest.transpose() * (centerWorld - tRest);

		Vec3f extentLocal;
		extentLocal[0] = fabs(RRest(0, 0)) * extentWorld[0] + fabs(RRest(1, 0)) * extentWorld[1] + fabs(RRest(2, 0)) * extentWorld[2];
		extentLocal[1] = fabs(RRest(0, 1)) * extentWorld[0] + fabs(RRest(1, 1)) * extentWorld[1] + fabs(RRest(2, 1)) * extentWorld[2];
		extentLocal[2] = fabs(RRest(0, 2)) * extentWorld[0] + fabs(RRest(1, 2)) * extentWorld[1] + fabs(RRest(2, 2)) * extentWorld[2];

		AABB localAabb;
		localAabb.v0 = centerLocal - extentLocal;
		localAabb.v1 = centerLocal + extentLocal;
		return localAabb;
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

	// template<typename Real, typename Coord, typename Matrix>
	inline void NLQ_GetRelativeTransformHost(
		int shapeId,
		const CArray<int>& shape2RigidBodyIds,
		const CArray<Vec3f>& centers,
		const CArray<Mat3f>& rotations,
		const CArray<Vec3f>& restShapeCenters,
		const CArray<Mat3f>& restShapeRotations,
		Mat3f& RRel,
		Vec3f& tRel,
		int& bodyId)
	{
		bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0 && shapeId >= 0 && shapeId < (int)shape2RigidBodyIds.size())
			bodyId = shape2RigidBodyIds[shapeId];

		if (bodyId < 0 || bodyId >= (int)centers.size() || bodyId >= (int)rotations.size())
		{
			RRel = Mat3f::identityMatrix();
			tRel = Vec3f(Real(0));
			return;
		}

		Vec3f tCurr = centers[bodyId];
		Mat3f RCurr = rotations[bodyId];

		Vec3f tRest = Vec3f(Real(0));
		Mat3f RRest = Mat3f::identityMatrix();
		if (shapeId >= 0 && shapeId < (int)restShapeCenters.size())
			tRest = restShapeCenters[shapeId];
		if (shapeId >= 0 && shapeId < (int)restShapeRotations.size())
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

	__global__ void NLQ_MarkActiveTargets(
		DArray<int> activeFlags,
		DArray<int> targetCounts,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapeCount) return;
		activeFlags[tId] = targetCounts[tId] > 0 ? 1 : 0;
	}

	__global__ void NLQ_CompactActiveTargets(
		DArray<int> activeIds,
		DArray<int> activeFlags,
		DArray<int> activeOffsets,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapeCount) return;
		if (activeFlags[tId] == 0) return;
		int out = activeOffsets[tId];
		if (out >= 0 && out < activeIds.size())
			activeIds[out] = tId;
	}

	__global__ void NLQ_BuildActiveTargetInfos(
		DArray<TargetGroupInfo> infos,
		DArray<int> activeIds,
		DArray<int> targetOffsets,
		DArray<int> targetCounts,
		DArray<int> shape2PatchOffsets,
		int patchCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= activeIds.size() || tId >= infos.size()) return;
		int target = activeIds[tId];
		int groupStart = target >= 0 && target < targetOffsets.size() ? targetOffsets[target] : -1;
		int groupCount = target >= 0 && target < targetCounts.size() ? targetCounts[target] : 0;
		int tBegin = (target + 1 < shape2PatchOffsets.size()) ? NLQ_ClampInt(shape2PatchOffsets[target], 0, patchCount) : 0;
		int tEnd = (target + 1 < shape2PatchOffsets.size()) ? NLQ_ClampInt(shape2PatchOffsets[target + 1], 0, patchCount) : 0;
		int tCount = tEnd - tBegin;
		TargetGroupInfo info;
		info.targetId = target;
		info.groupStart = groupStart;
		info.groupCount = groupCount;
		info.tBegin = tBegin;
		info.tCount = tCount;
		infos[tId] = info;
	}

	__global__ void NLQ_CountGroupSourcePatches(
		DArray<int> counts,
		DArray<int> groupedSources,
		DArray<int> shape2PatchOffsets,
		int patchCount,
		int groupStart,
		int groupCount)
	{
		int localId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (localId >= groupCount || localId >= counts.size()) return;
		int idx = groupStart + localId;
		if (idx < 0 || idx >= groupedSources.size())
		{
			counts[localId] = 0;
			return;
		}
		int shapeId = groupedSources[idx];
		if (shapeId < 0 || shapeId + 1 >= shape2PatchOffsets.size())
		{
			counts[localId] = 0;
			return;
		}
		int sBegin = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, patchCount);
		int sEnd = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, patchCount);
		int sCount = sEnd - sBegin;
		counts[localId] = sCount > 0 ? sCount : 0;
	}

	__global__ void NLQ_FillGroupPatchData(
		DArray<AABB> outAabbs,
		DArray<uint> outIds,
		DArray<AABB> patchAabbsWorld,
		DArray<int> groupedSources,
		DArray<int> shape2PatchOffsets,
		DArray<int> offsets,
		int patchCount,
		int groupStart,
		int groupCount)
	{
		int localId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (localId >= groupCount || localId >= offsets.size()) return;
		int idx = groupStart + localId; // global group index
		if (idx < 0 || idx >= groupedSources.size()) return;
		int shapeId = groupedSources[idx]; // source shape idx
		if (shapeId < 0 || shapeId + 1 >= shape2PatchOffsets.size()) return;
		int sBegin = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, patchCount); // source patch begin
		int sEnd = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, patchCount); // source patch end
		int sCount = sEnd - sBegin;// source patch count
		if (sCount <= 0) return;
		int outBase = offsets[localId]; // source patch's begin in the souce patch output array
		// TODO: parallel
		for (int i = 0; i < sCount; ++i)
		{
			int srcIdx = sBegin + i; // source patch index
			int outIdx = outBase + i; // source patch output array index
			if (srcIdx < 0 || srcIdx >= patchAabbsWorld.size()) continue;
			if (outIdx < 0 || outIdx >= outAabbs.size()) continue;
			outAabbs[outIdx] = patchAabbsWorld[srcIdx]; // output source patch aabb
			if (outIdx < outIds.size())
				outIds[outIdx] = (uint)srcIdx; 		 // output source patch id
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

	// template<typename Real, typename Coord, typename Matrix, typename AABB>
	// Update Patch AABBs from rest-local space to current-world space
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
			printf("[NeighborTriMeshQuery] UpdatePatchAabbs failure, patchId: %d, shapeId: %d\n", patchId, shapeId);
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

		worldAabbs[patchId] = NLQ_TransformLocalAabbToWorld(
			localAabbs[patchId],
			rotations[bodyId],
			centers[bodyId]);
	}

	__global__ void NLQ_UpdatePatchAabbsFromRestWorld(
		DArray<AABB> worldAabbs,
		DArray<AABB> restWorldAabbs,
		DArray<uint> patch2Shape,
		DArray<Mat3f> shapeRestR,
		DArray<Vec3f> shapeRestT)
	{
		int patchId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (patchId >= restWorldAabbs.size() || patchId >= worldAabbs.size())
			return;

		int shapeId = patchId < patch2Shape.size() ? (int)patch2Shape[patchId] : -1;
		if (shapeId < 0)
		{
			return;
		}

		Mat3f RRel = Mat3f::identityMatrix();
		Vec3f tRel = Vec3f(Real(0));
		if (shapeId >= 0 && shapeId < shapeRestR.size() && shapeId < shapeRestT.size())
		{
			RRel = shapeRestR[shapeId];
			tRel = shapeRestT[shapeId];
		}
			
		worldAabbs[patchId] = NLQ_TransformLocalAabbToWorld(
			restWorldAabbs[patchId],
			RRel,
			tRel);	
	}

	__global__ void NLQ_TransformPatchAabbsToTargetRest(
		DArray<AABB> aabbs,
		DArray<Mat3f> shapeRestR,
		DArray<Vec3f> shapeRestT,
		int targetId)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= aabbs.size()) return;
		if (targetId < 0 || targetId >= shapeRestR.size() || targetId >= shapeRestT.size()) return;
		Mat3f RRest = shapeRestR[targetId];
		Vec3f tRest = shapeRestT[targetId];
		aabbs[tId] = NLQ_TransformWorldAabbToRest(aabbs[tId], RRest, tRest);
	}

	template<typename TDataType, typename AABB>
	__global__ void NLQ_RequestIntersectionNumberBVH(
		DArray<uint> count,
		DArray<AABB> aabbs,
		LinearBVH<TDataType> bvh,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= aabbs.size()) return;

		if (self_collision)
			count[tId] = bvh.requestIntersectionNumber(aabbs[tId], tId);
		else
			count[tId] = bvh.requestIntersectionNumber(aabbs[tId]);
	}

	template<typename TDataType, typename AABB>
	__global__ void NLQ_RequestIntersectionIdsBVH(
		DArrayList<int> idLists,
		DArray<AABB> aabbs,
		LinearBVH<TDataType> bvh,
		bool self_collision)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= aabbs.size()) return;

		if (self_collision)
			bvh.requestIntersectionIds(idLists[tId], aabbs[tId], tId);
		else
			bvh.requestIntersectionIds(idLists[tId], aabbs[tId]);
	}

	__global__ void NLQ_MarkTouchedShapesFromPairs(
		DArray<int> shapeTouched,
		DArray<Pair<uint, uint>> shapePairs,
		int shapeCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> lp = shapePairs[tId];
		int s0 = (int)lp.first;
		int s1 = (int)lp.second;

		if (s0 >= 0 && s0 < shapeCount)
			atomicExch(&shapeTouched[s0], 1);
		if (s1 >= 0 && s1 < shapeCount)
			atomicExch(&shapeTouched[s1], 1);
	}

	__global__ void NLQ_CompactTouchedShapes(
		DArray<int> touchedShapeIds,
		DArray<int> shapeTouched,
		DArray<int> prefix,
		int shapeCount)
	{
		int shapeId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (shapeId >= shapeCount) return;

		if (shapeTouched[shapeId] == 0)
			return;

		int out = prefix[shapeId];
		if (out >= 0 && out < touchedShapeIds.size())
			touchedShapeIds[out] = shapeId;
	}

	__global__ void NLQ_UpdatePatchAabbsForTouchedShapes(
		DArray<AABB> worldAabbs,
		DArray<AABB> localAabbs,
		DArray<int> shape2PatchOffsets,
		DArray<int> touchedShapeIds,
		DArray<Vec3f> centers,
		DArray<Mat3f> rotations,
		DArray<int> shape2RigidBodyIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= touchedShapeIds.size()) return;

		int patchCount = (int)localAabbs.size();
		int worldCount = (int)worldAabbs.size();
		int safeCount = patchCount < worldCount ? patchCount : worldCount;
		if (safeCount <= 0)
			return;

		int shapeId = touchedShapeIds[tId];
		if (shapeId < 0 || shapeId + 1 >= shape2PatchOffsets.size())
			return;

		int start = NLQ_ClampInt(shape2PatchOffsets[shapeId], 0, safeCount);
		int end = NLQ_ClampInt(shape2PatchOffsets[shapeId + 1], 0, safeCount);
		if (end <= start)
			return;

		int bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0)
		{
			if (shapeId >= shape2RigidBodyIds.size())
			{
				for (int p = start; p < end; ++p)
					worldAabbs[p] = localAabbs[p];
				return;
			}
			bodyId = shape2RigidBodyIds[shapeId];
		}

		if (bodyId < 0 || bodyId >= centers.size() || bodyId >= rotations.size())
		{
			for (int p = start; p < end; ++p)
				worldAabbs[p] = localAabbs[p];
			return;
		}

		Mat3f R = rotations[bodyId];
		Vec3f t = centers[bodyId];
		for (int p = start; p < end; ++p)
		{
			worldAabbs[p] = NLQ_TransformLocalAabbToWorld(
				localAabbs[p],
				R,
				t);
		}
	}

	template<typename Box3D>
	__global__ void NTQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset,
		Real boundary_expand)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		ElementType eleType = elementOffset.checkElementType(tId);

		AABB box;
		switch (eleType)
		{
		case ET_SPHERE:
		{
			// FIX: elementId is global, need offset
			box = spheres[tId - elementOffset.sphereIndex()].aabb();
			break;
		}
		case ET_BOX:
		{
			box = boxes[tId - elementOffset.boxIndex()].aabb();
			break;
		}
		case ET_TET:
		{
			box = tets[tId - elementOffset.tetIndex()].aabb();
			break;
		}
		case ET_CAPSULE:
		{
			box = caps[tId - elementOffset.capsuleIndex()].aabb();
			break;
		}
		case ET_TRI:
		{
			boundary_expand = 0.01;
			box = tris[tId - elementOffset.triangleIndex()].aabb();
			break;
		}
		default:
			break;
		}

		box.v0 -= boundary_expand;
		box.v1 += boundary_expand;

		boundingBox[tId] = box;
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
		if (tId >= contactList.size()) return;

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
		// tId: local source patch id
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size())
			return;
		if (tId >= source2PatchIds.size())
			return;

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;
		uint srcId = source2PatchIds[tId];// global source patch id

		List<int>& list_i = contactList[tId];
		for (int j = 0; j < list_i.size(); j++)
		{
			// targetIdx: local target patch id
			int targetIdx = list_i[j];
			// ignore invalid local target indices
			if (targetIdx < 0 || targetIdx >= targetCount)
				continue;

			if (write < size && (offset + write) < patchPairs.size())
			{
				// (targetBase + targetIdx): global target patch id
				patchPairs[offset + write] = Pair<uint, uint>(srcId, (uint)(targetBase + targetIdx));
				write++;
			}
		}
	}

	template<typename AABB>
	__global__ void NLQ_CountPatchPairsSingleTarget(
		DArray<int> counts,
		DArray<AABB> sourceAabbs,
		DArray<AABB> targetAabbs,
		int targetIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sourceAabbs.size())
			return;
		if (targetIndex < 0 || targetIndex >= targetAabbs.size())
		{
			counts[tId] = 0;
			return;
		}

		counts[tId] = sourceAabbs[tId].checkOverlap(targetAabbs[targetIndex]) ? 1 : 0;
	}

	template<typename AABB>
	__global__ void NLQ_SetPatchPairsSingleTarget(
		DArray<Pair<uint, uint>> patchPairs,
		DArray<AABB> sourceAabbs,
		DArray<AABB> targetAabbs,
		DArray<uint> source2PatchIds,
		int targetIndex,
		int targetPatchId,
		DArray<int> prefix,
		DArray<int> counts)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sourceAabbs.size() || tId >= source2PatchIds.size())
			return;
		if (targetIndex < 0 || targetIndex >= targetAabbs.size())
			return;
		if (!sourceAabbs[tId].checkOverlap(targetAabbs[targetIndex]))
			return;

		int offset = prefix[tId];
		if (offset >= 0 && offset < patchPairs.size() && counts[tId] > 0)
		{
			patchPairs[offset] = Pair<uint, uint>(source2PatchIds[tId], (uint)targetPatchId);
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
		int warpId = tId >> 5;
		int lane = tId & 31;
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

	// Count the sizes of each triangle contact list
	__global__ void NLQ_Narrow_CountTriContactSizes(
		DArray<int> listSizes,
		DArrayList<int> triContactList)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triContactList.size()) return;

		listSizes[tId] = (int)triContactList[tId].size();
	}

	__global__ void NLQ_Narrow_FlattenTriContactList(
		DArray<int> pairListIds,
		DArray<int> pairCandidateOffsets,
		DArrayList<int> triContactList,
		DArray<int> listOffsets,
		DArray<int> listSizes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triContactList.size()) return;

		int base = listOffsets[tId];
		int count = listSizes[tId];
		if (count <= 0) return;

		List<int>& list = triContactList[tId];
		int listSize = (int)list.size();
		if (listSize < count)
			count = listSize;

		for (int i = 0; i < count; ++i)
		{
			int out = base + i;
			if (out >= pairListIds.size() || out >= pairCandidateOffsets.size())
				break;
			pairListIds[out] = tId;
			pairCandidateOffsets[out] = i;
		}
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void NLQ_Narrow_ThreadCount(
		DArray<int> counts,
		DArrayList<int> triContactList,
		DArray<int> triListTriIds,
		DArray<int> triListSide,
		DArray<int> pairListIds,
		DArray<int> pairCandidateOffsets,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		Real dHat,
		int triCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counts.size()) return;

		int listId = pairListIds[tId];
		int candOffset = pairCandidateOffsets[tId];
		if (listId < 0 || listId >= triContactList.size())
		{
			counts[tId] = 0;
			return;
		}

		int triIdCurrent = triListTriIds[listId];
		int side = triListSide[listId];
		if (triIdCurrent < 0 || triIdCurrent >= triCount)
		{
			counts[tId] = 0;
			return;
		}

		List<int>& list = triContactList[listId];
		if (candOffset < 0 || candOffset >= list.size())
		{
			counts[tId] = 0;
			return;
		}

		int triIdCandidate = list[candOffset];
		if (triIdCandidate < 0 || triIdCandidate >= triCount)
		{
			counts[tId] = 0;
			return;
		}

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
		counts[tId] = manifold.contactCount;
	}

	template<typename Real, typename Coord, typename Triangle, typename ContactPair>
	__global__ void NLQ_Narrow_ThreadSet(
		DArray<ContactPair> contacts,
		DArrayList<int> triContactList,
		DArray<int> triListTriIds,
		DArray<int> triListPairIds,
		DArray<int> triListSide,
		DArray<int> pairListIds,
		DArray<int> pairCandidateOffsets,
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
		if (tId >= counts.size()) return;

		int count = counts[tId];
		if (count <= 0) return;

		int listId = pairListIds[tId];
		int candOffset = pairCandidateOffsets[tId];
		if (listId < 0 || listId >= triContactList.size())
			return;

		int triIdCurrent = triListTriIds[listId];
		int side = triListSide[listId];
		int pairId = triListPairIds[listId];
		if (triIdCurrent < 0 || triIdCurrent >= triCount)
			return;
		if (pairId < 0 || pairId >= patchPairs.size())
			return;

		List<int>& list = triContactList[listId];
		if (candOffset < 0 || candOffset >= list.size())
			return;

		int triIdCandidate = list[candOffset];
		if (triIdCandidate < 0 || triIdCandidate >= triCount)
			return;

		Pair<uint, uint> pp = patchPairs[pairId];
		int patch0 = (int)pp.first;
		int patch1 = (int)pp.second;

		int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
		int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

		int bodyId0 = shape0;
		int bodyId1 = shape1;

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

		int writeBase = prefix[tId];
		int writeLimit = writeBase + count;

		for (int n = 0; n < manifold.contactCount; ++n)
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
		int patch0 = (int)pp.first; // source patch global id
		int patch1 = (int)pp.second; // target patch global id

		// if patch index out of range, then return
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

		// int shape0 = patch0 < patch2Shape.size() ? (int)patch2Shape[patch0] : -1;
		// int shape1 = patch1 < patch2Shape.size() ? (int)patch2Shape[patch1] : -1;

		// Matrix RRel0 = Matrix::identityMatrix();
		// Matrix RRel1 = Matrix::identityMatrix();
		// Coord tRel0 = Coord(Real(0));
		// Coord tRel1 = Coord(Real(0));
		// int bodyId0 = shape0;
		// int bodyId1 = shape1;

		// if (shape0 >= 0)
		// {	
		// 	// Get relative transform of shape0
		// 	NLQ_GetRelativeTransform(
		// 		shape0,
		// 		shape2RigidBodyIds,
		// 		centers,
		// 		rotations,
		// 		restShapeCenters,
		// 		restShapeRotations,
		// 		RRel0,
		// 		tRel0,
		// 		bodyId0);
		// }

		// if (shape1 >= 0)
		// {
		// 	// Get relative transform of shape1
		// 	NLQ_GetRelativeTransform(
		// 		shape1,
		// 		shape2RigidBodyIds,
		// 		centers,
		// 		rotations,
		// 		restShapeCenters,
		// 		restShapeRotations,
		// 		RRel1,
		// 		tRel1,
		// 		bodyId1);
		// }

		int cnt = 0;
		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patch2TriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;
			// compute triangle 0 in world space
			Triangle tri0 = triangles[triId0];
			// Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
			// Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
			// Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
			Coord p00 = vertices[tri0[0]];
			Coord p01 = vertices[tri0[1]];
			Coord p02 = vertices[tri0[2]];
			TTriangle3D<Real> t0(p00, p01, p02);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patch2TriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				// compute triangle 1 in world space
				Triangle tri1 = triangles[triId1];
				// Coord p10 = RRel1 * vertices[tri1[0]] + tRel1;
				// Coord p11 = RRel1 * vertices[tri1[1]] + tRel1;
				// Coord p12 = RRel1 * vertices[tri1[2]] + tRel1;
				Coord p10 = vertices[tri1[0]];
				Coord p11 = vertices[tri1[1]];
				Coord p12 = vertices[tri1[2]];
				TTriangle3D<Real> t1(p10, p11, p12);

				// perform narrow-phase collision detection
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

		// Matrix RRel0 = Matrix::identityMatrix();
		// Matrix RRel1 = Matrix::identityMatrix();
		// Coord tRel0 = Coord(Real(0));
		// Coord tRel1 = Coord(Real(0));
		int bodyId0 = shape0;
		int bodyId1 = shape1;

		// if (shape0 >= 0)
		// {
		// 	NLQ_GetRelativeTransform(
		// 		shape0,
		// 		shape2RigidBodyIds,
		// 		centers,
		// 		rotations,
		// 		restShapeCenters,
		// 		restShapeRotations,
		// 		RRel0,
		// 		tRel0,
		// 		bodyId0);
		// }

		// if (shape1 >= 0)
		// {
		// 	NLQ_GetRelativeTransform(
		// 		shape1,
		// 		shape2RigidBodyIds,
		// 		centers,
		// 		rotations,
		// 		restShapeCenters,
		// 		restShapeRotations,
		// 		RRel1,
		// 		tRel1,
		// 		bodyId1);
		// }

		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patch2TriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;

			Triangle tri0 = triangles[triId0];
			// Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
			// Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
			// Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
			Coord p00 = vertices[tri0[0]];
			Coord p01 = vertices[tri0[1]];
			Coord p02 = vertices[tri0[2]];
			TTriangle3D<Real> t0(p00, p01, p02);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patch2TriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				Triangle tri1 = triangles[triId1];
				// Coord p10 = RRel1 * vertices[tri1[0]] + tRel1;
				// Coord p11 = RRel1 * vertices[tri1[1]] + tRel1;
				// Coord p12 = RRel1 * vertices[tri1[2]] + tRel1;
				Coord p10 = vertices[tri1[0]];
				Coord p11 = vertices[tri1[1]];
				Coord p12 = vertices[tri1[2]];
				TTriangle3D<Real> t1(p10, p11, p12);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);

				// Process contacts information
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
					printf("[NeighborTriMeshQuery] Contact found between shape %d (tri %d) and shape %d (tri %d), penetration=%f\n",
                            shape0, triId0, shape1, triId1, cp.interpenetration);

					contacts[offset + write] = cp;
					write++;
				}
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
		this->mPatchBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
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
		if (this->inShapeAABBs()->isEmpty()
			|| this->inPatchAABBs()->isEmpty()
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
			return;
		}

		// auto& shapeAabbs = this->inShapeAABBs()->getData();
		// int shapeCount = (int)shapeAabbs.size();
		int shapeCount = this->inShape2ElementIds()->size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			printf("[NeighborTriMeshQuery] shapeAABBs is empty.\n");
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
			return;
		}

		if (!updateShape2ElementIds(shapeCount))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
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
			return;
		}

		// MiddlePhase: shape pairs + patch CSR -> patch pairs 
		if (!middlePhase())
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			// printf("[NeighborTriMeshQuery] MiddlePhase failed.\n");
			return;
		}

		// NarrowPhase: patch pairs + triangle CSR -> contacts
		narrowPhase();
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::broadPhase()
	{
		// printf("[NeighborTriMeshQuery] BroadPhase started.\n");
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

		std::cout << "[NeighborTriMeshQuery] broadPhase found " << total << " shape pairs." << std::endl;

		return true;
	}

	template<typename TDataType>
	bool NeighborTriMeshQuery<TDataType>::middlePhase()
	{
		printf("[NeighborTriMeshQuery] MiddlePhase started.\n");

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

		// Ensure mPatchAabbsWorld is properly sized before the kernel writes to it
		if (mPatchAabbsWorld.size() != patchCount)
			mPatchAabbsWorld.resize(patchCount);

		// Update patch AABBs in world space (current pose) from rest-world patch AABBs.
		// Notd: Is full update is necessary?
		cuExecute((uint)patchCount,
			NLQ_UpdatePatchAabbsFromRestWorld,
			mPatchAabbsWorld,
			patchAabbs,
			mPatch2Shape,
			mShapeRestR,
			mShapeRestT);
		cuSynchronize();


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
		cuExecute(shapePairs.size(),
			NLQ_GroupShapePairsByTarget,
			mTarget2SourceShapes,
			mTargetShapeOffsets,
			mTargetShapeWrite,
			shapePairs,
			shapeCount);
		// Target to source shape mapping built

		// Build compact active target list
		// Mark active target shapes that have at least one source shape
		if (mTargetActiveFlags.size() != (uint)shapeCount)
			mTargetActiveFlags.resize(shapeCount);
		if (mTargetActiveOffsets.size() != (uint)shapeCount)
			mTargetActiveOffsets.resize(shapeCount);
		mTargetActiveFlags.reset();
		cuExecute((uint)shapeCount,
			NLQ_MarkActiveTargets,
			mTargetActiveFlags,
			mTargetShapeCounts,
			shapeCount);
		int activeTargetCount = mReduce.accumulate(mTargetActiveFlags.begin(), mTargetActiveFlags.size());
		if (activeTargetCount <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			return false;
		}
		mTargetActiveOffsets.assign(mTargetActiveFlags);
		mScan.exclusive(mTargetActiveOffsets, true);

		// Setup active target ids
		if (mTargetActiveIds.size() != (uint)activeTargetCount)
			mTargetActiveIds.resize(activeTargetCount);
		cuExecute((uint)shapeCount,
			NLQ_CompactActiveTargets,
			mTargetActiveIds,
			mTargetActiveFlags,
			mTargetActiveOffsets,
			shapeCount);

		// Build active target infos for each target shape	
		if (mActiveTargetInfos.size() != (uint)activeTargetCount)
			mActiveTargetInfos.resize(activeTargetCount);
		cuExecute((uint)activeTargetCount,
			NLQ_BuildActiveTargetInfos,
			mActiveTargetInfos,
			mTargetActiveIds,
			mTargetShapeOffsets,
			mTargetShapeCounts,
			mShape2PatchOffsets,
			patchCount);
		cuSynchronize();

		CArray<TargetGroupInfo> hActiveInfos;
		hActiveInfos.assign(mActiveTargetInfos);

		CArray<int> hTargetPairCounts;
		hTargetPairCounts.assign((uint)shapeCount, 0);

		std::vector<std::unique_ptr<DArray<PairUU>>> targetPairs; // Patch pairs per target shape
		targetPairs.resize(shapeCount);
		DArray<uint> localBroadPhaseCounter;

		// NOTE: CollisionDetectionBroadPhase uses internal buffers and is not thread-safe for
		// concurrent update() calls, so targetShape groups are processed sequentially here.
		// The target->sources grouping keeps the layout ready for batched parallelization.
		for (uint ai = 0; ai < hActiveInfos.size(); ++ai)
		{
			TargetGroupInfo info = hActiveInfos[ai];
			int target = info.targetId;       // target shape ID
			int groupStart = info.groupStart; // start index in mTarget2SourceShapes
			int groupCount = info.groupCount; // number of source shapes in group
			int tBegin = info.tBegin;         // target shape's first patch index
			int tCount = info.tCount;         // target shape's patch count
			// skip empty groups
			if (groupCount <= 0) {
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has empty source group, skipping.\n", target);
				continue;
			}
			// skip invalid target shape
			if (groupStart < 0 || groupStart >= (int)mTarget2SourceShapes.size()) {
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has invalid groupStart %d, skipping.\n", target, groupStart);
				continue;
			}
			// Skip target shapes with no patches
			if (tCount <= 0) {
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has no patches, skipping.\n", target);
				continue;
			}

			// Assign target shape's patch AABBs
			if (mTargetPatchAabbs.size() != (uint)tCount)
				mTargetPatchAabbs.resize(tCount);
			mTargetPatchAabbs.assign(patchAabbs, tCount, 0, tBegin);

			// Build source patch lists on GPU for this target
			if (mGroupSourcePatchCounts.size() != (uint)groupCount)
				mGroupSourcePatchCounts.resize(groupCount);
			if (mGroupSourcePatchOffsets.size() != (uint)groupCount)
				mGroupSourcePatchOffsets.resize(groupCount);
			cuExecute((uint)groupCount,
				NLQ_CountGroupSourcePatches,
				mGroupSourcePatchCounts,
				mTarget2SourceShapes,
				mShape2PatchOffsets,
				patchCount,
				groupStart,
				groupCount);
			int sourceTotal = mReduce.accumulate(mGroupSourcePatchCounts.begin(), mGroupSourcePatchCounts.size());
			if (sourceTotal <= 0)
				continue;
			if (mSourcePatchAabbs.size() != (uint)sourceTotal)
				mSourcePatchAabbs.resize(sourceTotal);
			if (mSource2PatchIds.size() != (uint)sourceTotal)
				mSource2PatchIds.resize(sourceTotal);
			mGroupSourcePatchOffsets.assign(mGroupSourcePatchCounts);
			mScan.exclusive(mGroupSourcePatchOffsets, true);
			cuExecute((uint)groupCount,
				NLQ_FillGroupPatchData,
				mSourcePatchAabbs,
				mSource2PatchIds,
				mPatchAabbsWorld,
				mTarget2SourceShapes,
				mShape2PatchOffsets,
				mGroupSourcePatchOffsets,
				patchCount,
				groupStart,
				groupCount);

			// Keep source patch AABBs in world coordinates to match target rest-world frame.
			if ( sourceTotal > 0) // useTargetRestLocal &&
			{
				// Check if mSourcePatchAabbs is allocated
				if (mSourcePatchAabbs.size() == 0 || mSourcePatchAabbs.isEmpty()) {
					printf("[NeighborTriMeshQuery] ERROR: mSourcePatchAabbs is empty or not allocated\n");
					continue;
				}
				
				cuExecute((uint)sourceTotal,
					NLQ_TransformPatchAabbsToTargetRest,
					mSourcePatchAabbs,
					mShapeRestR,
					mShapeRestT,
					target);
			}
			
			// BVH traversal assumes at least two target nodes; handle single-patch targets directly.
			if (tCount == 1)
			{
				DArray<int> contactCount;
				contactCount.resize(sourceTotal);
				contactCount.reset();

				cuExecute((uint)sourceTotal,
					NLQ_CountPatchPairsSingleTarget,
					contactCount,
					mSourcePatchAabbs,
					mTargetPatchAabbs,
					0);

				int total = mReduce.accumulate(contactCount.begin(), contactCount.size());
				hTargetPairCounts[target] = total;
				printf("[NeighborTriMeshQuery] Target %d broadphase pairs=%d (single-patch)\n", target, total);
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

				cuExecute((uint)sourceTotal,
					NLQ_SetPatchPairsSingleTarget,
					*pairs,
					mSourcePatchAabbs,
					mTargetPatchAabbs,
					mSource2PatchIds,
					0,
					tBegin,
					contactCount,
					contactCountCpy);

				// Patch pairs found for this target
				targetPairs[target] = std::move(pairs);

				contactCountCpy.clear();
				contactCount.clear();
				continue;
			}

			/*
			auto patchBroadPhaseCD = this->mPatchBroadPhaseCD;
			patchBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
			patchBroadPhaseCD->varSelfCollision()->setValue(false);
			
			patchBroadPhaseCD->inSource()->assign(mSourcePatchAabbs);
			patchBroadPhaseCD->inTarget()->assign(mTargetPatchAabbs);
			patchBroadPhaseCD->inSource()->tick();
			patchBroadPhaseCD->inTarget()->tick();
			patchBroadPhaseCD->varForceUpdate()->setValue(true);

			auto type = this->varSpatial()->getDataPtr()->currentKey();
			switch (type)
			{
			case Spatial::BVH:
				patchBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
				break;
			case Spatial::OCTREE:
				patchBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
				break;
			default:
				break;
			}

			patchBroadPhaseCD->update();

			auto& contactList = patchBroadPhaseCD->outContactList()->getData();
			*/

			auto patchBroadPhaseCD = this->mPatchBroadPhaseCD;
			if (patchBroadPhaseCD->outContactList()->isEmpty())
			{
				patchBroadPhaseCD->outContactList()->allocate();
			}

			auto targetBVH = this->getShapeBVH(target);
			if (!targetBVH)
			{
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has no prebuilt BVH, skipping.\n", target);
				continue;
			}
			int bvhNodeCount = (int)targetBVH->getSortedAABBs().size();
			if (bvhNodeCount <= 0 || (bvhNodeCount % 2) == 0)
			{
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has invalid BVH node count (%d), skipping.\n",
					target, bvhNodeCount);
				continue;
			}
			int bvhLeafCount = (bvhNodeCount + 1) / 2;
			if (bvhLeafCount <= 1)
			{
				printf("[NeighborTriMeshQuery] WARNING: target shape %d has BVH leaf count %d, skipping.\n",
					target, bvhLeafCount);
				continue;
			}
			int targetLeafCount = bvhLeafCount;
			if (bvhLeafCount != tCount)
			{
				printf("[NeighborTriMeshQuery] WARNING: target shape %d BVH leaf count %d != tCount %d, clamping.\n",
					target, bvhLeafCount, tCount);
				targetLeafCount = bvhLeafCount < tCount ? bvhLeafCount : tCount;
			}

			auto& contactList = patchBroadPhaseCD->outContactList()->getData();

			if (localBroadPhaseCounter.size() != (uint)mSourcePatchAabbs.size())
				localBroadPhaseCounter.resize(mSourcePatchAabbs.size());

			cuExecute(mSourcePatchAabbs.size(),
				NLQ_RequestIntersectionNumberBVH,
				localBroadPhaseCounter,
				mSourcePatchAabbs,
				*targetBVH,
				false);

			contactList.resize(localBroadPhaseCounter);

			cuExecute(mSourcePatchAabbs.size(),
				NLQ_RequestIntersectionIdsBVH,
				contactList,
				mSourcePatchAabbs,
				*targetBVH,
				false);

			printf("[NeighborTriMeshQuery] Target %d middle phase contacts=%u (lists=%u)\n",
				target,
				(unsigned int)contactList.elementSize(),
				(unsigned int)contactList.size());
			if (contactList.elementSize() == 0)
			{
				// No contact detected
				continue;
			}

			DArray<int> contactCount;
			contactCount.resize(contactList.size());
			contactCount.reset();

			// count contacts for each source patch
			cuExecute(contactList.size(),
				NLQ_CountContactList,
				contactCount,
				contactList);
			// Contact list counted
			// reduce contact counts to get total patch pairs for this target shape
			int total = mReduce.accumulate(contactCount.begin(), contactCount.size());
			hTargetPairCounts[target] = total; // Store total patch pairs for this target shape
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

			// Set patch pairs of this target shape
			cuExecute(contactList.size(),
				NLQ_SetPatchPairsFromContactList,
				*pairs,
				contactList,
				contactCount,
				contactCountCpy,
				mSource2PatchIds,
				tBegin,
				targetLeafCount);
			// Patch pairs found for this target
			targetPairs[target] = std::move(pairs);

			contactCountCpy.clear();
			contactCount.clear();
		}

		int totalPairs = 0;
		CArray<int> hTargetPairOffsets;
		hTargetPairOffsets.resize(shapeCount);
		// Build target shape patch pair offsets and count total patch pairs
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

		std::cout << "[NeighborTriMeshQuery] middlePhase found " << totalPairs << " patch pairs." << std::endl;

		this->outPotentialPatchPairs()->resize(totalPairs);
		auto& patchPairs = this->outPotentialPatchPairs()->getData();

		// Flatten patch pairs from all target shapes into outPotentialPatchPairs()
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

		printf("[NeighborTriMeshQuery] MiddlePhase completed.\n");
		return true;
	}

	template<typename TDataType>
	void NeighborTriMeshQuery<TDataType>::narrowPhase()
	{
		printf("[NeighborTriMeshQuery] NarrowPhase started.\n");
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
		printf("[NeighborTriMeshQuery] NarrowPhase contact count computed.\n");

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
		printf("[NeighborTriMeshQuery] NarrowPhase contacts generated: %d contacts found.\n", total);
		
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

			// Matrix RRel0 = Matrix::identityMatrix();
			// Matrix RRel1 = Matrix::identityMatrix();
			// Coord tRel0 = Coord(Real(0));
			// Coord tRel1 = Coord(Real(0));
			int bodyId0 = shape0;
			int bodyId1 = shape1;

			// NLQ_GetRelativeTransformHost(
			// 	shape0,
			// 	hShape2Rigid,
			// 	hCenters,
			// 	hRotations,
			// 	hRestCenters,
			// 	hRestRotations,
			// 	RRel0,
			// 	tRel0,
			// 	bodyId0);

			// NLQ_GetRelativeTransformHost(
			// 	shape1,
			// 	hShape2Rigid,
			// 	hCenters,
			// 	hRotations,
			// 	hRestCenters,
			// 	hRestRotations,
			// 	RRel1,
			// 	tRel1,
			// 	bodyId1);

			Triangle tri0 = hTriangles[triId0];
			// Coord p00 = RRel0 * hVertices[tri0[0]] + tRel0;
			// Coord p01 = RRel0 * hVertices[tri0[1]] + tRel0;
			// Coord p02 = RRel0 * hVertices[tri0[2]] + tRel0;
			Coord p00 = hVertices[tri0[0]];
			Coord p01 = hVertices[tri0[1]];
			Coord p02 = hVertices[tri0[2]];
			int base = (int)contactVertices.size();
			contactVertices.push_back(p00);
			contactVertices.push_back(p01);
			contactVertices.push_back(p02);
			contactTriangles.push_back(Triangle(base, base + 1, base + 2));

			Triangle tri1 = hTriangles[triId1];
			// Coord p10 = RRel1 * hVertices[tri1[0]] + tRel1;
			// Coord p11 = RRel1 * hVertices[tri1[1]] + tRel1;
			// Coord p12 = RRel1 * hVertices[tri1[2]] + tRel1;
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
		printf("[NeighborTriMeshQuery] NarrowPhase completed.\n");
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
