#include "NeighborMeshQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "Primitive/Primitive3D.h"
#include "Topology/TriangleSet.h"

#include <vector>

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborMeshQuery, TDataType)

	__device__ inline int NMQ_ClampInt(int v, int lo, int hi)
	{
		return v < lo ? lo : (v > hi ? hi : v);
	}

	__device__ inline bool NMQ_IsAdjacent(List<int>& adj, int other)
	{
		for (int i = 0; i < adj.size(); ++i)
		{
			if (adj[i] == other)
				return true;
		}
		return false;
	}

	template<typename Real, typename Coord, typename Matrix>
	__device__ inline void NMQ_GetRelativeTransform(
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

	inline int NMQ_ClampIntHost(int v, int lo, int hi)
	{
		return v < lo ? lo : (v > hi ? hi : v);
	}

	template<typename Real, typename Coord, typename Matrix>
	inline void NMQ_GetRelativeTransformHost(
		int shapeId,
		const CArray<int>& shape2RigidBodyIds,
		const CArray<Coord>& centers,
		const CArray<Matrix>& rotations,
		const CArray<Coord>& restShapeCenters,
		const CArray<Matrix>& restShapeRotations,
		Matrix& RRel,
		Coord& tRel,
		int& bodyId)
	{
		bodyId = shapeId;
		if (shape2RigidBodyIds.size() > 0 && shapeId >= 0 && shapeId < (int)shape2RigidBodyIds.size())
			bodyId = shape2RigidBodyIds[shapeId];

		if (bodyId < 0 || bodyId >= (int)centers.size() || bodyId >= (int)rotations.size())
		{
			RRel = Matrix::identityMatrix();
			tRel = Coord(Real(0));
			return;
		}

		Coord tCurr = centers[bodyId];
		Matrix RCurr = rotations[bodyId];

		Coord tRest = Coord(Real(0));
		Matrix RRest = Matrix::identityMatrix();
		if (shapeId >= 0 && shapeId < (int)restShapeCenters.size())
			tRest = restShapeCenters[shapeId];
		if (shapeId >= 0 && shapeId < (int)restShapeRotations.size())
			RRest = restShapeRotations[shapeId];

		RRel = RCurr * RRest.transpose();
		tRel = tCurr - RRel * tRest;
	}

	inline void NMQ_BuildShapeTriCSR(
		const CArray<int>& shape2PatchOffsets,
		const CArray<int>& patch2TriOffsets,
		const CArray<int>& patch2TriIndices,
		int shapeCount,
		int patchCount,
		std::vector<int>& shape2TriOffsets,
		std::vector<int>& shape2TriIndices)
	{
		shape2TriOffsets.assign(shapeCount + 1, 0);
		shape2TriIndices.clear();
		shape2TriIndices.reserve(patch2TriIndices.size());

		int patchTriCount = (int)patch2TriIndices.size();
		for (int shapeId = 0; shapeId < shapeCount; ++shapeId)
		{
			int pStart = NMQ_ClampIntHost(shape2PatchOffsets[shapeId], 0, patchCount);
			int pEnd = NMQ_ClampIntHost(shape2PatchOffsets[shapeId + 1], 0, patchCount);

			for (int p = pStart; p < pEnd; ++p)
			{
				if (p + 1 >= (int)patch2TriOffsets.size())
					break;

				int tStart = NMQ_ClampIntHost(patch2TriOffsets[p], 0, patchTriCount);
				int tEnd = NMQ_ClampIntHost(patch2TriOffsets[p + 1], 0, patchTriCount);
				for (int t = tStart; t < tEnd; ++t)
					shape2TriIndices.push_back(patch2TriIndices[t]);
			}

			shape2TriOffsets[shapeId + 1] = (int)shape2TriIndices.size();
		}
	}

    inline void NMQ_BuildShapeTriCSR(
        CArray<TopologyModule::Triangle> triangles,
		CArray<int>& cShape2TriOffsets,
        std::vector<int>& shape2TriOffsets,
		std::vector<int>& shape2TriIndices)
	{
		int triCount = triangles.size();
		int shapeCount = cShape2TriOffsets.size() - 1;
        shape2TriIndices.clear();
		shape2TriIndices.reserve(triCount);
		shape2TriOffsets.clear();
		shape2TriOffsets.reserve(shapeCount + 1);
        
        for (int triId = 0; triId < triCount; ++triId){
            shape2TriIndices.push_back(triId);
        }

        for (int shapeId = 0; shapeId < cShape2TriOffsets.size(); ++shapeId) {
            shape2TriOffsets.push_back(cShape2TriOffsets[shapeId]);
        }
		
	}

	template<typename Box3D>
	__global__ void NMQ_SetupAABBFromElementIds(
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

	__global__ void NMQ_CountShapePairs(
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

			if (useAdj && NMQ_IsAdjacent(adjacentShapes[tId], nb))
				continue;

			cnt++;
		}

		counts[tId] = cnt;
	}

	__global__ void NMQ_SetShapePairs(
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
		int count = counts[tId];
		int write = 0;
		bool useAdj = enableAdjacentFilter && tId < adjacentShapes.size();

		for (int j = 0; j < list_i.size(); j++)
		{
			int nb = list_i[j];
			if (nb <= tId || nb < 0 || nb >= shapeCount)
				continue;

			if (useAdj && NMQ_IsAdjacent(adjacentShapes[tId], nb))
				continue;

			if (offset + write >= shapePairs.size())
				break;

			shapePairs[offset + write] = Pair<uint, uint>((uint)tId, (uint)nb);
			write++;

			if (write >= count)
				break;
		}
	}

	template<typename Real, typename Coord, typename Matrix, typename Triangle>
	__global__ void NMQ_Narrow_Count(
		DArray<int> counts,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		DArray<int> patch2TriOffsets,
		DArray<int> patch2TriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<Coord> centers,
		DArray<Matrix> rotations,
		DArray<Coord> restShapeCenters,
		DArray<Matrix> restShapeRotations,
		DArray<int> shape2RigidBodyIds,
		Real dHat,
		int shapeCount,
		int patchCount,
		int triCount,
		int patchTriCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> sp = shapePairs[tId];
		int shape0 = (int)sp.first;
		int shape1 = (int)sp.second;

		if (shape0 < 0 || shape1 < 0 || shape0 >= shapeCount || shape1 >= shapeCount)
		{
			counts[tId] = 0;
			return;
		}

		if (shape0 + 1 >= shape2PatchOffsets.size() || shape1 + 1 >= shape2PatchOffsets.size())
		{
			counts[tId] = 0;
			return;
		}

		int s0Start = NMQ_ClampInt(shape2PatchOffsets[shape0], 0, patchCount);
		int s0End = NMQ_ClampInt(shape2PatchOffsets[shape0 + 1], 0, patchCount);
		int s1Start = NMQ_ClampInt(shape2PatchOffsets[shape1], 0, patchCount);
		int s1End = NMQ_ClampInt(shape2PatchOffsets[shape1 + 1], 0, patchCount);

		if (s0End <= s0Start || s1End <= s1Start)
		{
			counts[tId] = 0;
			return;
		}

		Matrix RRel0 = Matrix::identityMatrix();
		Matrix RRel1 = Matrix::identityMatrix();
		Coord tRel0 = Coord(Real(0));
		Coord tRel1 = Coord(Real(0));
		int bodyId0 = shape0;
		int bodyId1 = shape1;

		NMQ_GetRelativeTransform<Real, Coord, Matrix>(
			shape0,
			shape2RigidBodyIds,
			centers,
			rotations,
			restShapeCenters,
			restShapeRotations,
			RRel0,
			tRel0,
			bodyId0);

		NMQ_GetRelativeTransform<Real, Coord, Matrix>(
			shape1,
			shape2RigidBodyIds,
			centers,
			rotations,
			restShapeCenters,
			restShapeRotations,
			RRel1,
			tRel1,
			bodyId1);

		int cnt = 0;
		for (int p0 = s0Start; p0 < s0End; ++p0)
		{
			if (p0 + 1 >= patch2TriOffsets.size())
				continue;

			int t0Start = NMQ_ClampInt(patch2TriOffsets[p0], 0, patchTriCount);
			int t0End = NMQ_ClampInt(patch2TriOffsets[p0 + 1], 0, patchTriCount);

			for (int i = t0Start; i < t0End; ++i)
			{
				int triId0 = patch2TriIndices[i];
				if (triId0 < 0 || triId0 >= triCount)
					continue;

				Triangle tri0 = triangles[triId0];
				Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
				Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
				Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
				TTriangle3D<Real> t0(p00, p01, p02);

				for (int p1 = s1Start; p1 < s1End; ++p1)
				{
					if (p1 + 1 >= patch2TriOffsets.size())
						continue;

					int t1Start = NMQ_ClampInt(patch2TriOffsets[p1], 0, patchTriCount);
					int t1End = NMQ_ClampInt(patch2TriOffsets[p1 + 1], 0, patchTriCount);

					for (int j = t1Start; j < t1End; ++j)
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
			}
		}

		counts[tId] = cnt;
	}

	template<typename Real, typename Coord, typename Matrix, typename Triangle, typename ContactPair>
	__global__ void NMQ_Narrow_Set(
		DArray<ContactPair> contacts,
		DArray<Pair<uint, uint>> shapePairs,
		DArray<int> shape2PatchOffsets,
		DArray<int> patch2TriOffsets,
		DArray<int> patch2TriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<Coord> centers,
		DArray<Matrix> rotations,
		DArray<Coord> restShapeCenters,
		DArray<Matrix> restShapeRotations,
		DArray<int> shape2RigidBodyIds,
		DArray<int> prefix,
		DArray<int> counts,
		Real dHat,
		int shapeCount,
		int patchCount,
		int triCount,
		int patchTriCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapePairs.size()) return;

		Pair<uint, uint> sp = shapePairs[tId];
		int shape0 = (int)sp.first;
		int shape1 = (int)sp.second;

		if (shape0 < 0 || shape1 < 0 || shape0 >= shapeCount || shape1 >= shapeCount)
			return;

		if (shape0 + 1 >= shape2PatchOffsets.size() || shape1 + 1 >= shape2PatchOffsets.size())
			return;

		int s0Start = NMQ_ClampInt(shape2PatchOffsets[shape0], 0, patchCount);
		int s0End = NMQ_ClampInt(shape2PatchOffsets[shape0 + 1], 0, patchCount);
		int s1Start = NMQ_ClampInt(shape2PatchOffsets[shape1], 0, patchCount);
		int s1End = NMQ_ClampInt(shape2PatchOffsets[shape1 + 1], 0, patchCount);

		if (s0End <= s0Start || s1End <= s1Start)
			return;

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;

		Matrix RRel0 = Matrix::identityMatrix();
		Matrix RRel1 = Matrix::identityMatrix();
		Coord tRel0 = Coord(Real(0));
		Coord tRel1 = Coord(Real(0));
		int bodyId0 = shape0;
		int bodyId1 = shape1;

		NMQ_GetRelativeTransform<Real, Coord, Matrix>(
			shape0,
			shape2RigidBodyIds,
			centers,
			rotations,
			restShapeCenters,
			restShapeRotations,
			RRel0,
			tRel0,
			bodyId0);

		NMQ_GetRelativeTransform<Real, Coord, Matrix>(
			shape1,
			shape2RigidBodyIds,
			centers,
			rotations,
			restShapeCenters,
			restShapeRotations,
			RRel1,
			tRel1,
			bodyId1);

		for (int p0 = s0Start; p0 < s0End; ++p0)
		{
			if (p0 + 1 >= patch2TriOffsets.size())
				continue;

			int t0Start = NMQ_ClampInt(patch2TriOffsets[p0], 0, patchTriCount);
			int t0End = NMQ_ClampInt(patch2TriOffsets[p0 + 1], 0, patchTriCount);

			for (int i = t0Start; i < t0End; ++i)
			{
				int triId0 = patch2TriIndices[i];
				if (triId0 < 0 || triId0 >= triCount)
					continue;

				Triangle tri0 = triangles[triId0];
				Coord p00 = RRel0 * vertices[tri0[0]] + tRel0;
				Coord p01 = RRel0 * vertices[tri0[1]] + tRel0;
				Coord p02 = RRel0 * vertices[tri0[2]] + tRel0;
				TTriangle3D<Real> t0(p00, p01, p02);

				for (int p1 = s1Start; p1 < s1End; ++p1)
				{
					if (p1 + 1 >= patch2TriOffsets.size())
						continue;

					int t1Start = NMQ_ClampInt(patch2TriOffsets[p1], 0, patchTriCount);
					int t1End = NMQ_ClampInt(patch2TriOffsets[p1 + 1], 0, patchTriCount);

					for (int j = t1Start; j < t1End; ++j)
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
		}
	}

	template<typename TDataType>
	NeighborMeshQuery<TDataType>::NeighborMeshQuery()
		: NeighborTriMeshQuery<TDataType>()
	{
		this->varGridSizeLimit()->setValue(Real(0.01));
		this->varDHead()->setValue(Real(0));
	}

	template<typename TDataType>
	NeighborMeshQuery<TDataType>::~NeighborMeshQuery()
	{
	}

	template<typename TDataType>
	bool NeighborMeshQuery<TDataType>::updateShape2ElementIds(int shapeCount)
	{
		if (shapeCount <= 0)
			return false;

		const uint invalidElementId = static_cast<uint>(-1);

		if (!this->inShape2ElementIds()->isEmpty())
		{
			auto& pairs = this->inShape2ElementIds()->getData();
			if (pairs.size() != (uint)shapeCount)
			{
				if (!mMeshWarnedEmptyElementMapping)
				{
					printf("[NeighborMeshQuery] Shape2ElementIds size mismatch (shapeCount=%d, pairCount=%u), skip this frame.\n",
						shapeCount,
						(unsigned int)pairs.size());
					mMeshWarnedEmptyElementMapping = true;
				}
				return false;
			}

			CArray<Pair<uint, uint>> hostPairs;
			hostPairs.assign(pairs);

			std::vector<int> shape2ElementIds(shapeCount, -1);
			bool warnedDuplicate = false;
			for (uint i = 0; i < hostPairs.size(); ++i)
			{
				uint shapeId = hostPairs[i].first;
				uint elementId = hostPairs[i].second;

				if (elementId == invalidElementId)
					continue;
				if (shapeId >= (uint)shapeCount)
					continue;

				if (shape2ElementIds[shapeId] >= 0 && !warnedDuplicate)
				{
					printf("[NeighborMeshQuery] Shape2ElementPairs has duplicate shapeId=%u, overwriting.\n", shapeId);
					warnedDuplicate = true;
				}
				shape2ElementIds[shapeId] = (int)elementId;
			}

			bool allReady = true;
			for (int i = 0; i < shapeCount; ++i)
			{
				if (shape2ElementIds[i] < 0)
					allReady = false;
			}

			if (!allReady)
			{
				if (!mMeshWarnedEmptyElementMapping)
				{
					printf("[NeighborMeshQuery] Shape2ElementMapping incomplete (shapeCount=%d), skip this frame.\n", shapeCount);
					mMeshWarnedEmptyElementMapping = true;
				}
				return false;
			}

			auto topo = this->inDiscreteElements()->getDataPtr();
			if (topo == nullptr)
			{
				if (!mMeshWarnedEmptyElementMapping)
				{
					printf("[NeighborMeshQuery] Shape2ElementMapping not ready yet (topology unavailable, shapeCount=%d), skip this frame.\n",
						shapeCount);
					mMeshWarnedEmptyElementMapping = true;
				}
				return false;
			}

			auto& mapping = topo->shape2RigidBodyMapping();
			if (mapping.size() == 0)
			{
				if (!mMeshWarnedEmptyMapping)
				{
					printf("[NeighborMeshQuery] Shape2RigidBodyMapping not ready yet (shapeCount=%d, mappingSize=%u), skip this frame.\n",
						shapeCount,
						(unsigned int)mapping.size());
					mMeshWarnedEmptyMapping = true;
				}
				return false;
			}

			CArray<Pair<uint, uint>> hostMapping;
			hostMapping.assign(mapping);

			uint totalSize = topo->totalSize();
			if (totalSize == 0)
			{
				if (!mMeshWarnedEmptyMapping)
				{
					printf("[NeighborMeshQuery] Shape2RigidBodyMapping not ready yet (totalSize=0), skip this frame.\n");
					mMeshWarnedEmptyMapping = true;
				}
				return false;
			}

			std::vector<int> element2Rigid(totalSize, -1);
			for (uint i = 0; i < hostMapping.size(); ++i)
			{
				uint elementId = hostMapping[i].first;
				if (elementId < totalSize)
					element2Rigid[elementId] = (int)hostMapping[i].second;
			}

			std::vector<int> shape2RigidBodyIds(shapeCount, -1);
			bool rigidReady = true;
			for (int i = 0; i < shapeCount; ++i)
			{
				int elementId = shape2ElementIds[i];
				if (elementId < 0 || (uint)elementId >= totalSize)
				{
					rigidReady = false;
					continue;
				}
				int bodyId = element2Rigid[elementId];
				if (bodyId < 0)
				{
					rigidReady = false;
					continue;
				}
				shape2RigidBodyIds[i] = bodyId;
			}

			if (!rigidReady)
			{
				if (!mMeshWarnedEmptyMapping)
				{
					printf("[NeighborMeshQuery] Shape2RigidBodyMapping incomplete (shapeCount=%d), skip this frame.\n", shapeCount);
					mMeshWarnedEmptyMapping = true;
				}
				return false;
			}

			mShape2ElementIds.assign(shape2ElementIds);
			mShape2RigidBodyIds.assign(shape2RigidBodyIds);
			mMeshWarnedEmptyElementMapping = false;
			mMeshWarnedEmptyMapping = false;

			return true;
		}
		else
		{
			printf("[NeighborMeshQuery] Shape2ElementIds input is empty, skip this frame.\n");
			return false;
		}
	}

	template<typename TDataType>
	void NeighborMeshQuery<TDataType>::compute()
	{
		if (this->outPotentialShapePairs()->isEmpty())
			this->outPotentialShapePairs()->allocate();
		if (this->outPotentialPatchPairs()->isEmpty())
			this->outPotentialPatchPairs()->allocate();
		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();
		if (this->outPotentialTriSet()->isEmpty())
			this->outPotentialTriSet()->allocate();
		// auto potentialTriSet = this->outPotentialTriSet()->getDataPtr();

		if (this->inShape2TriOffsets()->isEmpty()
			|| this->inCenter()->isEmpty()
			|| this->inRotationMatrix()->isEmpty()
			|| this->inRestShapeCenter()->isEmpty()
			|| this->inRestShapeRotation()->isEmpty()
			|| this->inTriangleSet()->isEmpty())
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->outPotentialTriSet()->getDataPtr()->clear();
			printf("[NeighborMeshQuery] Missing input data.\n");
			return;
		}

		int shapeCount = this->inShape2ElementIds()->size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->outPotentialTriSet()->getDataPtr()->clear();
			printf("[NeighborMeshQuery] shape count is empty.\n");
			return;
		}

		if (!updateShape2ElementIds(shapeCount))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->outPotentialTriSet()->getDataPtr()->clear();
			return;
		}

		auto& shape2TriOffsetsData = this->inShape2TriOffsets()->getData();
		if (shape2TriOffsetsData.size() != (uint)(shapeCount + 1))
		{
			this->outPotentialShapePairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			this->outPotentialTriSet()->getDataPtr()->clear();
			printf("[NeighborMeshQuery] Shape2TriOffsets size mismatch.\n");
			return;
		}

		this->outPotentialPatchPairs()->resize(0);

		if (!broadPhase())
		{
			this->outContacts()->resize(0);
			this->outPotentialTriSet()->getDataPtr()->clear();
			return;
		}

		narrowPhase();
	}

	template<typename TDataType>
	bool NeighborMeshQuery<TDataType>::broadPhase()
	{
		auto inTopo = this->inDiscreteElements()->getDataPtr();
		if (inTopo == nullptr)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborMeshQuery] DiscreteElements missing.\n");
			return false;
		}

		int shapeCount = (int)mShape2ElementIds.size();
		if (shapeCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborMeshQuery] Shape2ElementIds size mismatch.\n");
			return false;
		}

		int elementCount = inTopo->totalSize();
		if (elementCount <= 0)
		{
			this->outPotentialShapePairs()->resize(0);
			printf("[NeighborMeshQuery] DiscreteElements size mismatch.\n");
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

		cuExecute((uint)shapeCount,
			NMQ_SetupAABBFromElementIds,
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
		case NeighborTriMeshQuery<TDataType>::Spatial::BVH:
			this->mBroadPhaseCD->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
			break;
		case NeighborTriMeshQuery<TDataType>::Spatial::OCTREE:
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
			NMQ_CountShapePairs,
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
			NMQ_SetShapePairs,
			this->outPotentialShapePairs()->getData(),
			contactList,
			pairCount,
			pairCountCpy,
			adj,
			useAdj,
			shapeCount);

		pairCountCpy.clear();
		pairCount.clear();

		printf("[NeighborMeshQuery] broadPhase found %d shape pairs.\n", total);

		return true;
	}

	template<typename TDataType>
	void NeighborMeshQuery<TDataType>::narrowPhase()
	{
		auto& shapePairs = this->outPotentialShapePairs()->getData();
		if (shapePairs.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

		auto& shape2TriOffsetsData = this->inShape2TriOffsets()->getData();
		int shapeCount = (int)shape2TriOffsetsData.size() - 1;
		if (shapeCount <= 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

		CArray<Pair<uint, uint>> hShapePairs;
		hShapePairs.assign(shapePairs);
		CArray<Coord> hVertices;
		hVertices.assign(ts->getPoints());
		CArray<Triangle> hTriangles;
		hTriangles.assign(ts->triangleIndices());

		if (hTriangles.size() == 0 || hVertices.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

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

		std::vector<int> shape2TriOffsets;
        CArray<int> cShape2TriOffsets;
        cShape2TriOffsets.assign(this->inShape2TriOffsets()->getData());

		// std::vector<int> shape2TriIndices;
		// NMQ_BuildShapeTriCSR(
        //     hTriangles,
		// 	cShape2TriOffsets,
        //     shape2TriOffsets,
		// 	shape2TriIndices);

		if (cShape2TriOffsets.size() != (size_t)(shapeCount + 1))
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

		Real dHat = this->varDHead()->getValue();
		int triCount = (int)hTriangles.size();

		std::vector<ContactPair> contacts;
		contacts.reserve(hShapePairs.size());

		std::vector<Coord> contactVertices;
		std::vector<Triangle> contactTriangles;

		for (uint i = 0; i < hShapePairs.size(); ++i)
		{
			int shape0 = (int)hShapePairs[i].first;
			int shape1 = (int)hShapePairs[i].second;
			if (shape0 < 0 || shape1 < 0 || shape0 >= shapeCount || shape1 >= shapeCount)
				continue;

			// int triStart0 = shape2TriOffsets[shape0];
			// int triEnd0 = shape2TriOffsets[shape0 + 1];
			// int triStart1 = shape2TriOffsets[shape1];
			// int triEnd1 = shape2TriOffsets[shape1 + 1];
			int triStart0 = cShape2TriOffsets[shape0];
			int triEnd0 = cShape2TriOffsets[shape0 + 1];
			int triStart1 = cShape2TriOffsets[shape1];
			int triEnd1 = cShape2TriOffsets[shape1 + 1];
			if (triEnd0 <= triStart0 || triEnd1 <= triStart1)
				continue;

			// Matrix RRel0 = Matrix::identityMatrix();
			// Matrix RRel1 = Matrix::identityMatrix();
			// Coord tRel0 = Coord(Real(0));
			// Coord tRel1 = Coord(Real(0));
			int bodyId0 = hShape2Rigid[shape0];
			int bodyId1 = hShape2Rigid[shape1];

			// NMQ_GetRelativeTransformHost<Real, Coord, Matrix>(
			// 	shape0,
			// 	hShape2Rigid,
			// 	hCenters,
			// 	hRotations,
			// 	hRestCenters,
			// 	hRestRotations,
			// 	RRel0,
			// 	tRel0,
			// 	bodyId0);

			// NMQ_GetRelativeTransformHost<Real, Coord, Matrix>(
			// 	shape1,
			// 	hShape2Rigid,
			// 	hCenters,
			// 	hRotations,
			// 	hRestCenters,
			// 	hRestRotations,
			// 	RRel1,
			// 	tRel1,
			// 	bodyId1);

			if (bodyId0 >= 0 && bodyId1 >= 0 && bodyId0 == bodyId1)
				continue;

			for (int a = triStart0; a < triEnd0; ++a)
			{
				// int triId0 = shape2TriIndices[a];
				int triId0 = a;
				if (triId0 < 0 || triId0 >= triCount)
					continue;

				Triangle tri0 = hTriangles[triId0];
				// Coord p00 = RRel0 * hVertices[tri0[0]] + tRel0;
				// Coord p01 = RRel0 * hVertices[tri0[1]] + tRel0;
				// Coord p02 = RRel0 * hVertices[tri0[2]] + tRel0;
				Coord p00 = hVertices[tri0[0]];
				Coord p01 = hVertices[tri0[1]];
				Coord p02 = hVertices[tri0[2]];
				TTriangle3D<Real> t0(p00, p01, p02);

				for (int b = triStart1; b < triEnd1; ++b)
				{
					// int triId1 = shape2TriIndices[b];
					int triId1 = b;
					if (triId1 < 0 || triId1 >= triCount)
						continue;

					Triangle tri1 = hTriangles[triId1];
					// Coord p10 = RRel1 * hVertices[tri1[0]] + tRel1;
					// Coord p11 = RRel1 * hVertices[tri1[1]] + tRel1;
					// Coord p12 = RRel1 * hVertices[tri1[2]] + tRel1;
					Coord p10 = hVertices[tri1[0]];
					Coord p11 = hVertices[tri1[1]];
					Coord p12 = hVertices[tri1[2]];
					TTriangle3D<Real> t1(p10, p11, p12);

					TManifold<Real> manifold;
					CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);
					// CollisionDetection<Real>::request(manifold, t0, t1, 0.01, 0.1);

					if (manifold.contactCount > 0)
					{
						int base = (int)contactVertices.size();
						contactVertices.push_back(p00);
						contactVertices.push_back(p01);
						contactVertices.push_back(p02);
						contactTriangles.push_back(Triangle(base, base + 1, base + 2));

						base = (int)contactVertices.size();
						contactVertices.push_back(p10);
						contactVertices.push_back(p11);
						contactVertices.push_back(p12);
						contactTriangles.push_back(Triangle(base, base + 1, base + 2));

						// printf("[NeighborMeshQuery] Contact triangles added: shape %d (tri %d) and shape %d (tri %d)\n",
						// 	shape0, triId0, shape1, triId1);
					}

					for (int n = 0; n < manifold.contactCount; ++n)
					{
						// ContactPair cp;
						// cp.bodyId1 = bodyId0;
						// cp.bodyId2 = bodyId1;
						// // cp.localId1 = triId0;
						// // cp.localId2 = triId1;
						// cp.pos1 = manifold.contacts[n].position;
						// cp.pos2 = manifold.contacts[n].position;
						// cp.normal1 = -manifold.normal;
						// cp.normal2 = manifold.normal;
						// cp.contactType = ContactType::CT_NONPENETRATION;
						// cp.interpenetration = -manifold.contacts[n].penetration;

						ContactPair cp;
						cp.pos1 = manifold.contacts[n].position + dHat * manifold.normal;
						cp.pos2 = manifold.contacts[n].position + dHat * manifold.normal;
						// cp.pos1 = manifold.contacts[n].position;
						// cp.pos2 = manifold.contacts[n].position;
						cp.normal1 = -manifold.normal;
						cp.normal2 = manifold.normal;
						cp.bodyId1 = bodyId0;
						cp.bodyId2 = bodyId1;
						cp.contactType = ContactType::CT_NONPENETRATION;
						cp.interpenetration = -manifold.contacts[n].penetration - 2 * dHat;
						// cp.interpenetration = -manifold.contacts[n].penetration;
                        printf("[NeighborMeshQuery] Contact found between shape %d (tri %d) and shape %d (tri %d), penetration=%f\n",
                            shape0, triId0, shape1, triId1, cp.interpenetration);

						contacts.push_back(cp);
					}
				}
			}
		}

		if (contacts.empty())
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			return;
		}

		CArray<ContactPair> hContacts;
		hContacts.assign(contacts);
		this->outContacts()->assign(hContacts);

		if (contactTriangles.empty())
		{
			this->triSet->clear();
			return;
		}
		// this->triSet->clear();
		this->triSet->setPoints(contactVertices);
		this->triSet->setTriangles(contactTriangles);
		this->triSet->update();

		if (this->triSet->isEmpty())
		{
			printf("[NeighborMeshQuery] triSet update failed.\n");
		}
		else
		{
			printf("[NeighborMeshQuery] triSet updated: %u vertices, %u triangles.\n",
				(unsigned int)this->triSet->getPointSize(),
				(unsigned int)this->triSet->triangleIndices().size());
			this->outPotentialTriSet()->setDataPtr(this->triSet);
		}
	}

	DEFINE_CLASS(NeighborMeshQuery);
}
