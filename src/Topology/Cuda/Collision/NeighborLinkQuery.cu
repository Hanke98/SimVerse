#include "NeighborLinkQuery.h"

#include "CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include <iostream>

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborLinkQuery, TDataType)

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

	__global__ void NLQ_CountLinkPairs(
		DArray<int> counts,
		DArrayList<int> contactList,
		DArrayList<int> adjacentLinks,
		bool enableAdjacentFilter,
		int linkCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		List<int>& list_i = contactList[tId];
		int cnt = 0;
		bool useAdj = enableAdjacentFilter && tId < adjacentLinks.size();

		for (int j = 0; j < list_i.size(); j++)
		{
			int nb = list_i[j];
			if (nb <= tId || nb < 0 || nb >= linkCount)
				continue;

			if (useAdj && NLQ_IsAdjacent(adjacentLinks[tId], nb))
				continue;

			cnt++;
		}

		counts[tId] = cnt;
	}

	__global__ void NLQ_SetLinkPairs(
		DArray<Pair<uint, uint>> linkPairs,
		DArrayList<int> contactList,
		DArray<int> prefix,
		DArray<int> counts,
		DArrayList<int> adjacentLinks,
		bool enableAdjacentFilter,
		int linkCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactList.size()) return;

		List<int>& list_i = contactList[tId];
		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;
		bool useAdj = enableAdjacentFilter && tId < adjacentLinks.size();

		for (int j = 0; j < list_i.size(); j++)
		{
			int nb = list_i[j];
			if (nb <= tId || nb < 0 || nb >= linkCount)
				continue;

			if (useAdj && NLQ_IsAdjacent(adjacentLinks[tId], nb))
				continue;

			if (write < size && (offset + write) < linkPairs.size())
			{
				linkPairs[offset + write] = Pair<uint, uint>((uint)tId, (uint)nb);
				write++;
			}
		}
	}

	template<typename AABB>
	__global__ void NLQ_CountPatchPairs(
		DArray<int> counts,
		DArray<Pair<uint, uint>> linkPairs,
		DArray<int> linkPatchOffsets,
		DArray<AABB> patchAabbs,
		int patchCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= linkPairs.size()) return;

		Pair<uint, uint> lp = linkPairs[tId];
		int link0 = (int)lp.first;
		int link1 = (int)lp.second;

		if (link0 + 1 >= linkPatchOffsets.size() || link1 + 1 >= linkPatchOffsets.size())
		{
			counts[tId] = 0;
			return;
		}

		int start0 = NLQ_ClampInt(linkPatchOffsets[link0], 0, patchCount);
		int end0 = NLQ_ClampInt(linkPatchOffsets[link0 + 1], 0, patchCount);
		int start1 = NLQ_ClampInt(linkPatchOffsets[link1], 0, patchCount);
		int end1 = NLQ_ClampInt(linkPatchOffsets[link1 + 1], 0, patchCount);

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

	template<typename AABB>
	__global__ void NLQ_SetPatchPairs(
		DArray<Pair<uint, uint>> patchPairs,
		DArray<Pair<uint, uint>> linkPairs,
		DArray<int> linkPatchOffsets,
		DArray<AABB> patchAabbs,
		DArray<int> prefix,
		DArray<int> counts,
		int patchCount)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= linkPairs.size()) return;

		Pair<uint, uint> lp = linkPairs[tId];
		int link0 = (int)lp.first;
		int link1 = (int)lp.second;

		if (link0 + 1 >= linkPatchOffsets.size() || link1 + 1 >= linkPatchOffsets.size())
			return;

		int start0 = NLQ_ClampInt(linkPatchOffsets[link0], 0, patchCount);
		int end0 = NLQ_ClampInt(linkPatchOffsets[link0 + 1], 0, patchCount);
		int start1 = NLQ_ClampInt(linkPatchOffsets[link1], 0, patchCount);
		int end1 = NLQ_ClampInt(linkPatchOffsets[link1 + 1], 0, patchCount);

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

	__global__ void NLQ_BuildPatchLinkIds(
		DArray<uint> patchLinkIds,
		DArray<int> linkPatchOffsets,
		int patchCount)
	{
		int linkId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (linkId + 1 >= linkPatchOffsets.size()) return;

		int start = NLQ_ClampInt(linkPatchOffsets[linkId], 0, patchCount);
		int end = NLQ_ClampInt(linkPatchOffsets[linkId + 1], 0, patchCount);

		for (int p = start; p < end; ++p)
		{
			patchLinkIds[p] = (uint)linkId;
		}
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void NLQ_Narrow_Count(
		DArray<int> counts,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patchTriOffsets,
		DArray<int> patchTriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
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

		if (patch0 + 1 >= patchTriOffsets.size() || patch1 + 1 >= patchTriOffsets.size())
		{
			counts[tId] = 0;
			return;
		}

		int start0 = NLQ_ClampInt(patchTriOffsets[patch0], 0, patchTriCount);
		int end0 = NLQ_ClampInt(patchTriOffsets[patch0 + 1], 0, patchTriCount);
		int start1 = NLQ_ClampInt(patchTriOffsets[patch1], 0, patchTriCount);
		int end1 = NLQ_ClampInt(patchTriOffsets[patch1 + 1], 0, patchTriCount);

		int cnt = 0;
		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patchTriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;

			Triangle tri0 = triangles[triId0];
			TTriangle3D<Real> t0(vertices[tri0[0]], vertices[tri0[1]], vertices[tri0[2]]);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patchTriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				Triangle tri1 = triangles[triId1];
				TTriangle3D<Real> t1(vertices[tri1[0]], vertices[tri1[1]], vertices[tri1[2]]);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);

				cnt += manifold.contactCount;
			}
		}

		counts[tId] = cnt;
	}

	template<typename Real, typename Coord, typename Triangle, typename ContactPair>
	__global__ void NLQ_Narrow_Set(
		DArray<ContactPair> contacts,
		DArray<Pair<uint, uint>> patchPairs,
		DArray<int> patchTriOffsets,
		DArray<int> patchTriIndices,
		DArray<Coord> vertices,
		DArray<Triangle> triangles,
		DArray<uint> patchLinkIds,
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

		if (patch0 + 1 >= patchTriOffsets.size() || patch1 + 1 >= patchTriOffsets.size())
			return;

		int start0 = NLQ_ClampInt(patchTriOffsets[patch0], 0, patchTriCount);
		int end0 = NLQ_ClampInt(patchTriOffsets[patch0 + 1], 0, patchTriCount);
		int start1 = NLQ_ClampInt(patchTriOffsets[patch1], 0, patchTriCount);
		int end1 = NLQ_ClampInt(patchTriOffsets[patch1 + 1], 0, patchTriCount);

		int offset = prefix[tId];
		int size = counts[tId];
		int write = 0;

		int link0 = patch0 < patchLinkIds.size() ? (int)patchLinkIds[patch0] : -1;
		int link1 = patch1 < patchLinkIds.size() ? (int)patchLinkIds[patch1] : -1;

		for (int i = start0; i < end0; ++i)
		{
			int triId0 = patchTriIndices[i];
			if (triId0 < 0 || triId0 >= triCount)
				continue;

			Triangle tri0 = triangles[triId0];
			TTriangle3D<Real> t0(vertices[tri0[0]], vertices[tri0[1]], vertices[tri0[2]]);

			for (int j = start1; j < end1; ++j)
			{
				int triId1 = patchTriIndices[j];
				if (triId1 < 0 || triId1 >= triCount)
					continue;

				Triangle tri1 = triangles[triId1];
				TTriangle3D<Real> t1(vertices[tri1[0]], vertices[tri1[1]], vertices[tri1[2]]);

				TManifold<Real> manifold;
				CollisionDetection<Real>::request(manifold, t0, t1, dHat, dHat);

				for (int n = 0; n < manifold.contactCount; ++n)
				{
					if (write >= size || (offset + write) >= contacts.size())
						break;

					ContactPair cp;
					cp.bodyId1 = link0;
					cp.bodyId2 = link1;
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
	NeighborLinkQuery<TDataType>::NeighborLinkQuery()
		: ComputeModule()
	{
		this->inAdjacentLinks()->tagOptional(true);
		this->inLinkPatchCounts()->tagOptional(true);

		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();

		this->varGridSizeLimit()->setValue(Real(0.01));
		this->varDHead()->setValue(Real(0));
	}

	template<typename TDataType>
	NeighborLinkQuery<TDataType>::~NeighborLinkQuery()
	{
	}

	template<typename TDataType>
	void NeighborLinkQuery<TDataType>::compute()
	{
		if (this->outPotentialLinkPairs()->isEmpty())
			this->outPotentialLinkPairs()->allocate();
		if (this->outPotentialPatchPairs()->isEmpty())
			this->outPotentialPatchPairs()->allocate();
		if (this->outContacts()->isEmpty())
			this->outContacts()->allocate();

		if (this->inLinkAABBs()->isEmpty()
			|| this->inPatchAABBs()->isEmpty()
			|| (this->inLinkPatchOffsets()->isEmpty() && this->inLinkPatchCounts()->isEmpty())
			|| this->inPatchTriOffsets()->isEmpty()
			|| this->inPatchTriIndices()->isEmpty()
			|| this->inTriangleSet()->isEmpty())
		{
			this->outPotentialLinkPairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
            printf("[NeighborLinkQuery] Missing input data.\n");
			return;
		}

		auto& linkAabbs = this->inLinkAABBs()->getData();
		int linkCount = (int)linkAabbs.size();
		if (linkCount <= 0)
		{
			this->outPotentialLinkPairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
            printf("[NeighborLinkQuery] linkAABBs is empty.\n");
			return;
		}

		if (!this->inLinkPatchOffsets()->isEmpty())
		{
			mLinkPatchOffsets.assign(this->inLinkPatchOffsets()->getData());
		}
		else
		{
			auto& counts = this->inLinkPatchCounts()->getData();
			if ((int)counts.size() != linkCount)
			{
				this->outPotentialLinkPairs()->resize(0);
				this->outPotentialPatchPairs()->resize(0);
				this->outContacts()->resize(0);
				return;
			}

			mLinkPatchOffsets.resize(counts.size() + 1);
			mLinkPatchOffsets.reset();
			mLinkPatchOffsets.assign(counts, counts.size(), 0, 0);
			mScan.exclusive(mLinkPatchOffsets, true);
		}

		if (mLinkPatchOffsets.size() != (uint)(linkCount + 1))
		{
			this->outPotentialLinkPairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patchCount <= 0)
		{
			this->outPotentialLinkPairs()->resize(0);
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		if (mPatchLinkIds.size() != (uint)patchCount)
			mPatchLinkIds.resize(patchCount);
		mPatchLinkIds.reset();
		cuExecute(linkCount,
			NLQ_BuildPatchLinkIds,
			mPatchLinkIds,
			mLinkPatchOffsets,
			patchCount);

		// BroadPhase: link AABB overlap -> i<j link pairs (count -> scan -> write)
		if (!broadPhase())
		{
			this->outPotentialPatchPairs()->resize(0);
			this->outContacts()->resize(0);
			return;
		}

		// MiddlePhase: link pairs + patch CSR -> patch pairs (count -> scan -> write)
		if (!middlePhase())
		{
			this->outContacts()->resize(0);
			return;
		}

		// NarrowPhase: patch pairs + triangle CSR -> contacts (count -> scan -> write)
		narrowPhase();
	}

	template<typename TDataType>
	bool NeighborLinkQuery<TDataType>::broadPhase()
	{
		auto& linkAabbs = this->inLinkAABBs()->getData();
		int linkCount = (int)linkAabbs.size();
		if (linkCount <= 0)
		{
			this->outPotentialLinkPairs()->resize(0);
			return false;
		}

		mBroadPhaseCD->varGridSizeLimit()->setValue(this->varGridSizeLimit()->getValue());
		mBroadPhaseCD->varSelfCollision()->setValue(true);

		mBroadPhaseCD->inSource()->assign(linkAabbs);
		mBroadPhaseCD->inTarget()->assign(linkAabbs);

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
			this->outPotentialLinkPairs()->resize(0);
			return false;
		}

		DArray<int> pairCount;
		pairCount.resize(contactList.size());
		pairCount.reset();

		DArray<int> pairCountCpy;

		bool useAdj = this->varEnableAdjacentFilter()->getValue() && !this->inAdjacentLinks()->isEmpty();
		DArrayList<int> dummyAdj;
		auto& adj = useAdj ? this->inAdjacentLinks()->getData() : dummyAdj;

		cuExecute(contactList.size(),
			NLQ_CountLinkPairs,
			pairCount,
			contactList,
			adj,
			useAdj,
			linkCount);

		int total = mReduce.accumulate(pairCount.begin(), pairCount.size());
		if (total <= 0)
		{
			this->outPotentialLinkPairs()->resize(0);
			pairCount.clear();
			pairCountCpy.clear();
			return false;
		}

		pairCountCpy.assign(pairCount);
		mScan.exclusive(pairCount, true);

		this->outPotentialLinkPairs()->resize(total);

		cuExecute(contactList.size(),
			NLQ_SetLinkPairs,
			this->outPotentialLinkPairs()->getData(),
			contactList,
			pairCount,
			pairCountCpy,
			adj,
			useAdj,
			linkCount);

		pairCountCpy.clear();
		pairCount.clear();

		return true;
	}

	template<typename TDataType>
	bool NeighborLinkQuery<TDataType>::middlePhase()
	{
		auto& linkPairs = this->outPotentialLinkPairs()->getData();
		if (linkPairs.size() == 0)
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

		DArray<int> pairCount;
		pairCount.resize(linkPairs.size());
		pairCount.reset();

		DArray<int> pairCountCpy;

		cuExecute(linkPairs.size(),
			NLQ_CountPatchPairs,
			pairCount,
			linkPairs,
			mLinkPatchOffsets,
			patchAabbs,
			patchCount);

		int total = mReduce.accumulate(pairCount.begin(), pairCount.size());
		if (total <= 0)
		{
			this->outPotentialPatchPairs()->resize(0);
			pairCount.clear();
			pairCountCpy.clear();
			return false;
		}

		pairCountCpy.assign(pairCount);
		mScan.exclusive(pairCount, true);

		this->outPotentialPatchPairs()->resize(total);

		cuExecute(linkPairs.size(),
			NLQ_SetPatchPairs,
			this->outPotentialPatchPairs()->getData(),
			linkPairs,
			mLinkPatchOffsets,
			patchAabbs,
			pairCount,
			pairCountCpy,
			patchCount);

		pairCountCpy.clear();
		pairCount.clear();

		return true;
	}

	template<typename TDataType>
	void NeighborLinkQuery<TDataType>::narrowPhase()
	{
		auto& patchPairs = this->outPotentialPatchPairs()->getData();
		if (patchPairs.size() == 0)
		{
			this->outContacts()->resize(0);
			return;
		}

		auto& patchTriOffsets = this->inPatchTriOffsets()->getData();
		auto& patchTriIndices = this->inPatchTriIndices()->getData();

		int patchCount = (int)this->inPatchAABBs()->size();
		if (patchTriOffsets.size() < (uint)(patchCount + 1) || patchTriIndices.size() == 0)
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

		int patchTriCount = (int)patchTriIndices.size();

		DArray<int> contactCount;
		contactCount.resize(patchPairs.size());
		contactCount.reset();

		DArray<int> contactCountCpy;

		Real dHat = this->varDHead()->getValue();

		cuExecute(patchPairs.size(),
			NLQ_Narrow_Count,
			contactCount,
			patchPairs,
			patchTriOffsets,
			patchTriIndices,
			vertices,
			triIndices,
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
			patchTriOffsets,
			patchTriIndices,
			vertices,
			triIndices,
			mPatchLinkIds,
			contactCount,
			contactCountCpy,
			dHat,
			patchCount,
			triCount,
			patchTriCount);

		contactCountCpy.clear();
		contactCount.clear();
	}

	DEFINE_CLASS(NeighborLinkQuery);
}

#ifdef UNIT_TEST
#include "Topology/TriangleSet.h"

void NeighborLinkQuery_UnitTest()
{
	using namespace dyno;

	NeighborLinkQuery<DataType3f> query;

	CArray<NeighborLinkQuery<DataType3f>::AABB> linkAabbs;
	linkAabbs.pushBack(NeighborLinkQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
	linkAabbs.pushBack(NeighborLinkQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
	query.inLinkAABBs()->assign(linkAabbs);

	CArray<NeighborLinkQuery<DataType3f>::AABB> patchAabbs;
	patchAabbs.pushBack(NeighborLinkQuery<DataType3f>::AABB(Vec3f(0.0f), Vec3f(1.0f)));
	patchAabbs.pushBack(NeighborLinkQuery<DataType3f>::AABB(Vec3f(0.5f), Vec3f(1.5f)));
	query.inPatchAABBs()->assign(patchAabbs);

	CArray<int> linkPatchOffsets;
	linkPatchOffsets.pushBack(0);
	linkPatchOffsets.pushBack(1);
	linkPatchOffsets.pushBack(2);
	query.inLinkPatchOffsets()->assign(linkPatchOffsets);

	CArray<int> patchTriOffsets;
	patchTriOffsets.pushBack(0);
	patchTriOffsets.pushBack(1);
	patchTriOffsets.pushBack(2);
	query.inPatchTriOffsets()->assign(patchTriOffsets);

	CArray<int> patchTriIndices;
	patchTriIndices.pushBack(0);
	patchTriIndices.pushBack(1);
	query.inPatchTriIndices()->assign(patchTriIndices);

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
