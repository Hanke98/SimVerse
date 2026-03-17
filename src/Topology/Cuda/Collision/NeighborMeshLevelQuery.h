#pragma once
#include "NeighborTriMeshQuery.h"

namespace dyno
{
	template<typename TDataType>
	class NeighborMeshLevelQuery : public NeighborTriMeshQuery<TDataType>
	{
		DECLARE_TCLASS(NeighborMeshLevelQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Tri2Edg Tri2Edg;
		typedef typename TopologyModule::Edg2Tri Edg2Tri;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename ::dyno::Pair<uint, uint> PairUU;

		NeighborMeshLevelQuery();
		~NeighborMeshLevelQuery() override;

	protected:
		bool initializeImpl() override;
		void narrowPhase() override;

	private:
		void initializeTopologyOwnershipCache(bool forceRebuild = false);
		void clearTopologyOwnershipCache();
		bool updateTriShapeLookup(bool forceRebuild = false);
		void buildCollisionTriSet();

	private:
		DArrayList<int> mVertexAdjacentFaces;
		DArrayList<int> mVertexIncidentEdges;
		DArray<Edg2Tri> mEdgeAdjacentFaces;
		DArray<Tri2Edg> mTriangleEdges;
		DArray<Edge> mEdgeVertexIndices;

		DArray<int> mFaceAssignedVertexOffsets;
		DArray<int> mFaceAssignedVertexIndices;
		DArray<int> mFaceAssignedEdgeOffsets;
		DArray<int> mFaceAssignedEdgeIndices;

		DArray<int> mVertexAssignedFaces;
		DArray<int> mEdgeAssignedFaces;
		DArray<Coord> mEdgeNormals;
		DArray<int> mTri2Shape;

		DArray<AABB> mTriangleAabbsWorld;
		DArray<Coord> mFaceNormalsWorld;
		DArray<Coord> mEdgeNormalsWorld;

		DArray<int> mPatchPairTriPairCounts;
		DArray<int> mPatchPairTriPairOffsets;
		DArray<int> mCandidateTri0;
		DArray<int> mCandidateTri1;
		DArray<int> mCandidatePatchPairId;

		DArray<int> mCoarsePassCounts;
		DArray<int> mCoarsePassOffsets;
		DArray<int> mFilteredTri0;
		DArray<int> mFilteredTri1;
		DArray<int> mFilteredPatchPairId;

		DArray<int> mPrimitivePassCounts;
		DArray<int> mPrimitivePassOffsets;
		DArray<ContactPair> mPrimitiveCandidateContacts;
			DArray<unsigned long long> mPrimitiveCandidateKeys;
			DArray<int> mPrimitiveCandidateSortedIndices;
			DArray<int> mPrimitiveCandidateKeepFlags;
			DArray<int> mSelectedPrimitiveCounts;

			DArray<int> mTriPairContactCounts;
			DArray<int> mTriPairContactOffsets;

		bool mTopologyOwnershipReady = false;
		bool mTriShapeReady = false;

		Scan<int> mScan;
		Reduction<int> mReduce;
	};
}
