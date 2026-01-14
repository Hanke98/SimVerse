#pragma once
#include "CollisionData.h"

#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"

#include "STL/Pair.h"

#include "Algorithm/Reduction.h"
#include "Algorithm/Scan.h"

namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	/**
	 * @brief A class implementation to find neighboring links for robot arm collision
	 *
	 * @tparam TDataType
	 */
	template<typename TDataType>
	class NeighborLinkQuery : public ComputeModule
	{
		DECLARE_TCLASS(NeighborLinkQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename ::dyno::Pair<uint, uint> PairUU;

		NeighborLinkQuery();
		~NeighborLinkQuery() override;

	public:
		DECLARE_ENUM(Spatial,
			BVH = 0,
			OCTREE = 1);

		DEF_ENUM(Spatial, Spatial, Spatial::BVH, "");

		DEF_VAR(bool, EnableAdjacentFilter, false, "");

		DEF_VAR(Real, DHead, Real(0.0), "D head");

		/**
		* @brief A positive value indicating the size of the smallest element, its value will also influence the level of Octree or hierarchical BVH
		*/
		DEF_VAR(Real, GridSizeLimit, Real(0.01), "Indicate the size of the smallest element");

		DEF_ARRAY_IN(AABB, LinkAABBs, DeviceType::GPU, "");

		DEF_ARRAY_IN(AABB, PatchAABBs, DeviceType::GPU, "");

		// CSR: length = linkCount + 1
		DEF_ARRAY_IN(int, LinkPatchOffsets, DeviceType::GPU, "");

		// Optional: length = linkCount, used when LinkPatchOffsets is not provided
		DEF_ARRAY_IN(int, LinkPatchCounts, DeviceType::GPU, "");

		// CSR: length = patchCount + 1
		DEF_ARRAY_IN(int, PatchTriOffsets, DeviceType::GPU, "");

		// Triangle indices per patch (indices into TriangleSet::triangleIndices())
		DEF_ARRAY_IN(int, PatchTriIndices, DeviceType::GPU, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		// Optional: adjacency list for links, used when EnableAdjacentFilter is true
		DEF_ARRAYLIST_IN(int, AdjacentLinks, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialLinkPairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialPatchPairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(ContactPair, Contacts, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
		bool broadPhase();
		bool middlePhase();
		void narrowPhase();

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<int> mLinkPatchOffsets;
		DArray<uint> mPatchLinkIds;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};
}
