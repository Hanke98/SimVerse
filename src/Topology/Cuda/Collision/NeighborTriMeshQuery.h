#pragma once
#include "CollisionData.h"

#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"

#include "STL/Pair.h"

#include "Algorithm/Reduction.h"
#include "Algorithm/Scan.h"

#include "Topology/DiscreteElements.h"

#include "NeighborElementQuery.h"
namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	/**
	 * @brief A class implementation to find neighboring shapes for shape/rigid body collision
	 *
	 * @tparam TDataType
	 */

	template<typename TDataType>
	class NeighborTriMeshQuery : public ComputeModule
	// class NeighborTriMeshQuery : public NeighborElementQuery<TDataType>
	{
		DECLARE_TCLASS(NeighborTriMeshQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename ::dyno::Pair<uint, uint> PairUU;

		NeighborTriMeshQuery();
		~NeighborTriMeshQuery() override;

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

		DEF_ARRAY_IN(AABB, ShapeAABBs, DeviceType::GPU, "");

		DEF_ARRAY_IN(AABB, PatchAABBs, DeviceType::GPU, "");

		// CSR: length = shapeCount + 1
		DEF_ARRAY_IN(int, Shape2PatchOffsets, DeviceType::GPU, "");

		// Optional: length = shapeCount, used when Shape2PatchOffsets is not provided
		DEF_ARRAY_IN(int, Shape2PatchCounts, DeviceType::GPU, "");

		// CSR: length = patchCount + 1
		DEF_ARRAY_IN(int, Patch2TriOffsets, DeviceType::GPU, "");

		// Triangle indices per patch (indices into TriangleSet::triangleIndices())
		DEF_ARRAY_IN(int, Patch2TriIndices, DeviceType::GPU, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		// Current rigid body pose arrays (indexed by bodyId)
		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "");

		// Deprecated: external mapping from shapeId to rigid body id (no longer required)
		DEF_ARRAY_IN(int, Shape2RigidBodyIds, DeviceType::GPU, "");

		// Rest pose for each shape (same order as shapeAABBs / patch offsets)
		DEF_ARRAY_IN(Coord, RestShapeCenter, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RestShapeRotation, DeviceType::GPU, "");

		// Optional: adjacency list for shapes, used when EnableAdjacentFilter is true
		DEF_ARRAYLIST_IN(int, AdjacentShapes, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialShapePairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialPatchPairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(ContactPair, Contacts, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
		bool broadPhase();
		bool middlePhase();
		void narrowPhase();
		bool updateShape2RigidBodyIds(int shapeCount);

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<int> mShape2PatchOffsets;
		DArray<uint> mPatch2Shape;
		DArray<int> mShape2RigidBodyIds;
		DArray<AABB> mShapeAabbsWorld;
		DArray<AABB> mPatchAabbsWorld;
		DArray<uint> mPatch2GlobalIds;
		DArray<AABB> mTargetPatchAabbs;
		DArray<AABB> mSourcePatchAabbs;
		DArray<uint> mSource2PatchIds;
		DArray<int> mTargetShapeCounts;
		DArray<int> mTargetShapeOffsets;
		DArray<int> mTargetShapeWrite;
		DArray<int> mTarget2SourceShapes;

		bool mMappingReady = false;
		bool mWarnedEmptyMapping = false;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};
}
