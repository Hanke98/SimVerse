#pragma once
#include "CollisionData.h"

#include "DeclarePort.h"
#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"

#include "STL/Pair.h"

#include "Algorithm/Reduction.h"
#include "Algorithm/Scan.h"

#include "Topology/DiscreteElements.h"

#include "NeighborElementQuery.h"
#include "Vector/Vector3D.h"
#include <memory>
namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	struct TargetGroupInfo
	{
		int targetId;
		int groupStart;
		int groupCount;
		int tBegin;
		int tCount;
	};

	/**
	 * @brief A class implementation to find neighboring shapes for shape/rigid body collision
	 *
	 * @tparam TDataType
	 */

	template<typename TDataType>
	class NeighborTriMeshQuery : public NeighborElementQuery<TDataType>
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
		DEF_VAR(bool, EnableBroadPhasePatchPairs, false, "");

		DEF_ARRAY_IN(AABB, ShapeAABBs, DeviceType::GPU, "");

		// Patch AABBs in world-space rest pose coordinates.
		DEF_ARRAY_IN(AABB, PatchAABBs, DeviceType::GPU, "");

		// CSR: length = shapeCount + 1
		DEF_ARRAY_IN(int, Shape2PatchOffsets, DeviceType::GPU, "");

		// // Optional: length = shapeCount, used when Shape2PatchOffsets is not provided
		// DEF_ARRAY_IN(int, Shape2PatchCounts, DeviceType::GPU, "");

		// CSR: length = patchCount + 1
		DEF_ARRAY_IN(int, Patch2TriOffsets, DeviceType::GPU, "");

		// Triangle indices per patch (indices into TriangleSet::triangleIndices())
		DEF_ARRAY_IN(int, Patch2TriIndices, DeviceType::GPU, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		// Current rigid body pose arrays (indexed by bodyId)
		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "");

		// External mapping from shapeId to rigid body id 
		DEF_ARRAY_IN(int, Shape2RigidBodyIds, DeviceType::GPU, "");

		// // Indices mapping from shapeId to discrete element id
		DEF_ARRAY_IN(int, Shape2ElementIdsDense, DeviceType::GPU, "");
		
		// Optional: mapping from texture mesh shapeId to discrete elementId
		DEF_ARRAY_IN(PairUU, Shape2ElementIds, DeviceType::GPU, "");

		// Rest pose for each shape (same order as shapeAABBs / patch offsets)
		DEF_ARRAY_IN(Coord, RestShapeCenter, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RestShapeRotation, DeviceType::GPU, "");

		// Adjacency list for shapes, used when EnableAdjacentFilter is true
		DEF_ARRAYLIST_IN(int, AdjacentShapes, DeviceType::GPU, "");

		DEF_ARRAY_IN(int, Shape2TriOffsets, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialShapePairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialPatchPairs, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, PotentialTriSet, "");

	protected:
		void compute() override;
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();

	private:
		bool broadPhase();
		bool middlePhase();
		void narrowPhase();
		bool updateShape2RigidBodyIds(int shapeCount);
		bool updateShape2ElementIds(int shapeCount);
		bool buildPatchPairsFromContactList(int shapeCount, int patchCount);

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<int> mShape2PatchOffsets;
		DArray<int> mShape2ElementIds;
		DArray<uint> mPatch2Shape;
		DArray<int> mShape2RigidBodyIds;
		DArray<AABB> mShapeAabbsWorld;
		DArray<AABB> mPatchAabbsWorld;
		DArray<AABB> mTargetPatchAabbs;
		DArray<AABB> mSourcePatchAabbs;
		DArray<uint> mSource2PatchIds;
		DArray<int> mTouchedShapeFlags;
		DArray<int> mTouchedShapeOffsets;
		DArray<int> mTouchedShapeIds;
		DArray<int> mTargetShapeCounts;
		DArray<int> mTargetShapeOffsets;
		DArray<int> mTargetShapeWrite;
		DArray<int> mTarget2SourceShapes;
		DArray<Vec3f> mPatchRelCenterTrans;
		DArray<Mat3f> mPatchRelRotationTrans;
		DArray<Vec3f> mShapeRestT;
		DArray<Mat3f> mShapeRestR;
		DArray<int> mTargetActiveFlags;
		DArray<int> mTargetActiveOffsets;
		DArray<int> mTargetActiveIds;
		DArray<TargetGroupInfo> mActiveTargetInfos;
		DArray<int> mGroupSourcePatchCounts;
		DArray<int> mGroupSourcePatchOffsets;
		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mPatchBroadPhaseCD;

		bool mMappingReady = false;
		bool mWarnedEmptyMapping = false;
		bool mUseBroadPhasePatchPairs = false;
		bool mWarnedEmptyPatchMapping = false;
		bool mWarnedEmptyElementMapping = false;

	};
}
