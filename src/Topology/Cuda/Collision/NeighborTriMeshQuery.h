#pragma once
#include "CollisionData.h"

#include "DeclarePort.h"
#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"

#include "STL/Pair.h"

#include "Algorithm/Reduction.h"
#include "Algorithm/Scan.h"
#include "Array/ArrayList.h"

#include "Topology/DiscreteElements.h"
#include "Topology/LinearBVH.h"

#include "NeighborElementQuery.h"
#include "Vector/Vector3D.h"
#include <memory>
#include <vector>
namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

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
		using ShapeBVHList = std::vector<std::shared_ptr<LinearBVH<TDataType>>>;

		NeighborTriMeshQuery();
		~NeighborTriMeshQuery() override;

	public:
		DECLARE_ENUM(Spatial,
			BVH = 0,
			OCTREE = 1);

		DEF_ENUM(Spatial, Spatial, Spatial::BVH, "");

		DEF_VAR(bool, EnableAdjacentFilter, false, "");
		DEF_VAR(bool, EnableBroadPhasePatchPairs, false, "");
		DEF_VAR(bool, DisableContactReduction, false, "");
		// If true, TriangleSet::points are provided in rest-world space (static), and the query will transform
		// them to current world space using per-shape relative transforms.
		DEF_VAR(bool, InputVerticesInRestWorld, false, "");

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

		// Rest pose for each shape (same order as patch offsets)
		DEF_ARRAY_IN(Coord, RestShapeCenter, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RestShapeRotation, DeviceType::GPU, "");

		// Adjacency list for shapes, used when EnableAdjacentFilter is true
		DEF_ARRAYLIST_IN(int, AdjacentShapes, DeviceType::GPU, "");

		DEF_ARRAY_IN(int, Shape2TriOffsets, DeviceType::GPU, "");

		DEF_VAR_IN(ShapeBVHList, ShapeBVHs, "");

		DEF_VAR_IN(Bool, EnableVisualizeCollisionTriSet, "Enable visualize collision triSet mesh");

		DEF_ARRAY_OUT(PairUU, PotentialShapePairs, DeviceType::GPU, "");

		DEF_ARRAY_OUT(PairUU, PotentialPatchPairs, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, PotentialTriSet, "");

	public:
		std::shared_ptr<LinearBVH<TDataType>> getShapeBVH(int shapeId)
		{
			auto& shapeBVHs = this->inShapeBVHs()->constDataPtr();
			if (shapeBVHs == nullptr)
				return nullptr;
			if (shapeId < 0 || shapeId >= static_cast<int>(shapeBVHs->size()))
				return nullptr;
			return (*shapeBVHs)[shapeId];
		}

		// Initialize static shape->patch CSR once from external setup code (e.g. BatchRigidBodySystem).
		bool setStaticShape2PatchOffsets(const std::vector<int>& offsets);
		// Initialize static patch->shape lookup once from external setup code (e.g. BatchRigidBodySystem).
		bool setStaticPatch2Shape(const std::vector<uint>& patch2Shape);
		// Initialize static target BVH cache once from external setup code (e.g. BatchRigidBodySystem).
		bool setStaticTargetBVHCache(const ShapeBVHList& shapeBVHs);

	protected:
		void compute() override;
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		virtual void narrowPhase();
		const DArray<uint>& patch2ShapeData() const { return mPatch2Shape; }
		const DArray<Vec3f>& shapeRestTranslationsData() const { return mShapeRestT; }
		const DArray<Mat3f>& shapeRestRotationsData() const { return mShapeRestR; }

	private:
		bool broadPhase();
		bool middlePhase();
		bool updatePatchFaceLimitState(int patchCount);
		bool updateShape2RigidBodyIds(int shapeCount);
		bool buildPatchPairsFromContactList(int shapeCount, int patchCount);
		void ensureMiddleWorkspace(int totalSource);
		void ensureNarrowWorkspace(int totalTriLists);
		void clearWorkspace();

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<int> mShape2PatchOffsets;
		DArray<uint> mPatch2Shape;
		DArray<AABB> mShapeAabbsWorld;
		DArray<AABB> mPatchAabbsWorld;
		DArray<AABB> mSourcePatchAabbs;
		DArray<uint> mSource2PatchIds;
		DArray<int> mTouchedShapeFlags;
		DArray<int> mTouchedShapeOffsets;
		DArray<int> mTouchedShapeIds;
		DArray<int> mTargetShapeCounts;
		DArray<int> mTargetShapeOffsets;
		DArray<int> mTarget2SourceShapes;
		DArray<int> mSortedPairTargets;
		DArray<Vec3f> mPatchRelCenterTrans;
		DArray<Mat3f> mPatchRelRotationTrans;
		DArray<Vec3f> mShapeRestT;
		DArray<Mat3f> mShapeRestR;
		DArray<int> mGroup2PatchCounts;
		DArray<int> mGroup2PatchOffsets;
		DArray<int> mGroup2GlobalOffsets;
		DArray<int> mTarget2SourceCounts;
		DArray<int> mTarget2SourceOffsets;
		DArray<int> mGroup2TargetIds;
		DArray<int> mSource2TargetIds;
		DArray<LinearBVH<TDataType>> mTargetBVHs;
		DArray<int> mTargetBVHValid;
		DArray<uint> mMiddleLocalBroadPhaseCounter;
		DArrayList<int> mMiddleContactList;
		DArray<int> mMiddleContactCount;
		DArray<int> mMiddleContactCountCpy;
		DArray<int> mNarrowTriListSizes;
		DArray<int> mNarrowTriListOffsets;
		DArrayList<int> mNarrowTriContactList;
		DArray<int> mNarrowTriListTriIds;
		DArray<int> mNarrowTriListPairIds;
		DArray<int> mNarrowTriListSide;
		DArray<int> mNarrowTriContactCounts;
		DArray<int> mNarrowTriContactOffsets;
		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mPatchBroadPhaseCD;

		bool mMappingReady = false;
		bool mWarnedEmptyMapping = false;
		bool mUseBroadPhasePatchPairs = false;
		bool mWarnedEmptyPatchMapping = false;
		bool mWarnedEmptyElementMapping = false;
		bool mPatchFaceLimitReady = false;
		bool mPatchFaceLimitValid = false;
		bool mWarnedPatchFaceLimit = false;
		bool mStaticShape2PatchOffsetsReady = false;
		bool mStaticPatch2ShapeReady = false;
		bool mStaticTargetBVHCacheReady = false;
		int mCachedPatchCount = -1;
		uint mCachedPatch2TriOffsetsSize = 0;
		uint mCachedPatch2TriIndicesSize = 0;
		int mCachedMaxPatchFaces = 0;
	};
}
