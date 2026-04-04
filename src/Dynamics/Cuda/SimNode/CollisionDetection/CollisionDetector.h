#pragma once

#include <memory>
#include <vector>

#include <Algorithm/Reduction.h>
#include <Algorithm/Scan.h>
#include <Array/Array.h>
#include <Array/ArrayList.h>
#include <Array/Array2D.h>
#include <STL/Pair.h>
#include <Topology/TriangleSet.h>

#include "MeshCollisionTypes.h"

namespace dyno {

    template<typename TDataType>
    struct RigidBody;

    struct BatchCollisionConstraints;

    template<typename TDataType>
    class CollisionDetectionBroadPhase;

    template<typename TDataType>
    class MeshCollisionDetector
    {
    public:
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using Matrix = typename TDataType::Matrix;
        using AABB = TAlignedBox3D<Real>;
        using Triangle = TopologyModule::Triangle;
        using PairUU = Pair<uint, uint>;
        using ContactPair = TContactPair<Real>;

        MeshCollisionDetector() {};
        ~MeshCollisionDetector() {};

        void Initialize(int num_envs, int max_bodies, const RigidBody<TDataType>& rb);
        void Detect(const RigidBody<TDataType>& rb, BatchCollisionConstraints& out, int num_envs);
        void DetectGround(const RigidBody<TDataType>& rb, BatchCollisionConstraints& out, int num_envs);

    private:
        bool broad_phase(const RigidBody<TDataType>& rb, int num_envs);
        bool middle_phase(const RigidBody<TDataType>& rb, int num_envs, std::vector<BodyPair>& bodyPairsHost);
        void narrow_phase(const RigidBody<TDataType>& rb,
            BatchCollisionConstraints& out,
            int num_envs);
        void refreshMeshShapeLayoutCache(int shapeCount);
        void detectMeshMeshInternal(const RigidBody<TDataType>& rb,
            BatchCollisionConstraints& out,
            int num_envs);

    private:
        bool m_initialized = false;
        int m_numEnvs = 0;
        int m_maxBodies = 0;
        int m_cachedMeshShapeCount = -1;
        Real m_dHat = Real(1e-3);
        Real m_edgeEdgeActivationMargin = Real(3e-3);

        MeshTemplateData<TDataType> m_cubeTemplate;
        std::shared_ptr<TriangleSet<TDataType>> m_cubeTemplateTriSet;
        std::vector<Coord> m_cubeVerticesHost;
        std::vector<Triangle> m_cubeTrianglesHost;

        DArray<AABB> m_bodyAABBs;
        DArray<BodyPair> m_bodyPairs;
        std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_bodyBroadPhase;

        DArray<PairUU> m_shapePairs;
        DArray<int> m_shape2BodyFlat;
        DArray<Coord> m_shapeCenters;
        DArray<Matrix> m_shapeRotations;
        DArray<Coord> m_shapeHalfLengths;
        DArray<Coord> m_shapeInvHalfLengths;

        DArray<int> m_shape2PatchOffsets;
        DArray<int> m_shape2TriOffsets;
        DArray<int> m_shape2EdgeOffsets;
        DArray<int> m_shape2VertexOffsets;
        DArray<int> m_patch2Shape;
        DArray<int> m_patch2TriOffsets;
        DArray<int> m_patch2TriIndices;

        DArray<int> m_sourcePatchIds;
        DArray<int> m_sourceTargetShapeIds;
        DArray<int> m_middleHitCounts;
        DArray<int> m_middleHitOffsets;
        DArrayList<int> m_middleHitLists;
        DArray<PatchPair> m_patchPairs;

        DArray<AABB> m_triAabbsWorld;
        DArray<Coord> m_faceNormalsWorld;
        DArray<Coord> m_edgeNormalsWorld;

        DArray<int> m_patchPairTriPairCounts;
        DArray<int> m_patchPairTriPairOffsets;
        DArray<int> m_candidateTri0;
        DArray<int> m_candidateTri1;
        DArray<int> m_candidatePatchPairId;

        DArray<int> m_coarsePassCounts;
        DArray<int> m_coarsePassOffsets;
        DArray<int> m_filteredTri0;
        DArray<int> m_filteredTri1;
        DArray<int> m_filteredPatchPairId;

        DArray<int> m_primitivePassCounts;
        DArray<int> m_primitivePassOffsets;
        DArray<ContactPair> m_primitiveCandidateContacts;
        DArray<unsigned long long> m_primitiveCandidateKeys;
        DArray<int> m_primitiveCandidateSortedIndices;
        DArray<int> m_primitiveCandidateKeepFlags;
        DArray<int> m_selectedPrimitiveCounts;

        DArray<int> m_triPairContactCounts;
        DArray<int> m_triPairContactOffsets;
        DArray<ContactPair> m_meshContacts;

        Scan<int> m_scan;
        Reduction<int> m_reduce;
    };

}
