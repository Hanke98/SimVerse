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
#include "Utils/SimBlockVector.h"

#include "MeshCollisionTypes.h"

namespace dyno {

    template<typename TDataType>
    struct RigidBody;

    struct BatchCollisionConstraints;

    template<typename TDataType>
    class CollisionDetectionBroadPhase;

    struct BodyContactId
	{
        int env_id = INVLIDA_ID;
		int body_id_1 = INVLIDA_ID; // env-local rigid body id
		int body_id_2 = INVLIDA_ID; // env-local rigid body id
	};

    template<typename Real>
    struct MeshContact
    {
        Vector<Real, 3> pos;      // world-space contact point
        Vector<Real, 3> normal;   // points from body_id_1 toward body_id_0
        Real depth = Real(0);     // penetration depth (always >= 0)
        int env_id = INVLIDA_ID;
        int body_id_0 = INVLIDA_ID;
        int body_id_1 = INVLIDA_ID;
        ContactType contact_type = CT_UNKNOWN;
    };

    template<typename TDataType>
    class MeshCollisionDetector
    {
    public:
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using Matrix = typename TDataType::Matrix;
        using AABB = TAlignedBox3D<Real>;
        using Triangle = TopologyModule::Triangle;
        using ContactPair = MeshContact<Real>;

        MeshCollisionDetector() {};
        ~MeshCollisionDetector() {};

        void Initialize(int num_envs, int max_bodies, const RigidBody<TDataType>& rb);
        void Detect(const RigidBody<TDataType>& rb, BatchCollisionConstraints& out, int num_envs);
        void DetectGround(const RigidBody<TDataType>& rb, BatchCollisionConstraints& out, int num_envs);
        int RegisterMeshTemplate(const std::vector<Coord>& vertices, const std::vector<Triangle>& triangles);
        void SetBodyMeshTemplate(int envId, int bodyId, int templateId);

    private:
        bool broad_phase(const RigidBody<TDataType>& rb, int num_envs);
        bool middle_phase(const RigidBody<TDataType>& rb, int num_envs, std::vector<BodyPair>& bodyPairsHost);
        void narrow_phase(const RigidBody<TDataType>& rb,
            BatchCollisionConstraints& out,
            int num_envs);
        void refreshMeshShapeLayoutCache(const RigidBody<TDataType>& rb, int num_envs);
        void runMeshMeshNarrowPhase(const RigidBody<TDataType>& rb,
            const DArray<BodyContactId>& bodyPairs,
            BatchCollisionConstraints& out,
            int num_envs);
        void detectMeshMeshInternal(const RigidBody<TDataType>& rb,
            BatchCollisionConstraints& out,
            int num_envs);
        void rebuildMeshTemplateViews();
        void initializeDefaultBodyTemplateMapping(const RigidBody<TDataType>& rb, int num_envs);
        int hostBodyTemplateId(int envId, int bodyId) const;
        int hostBodyTriOffset(int envId, int bodyId) const;
        int hostBodyEdgeOffset(int envId, int bodyId) const;

    private:
        bool m_initialized = false;
        int m_numEnvs = 0;
        int m_maxBodies = 0;
        int m_meshTemplateVersion = 0;
        int m_cachedMeshLayoutVersion = -1;
        int m_cachedMeshLayoutEnvCount = -1;
        std::vector<int> m_cachedMeshBodyCounts;
        Real m_dHat = Real(1e-3);
        Real m_edgeEdgeActivationMargin = Real(3e-3);

        std::vector<MeshTemplateData<TDataType>> m_meshTemplates;
        std::vector<std::shared_ptr<TriangleSet<TDataType>>> m_meshTemplateBuilders;
        DArray<MeshTemplateKernelView<TDataType>> m_meshTemplateViews;
        DevArr2D<int> m_bodyToMeshTemplate;
        std::vector<int> m_bodyToMeshTemplateHost;
        std::vector<int> m_body2TriOffsetsHost;
        std::vector<int> m_body2EdgeOffsetsHost;

        DArray<AABB> m_bodyAABBs;
        DArray<BodyContactId> m_bodyContactPairs;
        DArray<BodyPair> m_bodyPairs;
        std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> m_bodyBroadPhase;

        DevArr2D<int> m_body2PatchOffsets;
        DevArr2D<int> m_body2TriOffsets;
        DevArr2D<int> m_body2EdgeOffsets;
        DevArr2D<int> m_body2VertexOffsets;
        DArray<MeshBodyId> m_patch2Body;
        DArray<MeshBodyId> m_tri2Body;
        DArray<MeshBodyId> m_edge2Body;
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
        DArray<int> m_worklistTriIds;
        DArray<int> m_worklistEdgeIds;

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
