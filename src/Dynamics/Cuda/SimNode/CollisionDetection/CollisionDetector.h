#pragma once

#include <memory>
#include <vector>

#include <Array/Array.h>
#include <Array/Array2D.h>

#include "MeshCollisionTypes.h"

namespace dyno {

    template<typename TDataType>
    struct RigidBody;

    struct BatchCollisionConstraints;

    template<typename TDataType>
    class NeighborMeshLevelQuery;

    template<typename TDataType>
    class TriangleSet;

    template<typename TDataType>
    class MeshCollisionDetector
    {
    public:
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using Matrix = typename TDataType::Matrix;
        using AABB = TAlignedBox3D<Real>;
        using Triangle = TopologyModule::Triangle;
        using ContactPair = TContactPair<Real>;

        MeshCollisionDetector() {};
        ~MeshCollisionDetector() {};

        void Initialize(int num_envs, int max_bodies, const RigidBody<TDataType>& rb);
        void Detect(const RigidBody<TDataType>& rb, BatchCollisionConstraints& out, int num_envs);

    private:
        void resetQueryStaticMappingIfNeeded(int shapeCount,
            const std::vector<int>& shape2PatchOffsets,
            const std::vector<uint>& patch2Shape);

    private:
        bool m_initialized = false;
        int m_numEnvs = 0;
        int m_maxBodies = 0;
        int m_cachedMeshShapeCount = -1;
        Real m_dHat = Real(1e-3);

        MeshTemplateData<TDataType> m_cubeTemplate;
        std::vector<Coord> m_cubeVerticesHost;
        std::vector<Triangle> m_cubeTrianglesHost;

        std::shared_ptr<NeighborMeshLevelQuery<TDataType>> m_meshNarrowQuery;
        std::shared_ptr<TriangleSet<TDataType>> m_meshTriangleSet;
        std::vector<ContactPair> m_meshContactsHost;
    };

}
