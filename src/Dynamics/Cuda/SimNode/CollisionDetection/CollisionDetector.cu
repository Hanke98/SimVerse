#include "CollisionDetector.h"
#include "CubeToMesh.h"

#include "PhysicalField/RigidBody/RigidBody.h"
#include "Utils/Constraints.h"

#include "Array/ArrayList.h"
#include "Collision/CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "Collision/NeighborMeshLevelQuery.h"
#include "Topology/DiscreteElements.h"
#include "Topology/TriangleSet.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#if defined(__clang__)
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif

namespace dyno {

namespace {

static constexpr int CD_MAX_CONTACTS_PER_ENV = 1024;


template<typename Real>
__device__ inline void CD_WriteContact(
    BatchCollisionConstraints& out,
    int env_id,
    int body_a,
    int body_b,
    Real depth,
    const Vector<Real, 3>& normal,
    const Vector<Real, 3>& point,
    Real mu)
{
    if (depth <= Real(0))
        return;

    int idx = atomicAdd(&out.collision_nums[env_id], 1);
    if (idx < CD_MAX_CONTACTS_PER_ENV)
    {
        out.body_idxs(env_id, idx) = Pair<int, int>(body_a, body_b);
        out.depth(env_id, idx) = depth;
        out.normal(env_id, idx) = normal;
        out.point(env_id, idx) = point;
        out.mu(env_id, idx) = mu;
    }
    else
    {
        atomicSub(&out.collision_nums[env_id], 1);
    }
}

template<typename TDataType>
__global__ void CD_ComputeBodyAABBsKernel(
    DArray<TAlignedBox3D<typename TDataType::Real>> bodyAabbs,
    DArray<int> batch_bodies,
    DArray2D<Vector<typename TDataType::Real, 3>> batch_pos,
    DArray2D<SquareMatrix<typename TDataType::Real, 3>> batch_rot,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    typename TDataType::Real dHat,
    int maxBodies)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;
    using Matrix = SquareMatrix<Real, 3>;
    using AABB = TAlignedBox3D<Real>;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bodyAabbs.size())
        return;

    int env_id = tid / maxBodies;
    int body_id = tid - env_id * maxBodies;

    AABB box;
    box.v0 = Coord(Real(1e30));
    box.v1 = Coord(Real(-1e30));

    if (env_id < batch_bodies.size() && body_id < batch_bodies[env_id])
    {
        int st = shape_type(env_id, body_id);
        int sidx = shape_idx(env_id, body_id);

        const Coord pos = batch_pos(env_id, body_id);
        const Matrix rot = batch_rot(env_id, body_id);

        if (st == 0)
        {
            const SphereInfo sph = spheres(env_id, sidx);
            Coord c = pos + rot * sph.center;
            Real r = sph.radius + dHat;
            box.v0 = c - Coord(r);
            box.v1 = c + Coord(r);
        }
        else if (st == 1)
        {
            const BoxInfo b = boxes(env_id, sidx);
            Coord center = pos + rot * b.center;

            Real ex = std::abs(rot(0, 0)) * b.halfLength[0]
                + std::abs(rot(0, 1)) * b.halfLength[1]
                + std::abs(rot(0, 2)) * b.halfLength[2];
            Real ey = std::abs(rot(1, 0)) * b.halfLength[0]
                + std::abs(rot(1, 1)) * b.halfLength[1]
                + std::abs(rot(1, 2)) * b.halfLength[2];
            Real ez = std::abs(rot(2, 0)) * b.halfLength[0]
                + std::abs(rot(2, 1)) * b.halfLength[1]
                + std::abs(rot(2, 2)) * b.halfLength[2];

            Coord ext(ex + dHat, ey + dHat, ez + dHat);
            box.v0 = center - ext;
            box.v1 = center + ext;
        }
        else if (st == 2)
        {
            const CapsuleInfo cap = capsules(env_id, sidx);
            Coord center = pos + rot * cap.center;
            Quat<Real> worldRot = Quat<Real>(rot) * cap.rot;
            Coord axis = worldRot.rotate(Coord(Real(0), cap.halfLength, Real(0)));
            Coord p0 = center - axis;
            Coord p1 = center + axis;
            Real r = cap.radius + dHat;
            box.v0 = p0.minimum(p1) - Coord(r);
            box.v1 = p0.maximum(p1) + Coord(r);
        }
    }

    bodyAabbs[tid] = box;
}

template<typename TDataType>
__device__ inline void CD_NarrowPrimPrim(
    int env_id,
    int bodyA,
    int bodyB,
    int typeA,
    int typeB,
    int idxA,
    int idxB,
    const Vector<typename TDataType::Real, 3>& posA,
    const SquareMatrix<typename TDataType::Real, 3>& rotA,
    const Vector<typename TDataType::Real, 3>& posB,
    const SquareMatrix<typename TDataType::Real, 3>& rotB,
    DArray2D<SphereInfo>& spheres,
    DArray2D<CapsuleInfo>& capsules,
    typename TDataType::Real mu,
    typename TDataType::Real dHat,
    BatchCollisionConstraints& out)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;
    using Sphere3D = TSphere3D<Real>;
    using Capsule3D = TCapsule3D<Real>;
    using Segment3D = TSegment3D<Real>;

    TManifold<Real> manifold;

    if (typeA == 0 && typeB == 0)
    {
        Sphere3D sA(posA + rotA * spheres(env_id, idxA).center, spheres(env_id, idxA).radius);
        Sphere3D sB(posB + rotB * spheres(env_id, idxB).center, spheres(env_id, idxB).radius);
        CollisionDetection<Real>::request(manifold, sA, sB, dHat, dHat);
    }
    else if (typeA == 0 && typeB == 2)
    {
        Sphere3D sA(posA + rotA * spheres(env_id, idxA).center, spheres(env_id, idxA).radius);
        CapsuleInfo cInfo = capsules(env_id, idxB);
        Capsule3D cB(posB + rotB * cInfo.center, Quat<Real>(rotB) * cInfo.rot, cInfo.radius, cInfo.halfLength);
        Segment3D segB = cB.centerline();
        CollisionDetection<Real>::request(manifold, sA, segB, dHat, cB.radius + dHat);
    }
    else if (typeA == 2 && typeB == 0)
    {
        CapsuleInfo cInfo = capsules(env_id, idxA);
        Capsule3D cA(posA + rotA * cInfo.center, Quat<Real>(rotA) * cInfo.rot, cInfo.radius, cInfo.halfLength);
        Segment3D segA = cA.centerline();
        Sphere3D sB(posB + rotB * spheres(env_id, idxB).center, spheres(env_id, idxB).radius);
        CollisionDetection<Real>::request(manifold, segA, sB, cA.radius + dHat, dHat);
    }
    else if (typeA == 2 && typeB == 2)
    {
        CapsuleInfo cInfoA = capsules(env_id, idxA);
        CapsuleInfo cInfoB = capsules(env_id, idxB);
        Capsule3D cA(posA + rotA * cInfoA.center, Quat<Real>(rotA) * cInfoA.rot, cInfoA.radius, cInfoA.halfLength);
        Capsule3D cB(posB + rotB * cInfoB.center, Quat<Real>(rotB) * cInfoB.rot, cInfoB.radius, cInfoB.halfLength);
        Segment3D segA = cA.centerline();
        Segment3D segB = cB.centerline();
        CollisionDetection<Real>::request(manifold, segA, segB, cA.radius + dHat, cB.radius + dHat);
    }

    for (int c = 0; c < manifold.contactCount; ++c)
    {
        CD_WriteContact<Real>(
            out,
            env_id,
            bodyA,
            bodyB,
            -manifold.contacts[c].penetration,
            manifold.normal,
            manifold.contacts[c].position,
            mu);
    }
}

template<typename TDataType>
__device__ inline void CD_NarrowPrimMesh(
    int env_id,
    int primBody,
    int meshBody,
    int primType,
    int primIdx,
    int meshIdx,
    const Vector<typename TDataType::Real, 3>& primPos,
    const SquareMatrix<typename TDataType::Real, 3>& primRot,
    const Vector<typename TDataType::Real, 3>& meshPos,
    const SquareMatrix<typename TDataType::Real, 3>& meshRot,
    DArray2D<BoxInfo>& boxes,
    DArray2D<SphereInfo>& spheres,
    DArray2D<CapsuleInfo>& capsules,
    DArray<Vector<typename TDataType::Real, 3>>& cubeVertices,
    DArray<TopologyModule::Triangle>& cubeTriangles,
    typename TDataType::Real mu,
    typename TDataType::Real dHat,
    BatchCollisionConstraints& out)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;
    using Matrix = SquareMatrix<Real, 3>;
    using Sphere3D = TSphere3D<Real>;
    using Capsule3D = TCapsule3D<Real>;
    using Segment3D = TSegment3D<Real>;
    using Triangle3D = TTriangle3D<Real>;

    BoxInfo meshBox = boxes(env_id, meshIdx);
    Matrix shapeRot = meshRot * meshBox.rot.toMatrix3x3();
    Coord shapeCenter = meshPos + meshRot * meshBox.center;

    Sphere3D sphere;
    Capsule3D capsule;
    Segment3D capsuleSeg;
    if (primType == 0)
    {
        SphereInfo sphInfo = spheres(env_id, primIdx);
        sphere = Sphere3D(primPos + primRot * sphInfo.center, sphInfo.radius);
    }
    else if (primType == 2)
    {
        CapsuleInfo capInfo = capsules(env_id, primIdx);
        capsule = Capsule3D(primPos + primRot * capInfo.center, Quat<Real>(primRot) * capInfo.rot, capInfo.radius, capInfo.halfLength);
        capsuleSeg = capsule.centerline();
    }

    for (int t = 0; t < cubeTriangles.size(); ++t)
    {
        TopologyModule::Triangle triIdx = cubeTriangles[t];
        Coord p0 = cubeVertices[triIdx[0]];
        Coord p1 = cubeVertices[triIdx[1]];
        Coord p2 = cubeVertices[triIdx[2]];

        p0 = shapeCenter + shapeRot * Coord(p0[0] * meshBox.halfLength[0], p0[1] * meshBox.halfLength[1], p0[2] * meshBox.halfLength[2]);
        p1 = shapeCenter + shapeRot * Coord(p1[0] * meshBox.halfLength[0], p1[1] * meshBox.halfLength[1], p1[2] * meshBox.halfLength[2]);
        p2 = shapeCenter + shapeRot * Coord(p2[0] * meshBox.halfLength[0], p2[1] * meshBox.halfLength[1], p2[2] * meshBox.halfLength[2]);

        Triangle3D tri(p0, p1, p2);
        TManifold<Real> manifold;

        if (primType == 0)
        {
            CollisionDetection<Real>::request(manifold, sphere, tri, dHat, dHat);
        }
        else if (primType == 2)
        {
            CollisionDetection<Real>::request(manifold, capsuleSeg, tri, capsule.radius + dHat, dHat);
        }

        for (int c = 0; c < manifold.contactCount; ++c)
        {
            CD_WriteContact<Real>(
                out,
                env_id,
                primBody,
                meshBody,
                -manifold.contacts[c].penetration,
                manifold.normal,
                manifold.contacts[c].position,
                mu);
        }
    }
}

template<typename TDataType>
__global__ void CD_NarrowPrimitivePairsKernel(
    BatchCollisionConstraints out,
    DArray<BodyPair> bodyPairs,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<Vector<typename TDataType::Real, 3>> batch_pos,
    DArray2D<SquareMatrix<typename TDataType::Real, 3>> batch_rot,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    DevArr2D<int> is_static,
    DevArr2D<int> parent_idx,
    DevArr2D<typename TDataType::Real> friction_mu,
    DArray<Vector<typename TDataType::Real, 3>> cubeVertices,
    DArray<TopologyModule::Triangle> cubeTriangles,
    typename TDataType::Real dHat)
{
    using Real = typename TDataType::Real;

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= bodyPairs.size())
        return;

    BodyPair pair = bodyPairs[pid];
    int env_id = pair.env_id;
    int bodyA = pair.body_a;
    int bodyB = pair.body_b;

    int typeA = shape_type(env_id, bodyA);
    int typeB = shape_type(env_id, bodyB);
    int idxA = shape_idx(env_id, bodyA);
    int idxB = shape_idx(env_id, bodyB);

    bool meshA = (typeA == 1);
    bool meshB = (typeB == 1);

    if (is_static(env_id, bodyA) && is_static(env_id, bodyB))
        return;
    if (parent_idx(env_id, bodyA) == bodyB || parent_idx(env_id, bodyB) == bodyA)
        return;

    if (meshA && meshB)
        return;

    Real mu = sqrtf(friction_mu(env_id, bodyA) * friction_mu(env_id, bodyB));

    auto posA = batch_pos(env_id, bodyA);
    auto rotA = batch_rot(env_id, bodyA);
    auto posB = batch_pos(env_id, bodyB);
    auto rotB = batch_rot(env_id, bodyB);

    if (!meshA && !meshB)
    {
        CD_NarrowPrimPrim<TDataType>(
            env_id,
            bodyA,
            bodyB,
            typeA,
            typeB,
            idxA,
            idxB,
            posA,
            rotA,
            posB,
            rotB,
            spheres,
            capsules,
            mu,
            dHat,
            out);
        return;
    }

    if (meshA)
    {
        CD_NarrowPrimMesh<TDataType>(
            env_id,
            bodyB,
            bodyA,
            typeB,
            idxB,
            idxA,
            posB,
            rotB,
            posA,
            rotA,
            boxes,
            spheres,
            capsules,
            cubeVertices,
            cubeTriangles,
            mu,
            dHat,
            out);
    }
    else
    {
        CD_NarrowPrimMesh<TDataType>(
            env_id,
            bodyA,
            bodyB,
            typeA,
            idxA,
            idxB,
            posA,
            rotA,
            posB,
            rotB,
            boxes,
            spheres,
            capsules,
            cubeVertices,
            cubeTriangles,
            mu,
            dHat,
            out);
    }
}

template<typename TDataType>
__global__ void CD_AppendMeshContactsKernel(
    BatchCollisionConstraints out,
    DArray<TContactPair<typename TDataType::Real>> contacts,
    DArray<int> batch_bodies,
    DevArr2D<typename TDataType::Real> friction_mu,
    int maxBodies,
    int num_envs)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;

    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= contacts.size())
        return;

    const auto cp = contacts[cid];
    if (cp.bodyId1 < 0 || cp.bodyId2 < 0)
        return;

    int envA = cp.bodyId1 / maxBodies;
    int envB = cp.bodyId2 / maxBodies;
    if (envA != envB || envA < 0 || envA >= num_envs)
        return;

    int localA = cp.bodyId1 - envA * maxBodies;
    int localB = cp.bodyId2 - envA * maxBodies;
    if (localA < 0 || localA >= maxBodies || localB < 0 || localB >= maxBodies)
        return;
    if (localA >= batch_bodies[envA] || localB >= batch_bodies[envA])
        return;

    Coord normal = cp.normal2;
    if (normal.normSquared() < Real(1e-12))
        normal = -cp.normal1;
    Coord point = Real(0.5) * (cp.pos1 + cp.pos2);

    Real mu = sqrtf(friction_mu(envA, localA) * friction_mu(envA, localB));

    CD_WriteContact<Real>(
        out,
        envA,
        localA,
        localB,
        cp.interpenetration,
        normal,
        point,
        mu);
}

template<typename TDataType>
__global__ void CD_GroundCollisionKernel(
    BatchCollisionConstraints out,
    DArray<int> batch_bodies,
    DevArr2D<int> is_static,
    DArray2D<Vector<typename TDataType::Real, 3>> batch_pos,
    DArray2D<SquareMatrix<typename TDataType::Real, 3>> batch_rot,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    int num_envs)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;
    using Matrix = SquareMatrix<Real, 3>;

    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs)
        return;

    const Coord groundNormal(Real(0), Real(1), Real(0));
    int bodyCount = batch_bodies[env_id];

    for (int bid = 0; bid < bodyCount; ++bid)
    {
        if (is_static(env_id, bid))
            continue;

        int st = shape_type(env_id, bid);
        int sidx = shape_idx(env_id, bid);

        Coord pos = batch_pos(env_id, bid);
        Matrix rot = batch_rot(env_id, bid);

        if (st == 1)
        {
            BoxInfo box = boxes(env_id, sidx);
            Matrix shapeRot = rot * box.rot.toMatrix3x3();
            Coord shapeCenter = pos + rot * box.center;

            for (int i = 0; i < 8; ++i)
            {
                int sx = (i & 1) ? 1 : -1;
                int sy = (i & 2) ? 1 : -1;
                int sz = (i & 4) ? 1 : -1;
                Coord localV(
                    Real(sx) * box.halfLength[0],
                    Real(sy) * box.halfLength[1],
                    Real(sz) * box.halfLength[2]);
                Coord wv = shapeCenter + shapeRot * localV;

                if (wv[1] < Real(0))
                {
                    CD_WriteContact<Real>(
                        out,
                        env_id,
                        bid,
                        -1,
                        -wv[1],
                        groundNormal,
                        Coord(wv[0], Real(0.5) * wv[1], wv[2]),
                        Real(0.6));
                }
            }
        }
        else if (st == 0)
        {
            SphereInfo sph = spheres(env_id, sidx);
            Coord center = pos + rot * sph.center;
            Real depth = sph.radius - center[1];
            if (depth > Real(0))
            {
                CD_WriteContact<Real>(
                    out,
                    env_id,
                    bid,
                    -1,
                    depth,
                    groundNormal,
                    Coord(center[0], center[1] - Real(0.5) * depth, center[2]),
                    Real(0.6));
            }
        }
        else if (st == 2)
        {
            CapsuleInfo cap = capsules(env_id, sidx);
            Quat<Real> worldRot = Quat<Real>(rot) * cap.rot;
            Coord center = pos + rot * cap.center;
            Coord axis = worldRot.rotate(Coord(Real(0), cap.halfLength, Real(0)));
            Coord p0 = center - axis;
            Coord p1 = center + axis;

            for (int k = 0; k < 2; ++k)
            {
                Coord c = (k == 0) ? p0 : p1;
                Real depth = cap.radius - c[1];
                if (depth > Real(0))
                {
                    CD_WriteContact<Real>(
                        out,
                        env_id,
                        bid,
                        -1,
                        depth,
                        groundNormal,
                        Coord(c[0], c[1] - Real(0.5) * depth, c[2]),
                        Real(0.6));
                }
            }
        }
    }
}

struct CDMeshBodyInfo
{
    int env_id = -1;
    int local_body = -1;
    int global_body = -1;
    int shape_idx = -1;
};

} // namespace

template<typename TDataType>
void MeshCollisionDetector<TDataType>::Initialize(int num_envs, int max_bodies, const RigidBody<TDataType>&)
{
    m_numEnvs = num_envs;
    m_maxBodies = max_bodies;
    m_cachedMeshShapeCount = -1;

    GenerateUnitCubeMesh(m_cubeTemplate);

    {
        CArray<Coord> hv;
        hv.assign(m_cubeTemplate.vertices);
        m_cubeVerticesHost.assign(hv.begin(), hv.begin() + hv.size());

        CArray<Triangle> ht;
        ht.assign(m_cubeTemplate.triangles);
        m_cubeTrianglesHost.assign(ht.begin(), ht.begin() + ht.size());
    }

    if (m_cubeVerticesHost.empty() || m_cubeTrianglesHost.empty())
    {
        m_cubeVerticesHost.resize(8);
        for (int i = 0; i < 8; ++i)
        {
            Real x = (i & 1) ? Real(1) : Real(-1);
            Real y = (i & 2) ? Real(1) : Real(-1);
            Real z = (i & 4) ? Real(1) : Real(-1);
            m_cubeVerticesHost[i] = Coord(x, y, z);
        }

        m_cubeTrianglesHost = {
            Triangle(0, 3, 1), Triangle(0, 2, 3),
            Triangle(4, 5, 7), Triangle(4, 7, 6),
            Triangle(0, 1, 5), Triangle(0, 5, 4),
            Triangle(2, 7, 3), Triangle(2, 6, 7),
            Triangle(0, 4, 6), Triangle(0, 6, 2),
            Triangle(1, 3, 7), Triangle(1, 7, 5)
        };
    }

    m_bodyAABBs.resize(num_envs * max_bodies);

    m_bodyBroadPhase = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
    m_bodyBroadPhase->varSelfCollision()->setValue(true);
    m_bodyBroadPhase->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
    m_bodyBroadPhase->varGridSizeLimit()->setValue(Real(0.01));

    m_meshNarrowQuery = std::make_shared<NeighborMeshLevelQuery<TDataType>>();
    m_meshNarrowQuery->varEnableAdjacentFilter()->setValue(false);
    m_meshNarrowQuery->inEnableVisualizeCollisionTriSet()->setValue(false);
    m_meshNarrowQuery->varInputVerticesInRestWorld()->setValue(false);
    m_meshNarrowQuery->varGridSizeLimit()->setValue(Real(0.01));
    m_meshNarrowQuery->varDHead()->setValue(m_dHat);

    m_meshDiscreteElements = std::make_shared<DiscreteElements<TDataType>>();
    m_meshTriangleSet = std::make_shared<TriangleSet<TDataType>>();

    m_initialized = true;
    spdlog::info("[MeshCollisionDetector] Initialized (envs={}, maxBodies={})", num_envs, max_bodies);
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::resetQueryStaticMappingIfNeeded(
    int shapeCount,
    const std::vector<int>& shape2PatchOffsets,
    const std::vector<uint>& patch2Shape)
{
    if (!m_meshNarrowQuery)
        return;

    if (m_cachedMeshShapeCount == shapeCount)
        return;

    bool ok = m_meshNarrowQuery->setStaticShape2PatchOffsets(shape2PatchOffsets);
    ok = ok && m_meshNarrowQuery->setStaticPatch2Shape(patch2Shape);
    std::vector<std::shared_ptr<LinearBVH<TDataType>>> dummyBvhs(shapeCount, nullptr);
    ok = ok && m_meshNarrowQuery->setStaticTargetBVHCache(dummyBvhs);

    if (!ok)
    {
        spdlog::warn("[MeshCollisionDetector] Failed to set static mapping for mesh query (shapeCount={})", shapeCount);
    }
    else
    {
        m_cachedMeshShapeCount = shapeCount;
    }
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::detectMeshMeshByNeighborQuery(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    using Box3D = TOrientedBox3D<Real>;

    CArray<int> hBatchBodies;
    hBatchBodies.assign(rb.batch_bodies);

    CArray2D<int> hShapeType;
    CArray2D<int> hShapeIdx;
    CArray2D<Coord> hPos;
    CArray2D<Matrix> hRot;
    CArray2D<BoxInfo> hBoxes;

    hShapeType.assign(rb.shape_type);
    hShapeIdx.assign(rb.shape_idx);
    hPos.assign(rb.batch_pos);
    hRot.assign(rb.batch_rot);
    hBoxes.assign(rb.boxes);

    std::vector<CDMeshBodyInfo> meshBodies;
    meshBodies.reserve(num_envs * 8);

    for (int env = 0; env < num_envs; ++env)
    {
        int bodyCount = hBatchBodies[env];
        for (int b = 0; b < bodyCount; ++b)
        {
            if (hShapeType(env, b) != 1)
                continue;

            CDMeshBodyInfo info;
            info.env_id = env;
            info.local_body = b;
            info.global_body = env * m_maxBodies + b;
            info.shape_idx = hShapeIdx(env, b);
            meshBodies.push_back(info);
        }
    }

    const int shapeCount = static_cast<int>(meshBodies.size());
    if (shapeCount < 2)
        return;

    std::vector<Coord> triPoints;
    std::vector<Triangle> triIndices;
    std::vector<AABB> patchAabbs;
    std::vector<int> patch2TriOffsets;
    std::vector<int> patch2TriIndices;
    std::vector<int> shape2PatchOffsets(shapeCount + 1, 0);
    std::vector<int> shape2TriOffsets(shapeCount + 1, 0);
    std::vector<uint> patch2Shape;
    std::vector<int> shape2RigidBody(shapeCount, 0);
    std::vector<int> shape2ElementIds(shapeCount, 0);
    std::vector<Coord> restShapeCenters(shapeCount, Coord(0));
    std::vector<Matrix> restShapeRotations(shapeCount, Matrix::identityMatrix());

    CArray<int> tplPatchOffsets;
    CArray<int> tplPatchFaces;
    tplPatchOffsets.assign(m_cubeTemplate.patchOffsets);
    tplPatchFaces.assign(m_cubeTemplate.patchFaces);

    const int templatePatchCount = m_cubeTemplate.numPatches > 0 ? m_cubeTemplate.numPatches : 1;
    const int templateTriCount = static_cast<int>(m_cubeTrianglesHost.size());

    triPoints.reserve(shapeCount * m_cubeVerticesHost.size());
    triIndices.reserve(shapeCount * templateTriCount);
    patchAabbs.reserve(shapeCount * templatePatchCount);
    patch2Shape.reserve(shapeCount * templatePatchCount);
    patch2TriOffsets.reserve(shapeCount * templatePatchCount + 1);
    patch2TriOffsets.push_back(0);

    for (int s = 0; s < shapeCount; ++s)
    {
        const auto& m = meshBodies[s];
        const BoxInfo box = hBoxes(m.env_id, m.shape_idx);
        const Coord bodyPos = hPos(m.env_id, m.local_body);
        const Matrix bodyRot = hRot(m.env_id, m.local_body);

        shape2RigidBody[s] = m.global_body;
        shape2ElementIds[s] = s;
        restShapeCenters[s] = bodyPos;
        restShapeRotations[s] = bodyRot;

        const Matrix shapeRot = bodyRot * box.rot.toMatrix3x3();
        const Coord shapeCenter = bodyPos + bodyRot * box.center;

        const int vBase = static_cast<int>(triPoints.size());
        for (const Coord& v : m_cubeVerticesHost)
        {
            Coord scaled(v[0] * box.halfLength[0], v[1] * box.halfLength[1], v[2] * box.halfLength[2]);
            triPoints.push_back(shapeCenter + shapeRot * scaled);
        }

        const int triBase = static_cast<int>(triIndices.size());
        for (const Triangle& tri : m_cubeTrianglesHost)
        {
            triIndices.emplace_back(vBase + tri[0], vBase + tri[1], vBase + tri[2]);
        }

        shape2PatchOffsets[s + 1] = shape2PatchOffsets[s] + templatePatchCount;
        shape2TriOffsets[s + 1] = static_cast<int>(triIndices.size());

        for (int p = 0; p < templatePatchCount; ++p)
        {
            patch2Shape.push_back(static_cast<uint>(s));

            int addBegin = static_cast<int>(patch2TriIndices.size());
            if (tplPatchOffsets.size() >= static_cast<uint>(templatePatchCount + 1)
                && tplPatchFaces.size() > 0)
            {
                int pBegin = tplPatchOffsets[p];
                int pEnd = tplPatchOffsets[p + 1];
                if (pBegin < 0) pBegin = 0;
                if (pEnd > static_cast<int>(tplPatchFaces.size())) pEnd = static_cast<int>(tplPatchFaces.size());

                for (int fi = pBegin; fi < pEnd; ++fi)
                {
                    int localTri = tplPatchFaces[fi];
                    if (localTri >= 0 && localTri < templateTriCount)
                        patch2TriIndices.push_back(triBase + localTri);
                }
            }
            else
            {
                for (int localTri = 0; localTri < templateTriCount; ++localTri)
                    patch2TriIndices.push_back(triBase + localTri);
            }

            patch2TriOffsets.push_back(static_cast<int>(patch2TriIndices.size()));

            AABB paabb;
            paabb.v0 = Coord(Real(1e30));
            paabb.v1 = Coord(Real(-1e30));
            for (int i = addBegin; i < static_cast<int>(patch2TriIndices.size()); ++i)
            {
                const Triangle& tri = triIndices[patch2TriIndices[i]];
                paabb.v0 = paabb.v0.minimum(triPoints[tri[0]]).minimum(triPoints[tri[1]]).minimum(triPoints[tri[2]]);
                paabb.v1 = paabb.v1.maximum(triPoints[tri[0]]).maximum(triPoints[tri[1]]).maximum(triPoints[tri[2]]);
            }
            patchAabbs.push_back(paabb);
        }
    }

    const int totalBodies = num_envs * m_maxBodies;
    std::vector<Coord> centers(totalBodies, Coord(0));
    std::vector<Matrix> rotations(totalBodies, Matrix::identityMatrix());
    for (int env = 0; env < num_envs; ++env)
    {
        for (int b = 0; b < m_maxBodies; ++b)
        {
            centers[env * m_maxBodies + b] = hPos(env, b);
            rotations[env * m_maxBodies + b] = hRot(env, b);
        }
    }

    m_meshTriangleSet->setPoints(triPoints);
    m_meshTriangleSet->setTriangles(triIndices);
    m_meshTriangleSet->update();

    {
        CArray<Box3D> boxHost(shapeCount);
        for (int s = 0; s < shapeCount; ++s)
        {
            const auto& m = meshBodies[s];
            BoxInfo box = hBoxes(m.env_id, m.shape_idx);
            boxHost[s] = Box3D(box.center, box.rot, box.halfLength);
        }

        DArray<Box3D> dBoxes;
        dBoxes.assign(boxHost);
        m_meshDiscreteElements->setBoxes(dBoxes);

        DArray<TSphere3D<Real>> dSpheres;
        DArray<TCapsule3D<Real>> dCapsules;
        DArray<TTet3D<Real>> dTets;
        DArray<TTriangle3D<Real>> dTris;
        m_meshDiscreteElements->setSpheres(dSpheres);
        m_meshDiscreteElements->setCapsules(dCapsules);
        m_meshDiscreteElements->setTets(dTets);
        m_meshDiscreteElements->setTriangles(dTris);

        DArray<Coord> dPos;
        DArray<Matrix> dRot;
        dPos.assign(centers);
        dRot.assign(rotations);
        m_meshDiscreteElements->setPosition(dPos);
        m_meshDiscreteElements->setRotation(dRot);

        CArray<Pair<uint, uint>> mapping(shapeCount);
        for (int s = 0; s < shapeCount; ++s)
            mapping[s] = Pair<uint, uint>(static_cast<uint>(s), static_cast<uint>(shape2RigidBody[s]));
        m_meshDiscreteElements->shape2RigidBodyMapping().assign(mapping);

        m_meshDiscreteElements->update();
    }

    resetQueryStaticMappingIfNeeded(shapeCount, shape2PatchOffsets, patch2Shape);
    m_meshNarrowQuery->inDiscreteElements()->setDataPtr(m_meshDiscreteElements);
    m_meshNarrowQuery->inTriangleSet()->setDataPtr(m_meshTriangleSet);
    m_meshNarrowQuery->inPatchAABBs()->assign(patchAabbs);
    m_meshNarrowQuery->inShape2PatchOffsets()->assign(shape2PatchOffsets);
    m_meshNarrowQuery->inShape2TriOffsets()->assign(shape2TriOffsets);
    m_meshNarrowQuery->inPatch2TriOffsets()->assign(patch2TriOffsets);
    m_meshNarrowQuery->inPatch2TriIndices()->assign(patch2TriIndices);
    m_meshNarrowQuery->inCenter()->assign(centers);
    m_meshNarrowQuery->inRotationMatrix()->assign(rotations);
    m_meshNarrowQuery->inRestShapeCenter()->assign(restShapeCenters);
    m_meshNarrowQuery->inRestShapeRotation()->assign(restShapeRotations);
    m_meshNarrowQuery->inShape2RigidBodyIds()->assign(shape2RigidBody);
    m_meshNarrowQuery->inShape2ElementIdsDense()->assign(shape2ElementIds);
    m_meshNarrowQuery->varDHead()->setValue(m_dHat);
    m_meshNarrowQuery->varInputVerticesInRestWorld()->setValue(false);

    m_meshNarrowQuery->update();

    auto& contacts = m_meshNarrowQuery->outContacts()->getData();
    if (contacts.size() > 0)
    {
        const int threads = 128;
        const int blocks = (contacts.size() + threads - 1) / threads;
        CD_AppendMeshContactsKernel<TDataType><<<blocks, threads>>>(
            out,
            contacts,
            rb.batch_bodies,
            rb.friction_mu,
            m_maxBodies,
            num_envs);
        cudaDeviceSynchronize();
    }
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::Detect(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    if (!m_initialized)
    {
        spdlog::warn("[MeshCollisionDetector] Detect called before Initialize.");
        return;
    }

    out.collision_nums.reset();

    const int totalBodies = num_envs * m_maxBodies;
    if (m_bodyAABBs.size() != static_cast<uint>(totalBodies))
        m_bodyAABBs.resize(totalBodies);

    {
        const int threads = 128;
        const int blocks = (totalBodies + threads - 1) / threads;
        CD_ComputeBodyAABBsKernel<TDataType><<<blocks, threads>>>(
            m_bodyAABBs,
            rb.batch_bodies,
            rb.batch_pos,
            rb.batch_rot,
            rb.shape_type,
            rb.shape_idx,
            rb.boxes,
            rb.spheres,
            rb.capsules,
            m_dHat,
            m_maxBodies);
        cudaDeviceSynchronize();
    }

    m_bodyBroadPhase->inSource()->assign(m_bodyAABBs);
    m_bodyBroadPhase->inTarget()->assign(m_bodyAABBs);
    m_bodyBroadPhase->update();

    CArray<int> hBatchBodies;
    CArrayList<int> hContactList;

    hBatchBodies.assign(rb.batch_bodies);
    hContactList.assign(m_bodyBroadPhase->outContactList()->getData());

    std::vector<BodyPair> bodyPairsHost;
    bodyPairsHost.reserve(128);
    std::unordered_set<uint64_t> pairSet;

    for (int q = 0; q < totalBodies && q < static_cast<int>(hContactList.size()); ++q)
    {
        int envA = q / m_maxBodies;
        int bodyA = q - envA * m_maxBodies;
        if (envA < 0 || envA >= num_envs)
            continue;
        if (bodyA >= hBatchBodies[envA])
            continue;

        auto& nbr = hContactList[q];
        for (auto it = nbr.begin(); it != nbr.end(); ++it)
        {
            int r = *it;
            int envB = r / m_maxBodies;
            int bodyB = r - envB * m_maxBodies;
            if (envA != envB)
                continue;
            if (bodyB < 0 || bodyB >= hBatchBodies[envA])
                continue;
            if (bodyA == bodyB)
                continue;

            int a = bodyA;
            int b = bodyB;
            if (a > b)
            {
                int t = a;
                a = b;
                b = t;
            }

            uint64_t key = (static_cast<uint64_t>(envA) << 40)
                | (static_cast<uint64_t>(a) << 20)
                | static_cast<uint64_t>(b);
            if (pairSet.insert(key).second)
            {
                BodyPair p;
                p.env_id = envA;
                p.body_a = a;
                p.body_b = b;
                bodyPairsHost.push_back(p);
            }
        }
    }

    if (!bodyPairsHost.empty())
    {
        CArray<BodyPair> hPairs(static_cast<uint>(bodyPairsHost.size()));
        for (uint i = 0; i < hPairs.size(); ++i)
            hPairs[i] = bodyPairsHost[i];
        m_bodyPairs.assign(hPairs);

        const int threads = 128;
        const int blocks = (m_bodyPairs.size() + threads - 1) / threads;
        CD_NarrowPrimitivePairsKernel<TDataType><<<blocks, threads>>>(
            out,
            m_bodyPairs,
            rb.shape_type,
            rb.shape_idx,
            rb.batch_pos,
            rb.batch_rot,
            rb.boxes,
            rb.spheres,
            rb.capsules,
            rb.is_static,
            rb.parent_idx,
            rb.friction_mu,
            m_cubeTemplate.vertices,
            m_cubeTemplate.triangles,
            m_dHat);
        cudaDeviceSynchronize();
    }

    detectMeshMeshByNeighborQuery(rb, out, num_envs);
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::DetectGround(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    if (!m_initialized)
        return;

    const int threads = 128;
    const int blocks = (num_envs + threads - 1) / threads;
    CD_GroundCollisionKernel<TDataType><<<blocks, threads>>>(
        out,
        rb.batch_bodies,
        rb.is_static,
        rb.batch_pos,
        rb.batch_rot,
        rb.shape_type,
        rb.shape_idx,
        rb.boxes,
        rb.spheres,
        rb.capsules,
        num_envs);
    cudaDeviceSynchronize();
}

template class MeshCollisionDetector<DataType3f>;

} // namespace dyno
