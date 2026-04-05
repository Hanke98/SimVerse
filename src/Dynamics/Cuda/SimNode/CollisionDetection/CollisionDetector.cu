#include "CollisionDetector.h"
#include "CubeToMesh.h"
#include "MeshMeshCollisionInternal.cuh"

#include "PhysicalField/RigidBody/RigidBody.h"
#include "Utils/Constraints.h"

#include "Array/ArrayList.h"
#include "Collision/CollisionDetectionAlgorithm.h"
#include "Collision/CollisionDetectionBroadPhase.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <thrust/sort.h>
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
        out.normal(env_id, idx) = -normal;
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
    DArray<int> batch_body_offset,
    DArray2D<Vector<typename TDataType::Real, 3>> batch_pos,
    DArray2D<SquareMatrix<typename TDataType::Real, 3>> batch_rot,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    typename TDataType::Real dHat,
    int num_envs,
    int totalBodies)
{
    using Real = typename TDataType::Real;
    using Coord = Vector<Real, 3>;
    using Matrix = SquareMatrix<Real, 3>;
    using AABB = TAlignedBox3D<Real>;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bodyAabbs.size() || tid >= totalBodies)
        return;

    AABB box;
    box.v0 = Coord(Real(1e30));
    box.v1 = Coord(Real(-1e30));

    int envCount = num_envs;
    if (envCount > batch_bodies.size())
        envCount = batch_bodies.size();
    if (envCount > batch_body_offset.size())
        envCount = batch_body_offset.size();

    int left = 0;
    int right = envCount - 1;
    int env_id = -1;
    while (left <= right)
    {
        const int mid = left + ((right - left) >> 1);
        const int offset = batch_body_offset[mid];
        if (offset <= tid)
        {
            env_id = mid;
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    if (env_id < 0 || env_id >= envCount)
    {
        printf("env_id error in compute AABB");
        return;
    }

    const int body_id = tid - batch_body_offset[env_id];

    if (body_id >= 0 && body_id < batch_bodies[env_id])
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
            Matrix shapeRot = rot * b.rot.toMatrix3x3();
            Coord center = pos + rot * b.center;

            Real ex = std::abs(shapeRot(0, 0)) * b.halfLength[0]
                + std::abs(shapeRot(0, 1)) * b.halfLength[1]
                + std::abs(shapeRot(0, 2)) * b.halfLength[2];
            Real ey = std::abs(shapeRot(1, 0)) * b.halfLength[0]
                + std::abs(shapeRot(1, 1)) * b.halfLength[1]
                + std::abs(shapeRot(1, 2)) * b.halfLength[2];
            Real ez = std::abs(shapeRot(2, 0)) * b.halfLength[0]
                + std::abs(shapeRot(2, 1)) * b.halfLength[1]
                + std::abs(shapeRot(2, 2)) * b.halfLength[2];

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
        else if (st == 6)
        {
            //TODO: compute compact AABB for mesh
        }
    }

    bodyAabbs[tid] = box;
}

__device__ inline int CD_FindEnvIdByBodyOffset(
    const DArray<int>& batch_body_offset,
    int num_envs,
    int flat_body_id)
{
    int left = 0;
    int right = num_envs - 1;
    int env_id = -1;

    while (left <= right)
    {
        const int mid = left + ((right - left) >> 1);
        const int offset = batch_body_offset[mid];
        if (offset <= flat_body_id)
        {
            env_id = mid;
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    return env_id;
}

__device__ inline bool CD_DecodeBroadPhaseBodyIndex(
    int encodedId,
    const DArray<int>& batch_bodies,
    const DArray<int>& batch_body_offset,
    int num_envs,
    int& env_id,
    int& local_body_id)
{
    if (encodedId < 0)
        return false;

    int envCount = num_envs;
    if (envCount > batch_bodies.size())
        envCount = batch_bodies.size();
    if (envCount > batch_body_offset.size())
        envCount = batch_body_offset.size();
    if (envCount <= 0)
        return false;

    env_id = CD_FindEnvIdByBodyOffset(batch_body_offset, envCount, encodedId);
    if (env_id < 0 || env_id >= envCount)
    {
        printf("invalid env_id in collision detection\n");
        return false;
    }
        

    local_body_id = encodedId - batch_body_offset[env_id];

    if (local_body_id < 0 || local_body_id >= batch_bodies[env_id])
    {
        printf("invalid local_body_id in collision detection\n");
        return false;
    }

    return true;
}

__global__ void CD_CountBodyContactListSize(
    DArray<int> num,
    DArrayList<int> contactList,
    DArray<int> batch_bodies,
    DArray<int> batch_body_offset,
    int num_envs)
{
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tId >= contactList.size())
        return;

    int env_id = -1;
    int body_id = -1;
    if (!CD_DecodeBroadPhaseBodyIndex(
        tId,
        batch_bodies,
        batch_body_offset,
        num_envs,
        env_id,
        body_id))
    {
        num[tId] = 0;
        return;
    }

    int validCount = 0;
    auto& list_i = contactList[tId];
    for (int j = 0; j < list_i.size(); ++j)
    {
        int nbr_env_id = -1;
        int nbr_body_id = -1;
        if (!CD_DecodeBroadPhaseBodyIndex(
            list_i[j],
            batch_bodies,
            batch_body_offset,
            num_envs,
            nbr_env_id,
            nbr_body_id))
            continue;

        if (nbr_env_id != env_id || nbr_body_id == body_id)
            continue;

        ++validCount;
    }

    num[tId] = validCount;
}

__global__ void CD_SetupBodyContactIds(
    DArray<BodyContactId> ids,
    DArray<int> index,
    DArrayList<int> contactList,
    DArray<int> batch_bodies,
    DArray<int> batch_body_offset,
    int num_envs)
{
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tId >= contactList.size())
        return;

    int env_id = -1;
    int body_id = -1;
    if (!CD_DecodeBroadPhaseBodyIndex(
        tId,
        batch_bodies,
        batch_body_offset,
        num_envs,
        env_id,
        body_id))
        return;

    const int base = index[tId];

    auto& list_i = contactList[tId];
    int cursor = 0;
    for (int j = 0; j < list_i.size(); ++j)
    {
        int nbr_env_id = -1;
        int nbr_body_id = -1;
        if (!CD_DecodeBroadPhaseBodyIndex(
            list_i[j],
            batch_bodies,
            batch_body_offset,
            num_envs,
            nbr_env_id,
            nbr_body_id))
            continue;

        if (nbr_env_id != env_id || nbr_body_id == body_id)
            continue;

        BodyContactId id;
        id.env_id = env_id;
        id.body_id_1 = body_id;
        id.body_id_2 = nbr_body_id;
        ids[base + cursor] = id;
        ++cursor;
    }
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
    Coord point = cp.pos2;

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

} // namespace

template<typename TDataType>
void MeshCollisionDetector<TDataType>::Initialize(int num_envs, int max_bodies, const RigidBody<TDataType>&)
{
    m_numEnvs = num_envs;
    m_maxBodies = max_bodies;
    m_cachedMeshShapeCount = -1;

    GenerateUnitCubeMesh(m_cubeTemplate);
    m_cubeTemplateTriSet = std::make_shared<TriangleSet<TDataType>>();

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

    m_cubeTemplateTriSet->setPoints(m_cubeVerticesHost);
    m_cubeTemplateTriSet->setTriangles(m_cubeTrianglesHost);
    m_cubeTemplateTriSet->update();

    m_bodyAABBs.resize(num_envs * max_bodies);

    m_bodyBroadPhase = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
    m_bodyBroadPhase->varSelfCollision()->setValue(true);
    m_bodyBroadPhase->varAccelerationStructure()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
    m_bodyBroadPhase->varGridSizeLimit()->setValue(Real(0.01));

    m_initialized = true;
    spdlog::info("[MeshCollisionDetector] Initialized (envs={}, maxBodies={})", num_envs, max_bodies);
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::refreshMeshShapeLayoutCache(int shapeCount)
{
    if (shapeCount == m_cachedMeshShapeCount)
        return;

    if (shapeCount <= 0)
    {
        m_shape2PatchOffsets.clear();
        m_shape2TriOffsets.clear();
        m_shape2EdgeOffsets.clear();
        m_shape2VertexOffsets.clear();
        m_patch2Shape.clear();
        m_patch2TriOffsets.clear();
        m_patch2TriIndices.clear();
        m_cachedMeshShapeCount = 0;
        return;
    }

    const int templatePatchCount = m_cubeTemplate.numPatches > 0 ? m_cubeTemplate.numPatches : 1;
    const int templateTriCount = m_cubeTemplateTriSet != nullptr
        ? static_cast<int>(m_cubeTemplateTriSet->triangleIndices().size())
        : (m_cubeTemplate.numTriangles > 0 ? m_cubeTemplate.numTriangles : static_cast<int>(m_cubeTrianglesHost.size()));
    const int templateEdgeCount = m_cubeTemplateTriSet != nullptr
        ? static_cast<int>(m_cubeTemplateTriSet->edgeIndices().size())
        : (m_cubeTemplate.numEdges > 0 ? m_cubeTemplate.numEdges : static_cast<int>(m_cubeTemplate.edges.size()));
    const int templateVertexCount = m_cubeTemplateTriSet != nullptr
        ? static_cast<int>(m_cubeTemplateTriSet->getPoints().size())
        : (m_cubeTemplate.numVertices > 0 ? m_cubeTemplate.numVertices : static_cast<int>(m_cubeVerticesHost.size()));

    CArray<int> tplPatchOffsets;
    CArray<int> tplPatchFaces;
    if (m_cubeTemplate.patchOffsets.size() > 0)
        tplPatchOffsets.assign(m_cubeTemplate.patchOffsets);
    if (m_cubeTemplate.patchFaces.size() > 0)
        tplPatchFaces.assign(m_cubeTemplate.patchFaces);

    std::vector<int> shape2PatchOffsets(shapeCount + 1, 0);
    std::vector<int> shape2TriOffsets(shapeCount + 1, 0);
    std::vector<int> shape2EdgeOffsets(shapeCount + 1, 0);
    std::vector<int> shape2VertexOffsets(shapeCount + 1, 0);
    std::vector<int> patch2Shape(shapeCount * templatePatchCount, -1);
    std::vector<int> patch2TriOffsets(shapeCount * templatePatchCount + 1, 0);
    std::vector<int> patch2TriIndices;
    patch2TriIndices.reserve(shapeCount * templateTriCount);

    for (int shapeId = 0; shapeId < shapeCount; ++shapeId)
    {
        shape2PatchOffsets[shapeId + 1] = shape2PatchOffsets[shapeId] + templatePatchCount;
        shape2TriOffsets[shapeId + 1] = shape2TriOffsets[shapeId] + templateTriCount;
        shape2EdgeOffsets[shapeId + 1] = shape2EdgeOffsets[shapeId] + templateEdgeCount;
        shape2VertexOffsets[shapeId + 1] = shape2VertexOffsets[shapeId] + templateVertexCount;

        const int triBase = shape2TriOffsets[shapeId];
        for (int localPatchId = 0; localPatchId < templatePatchCount; ++localPatchId)
        {
            const int globalPatchId = shape2PatchOffsets[shapeId] + localPatchId;
            patch2Shape[globalPatchId] = shapeId;
            patch2TriOffsets[globalPatchId] = static_cast<int>(patch2TriIndices.size());

            bool usedTemplatePatchFaces = false;
            if (tplPatchOffsets.size() >= static_cast<uint>(templatePatchCount + 1)
                && tplPatchFaces.size() > 0)
            {
                int begin = tplPatchOffsets[localPatchId];
                int end = tplPatchOffsets[localPatchId + 1];
                begin = begin < 0 ? 0 : begin;
                end = end > static_cast<int>(tplPatchFaces.size()) ? static_cast<int>(tplPatchFaces.size()) : end;
                for (int fi = begin; fi < end; ++fi)
                {
                    const int localTriId = tplPatchFaces[fi];
                    if (localTriId < 0 || localTriId >= templateTriCount)
                        continue;
                    patch2TriIndices.push_back(triBase + localTriId);
                    usedTemplatePatchFaces = true;
                }
            }

            if (!usedTemplatePatchFaces)
            {
                for (int localTriId = 0; localTriId < templateTriCount; ++localTriId)
                    patch2TriIndices.push_back(triBase + localTriId);
            }

            patch2TriOffsets[globalPatchId + 1] = static_cast<int>(patch2TriIndices.size());
        }
    }

    CArray<int> dShape2PatchOffsets(shapeCount + 1);
    CArray<int> dShape2TriOffsets(shapeCount + 1);
    CArray<int> dShape2EdgeOffsets(shapeCount + 1);
    CArray<int> dShape2VertexOffsets(shapeCount + 1);
    for (int i = 0; i <= shapeCount; ++i)
    {
        dShape2PatchOffsets[i] = shape2PatchOffsets[i];
        dShape2TriOffsets[i] = shape2TriOffsets[i];
        dShape2EdgeOffsets[i] = shape2EdgeOffsets[i];
        dShape2VertexOffsets[i] = shape2VertexOffsets[i];
    }

    CArray<int> dPatch2Shape(static_cast<uint>(patch2Shape.size()));
    CArray<int> dPatch2TriOffsets(static_cast<uint>(patch2TriOffsets.size()));
    CArray<int> dPatch2TriIndices(static_cast<uint>(patch2TriIndices.size()));
    for (uint i = 0; i < dPatch2Shape.size(); ++i)
        dPatch2Shape[i] = patch2Shape[i];
    for (uint i = 0; i < dPatch2TriOffsets.size(); ++i)
        dPatch2TriOffsets[i] = patch2TriOffsets[i];
    for (uint i = 0; i < dPatch2TriIndices.size(); ++i)
        dPatch2TriIndices[i] = patch2TriIndices[i];

    m_shape2PatchOffsets.assign(dShape2PatchOffsets);
    m_shape2TriOffsets.assign(dShape2TriOffsets);
    m_shape2EdgeOffsets.assign(dShape2EdgeOffsets);
    m_shape2VertexOffsets.assign(dShape2VertexOffsets);
    m_patch2Shape.assign(dPatch2Shape);
    m_patch2TriOffsets.assign(dPatch2TriOffsets);
    m_patch2TriIndices.assign(dPatch2TriIndices);
    m_cachedMeshShapeCount = shapeCount;
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::detectMeshMeshInternal(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    CArray<int> hBatchBodies;
    CArrayList<int> hContactList;
    hBatchBodies.assign(rb.batch_bodies);
    hContactList.assign(m_bodyBroadPhase->outContactList()->getData());

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

    HostBlockVector<int> hIsStatic;
    HostBlockVector<int> hParentIdx;
    rb.is_static.Download(hIsStatic);
    rb.parent_idx.Download(hParentIdx);

    std::vector<int> shape2BodyFlatHost;
    std::vector<Coord> shapeCentersHost;
    std::vector<Matrix> shapeRotationsHost;
    std::vector<Coord> shapeHalfLengthsHost;
    std::vector<Coord> shapeInvHalfLengthsHost;
    std::vector<int> flatBodyToShape(num_envs * m_maxBodies, -1);
    shape2BodyFlatHost.reserve(num_envs * 8);
    shapeCentersHost.reserve(num_envs * 8);
    shapeRotationsHost.reserve(num_envs * 8);
    shapeHalfLengthsHost.reserve(num_envs * 8);
    shapeInvHalfLengthsHost.reserve(num_envs * 8);

    for (int env = 0; env < num_envs; ++env)
    {
        int bodyCount = hBatchBodies[env];
        for (int b = 0; b < bodyCount; ++b)
        {
            if (hShapeType(env, b) != 1)
                continue;

            const int sidx = hShapeIdx(env, b);
            const BoxInfo box = hBoxes(env, sidx);
            const Coord bodyPos = hPos(env, b);
            const Matrix bodyRot = hRot(env, b);

            const int flatBody = env * m_maxBodies + b;
            const int shapeId = static_cast<int>(shape2BodyFlatHost.size());
            flatBodyToShape[flatBody] = shapeId;
            shape2BodyFlatHost.push_back(flatBody);
            shapeCentersHost.push_back(bodyPos + bodyRot * box.center);
            shapeRotationsHost.push_back(bodyRot * box.rot.toMatrix3x3());
            shapeHalfLengthsHost.push_back(box.halfLength);
            shapeInvHalfLengthsHost.push_back(Coord(
                box.halfLength[0] != Real(0) ? Real(1) / box.halfLength[0] : Real(0),
                box.halfLength[1] != Real(0) ? Real(1) / box.halfLength[1] : Real(0),
                box.halfLength[2] != Real(0) ? Real(1) / box.halfLength[2] : Real(0)));
        }
    }

    const int shapeCount = static_cast<int>(shape2BodyFlatHost.size());
    if (shapeCount < 2)
        return;

    refreshMeshShapeLayoutCache(shapeCount);

    CArray<int> dShape2BodyFlat(shapeCount);
    CArray<Coord> dShapeCenters(shapeCount);
    CArray<Matrix> dShapeRotations(shapeCount);
    CArray<Coord> dShapeHalfLengths(shapeCount);
    CArray<Coord> dShapeInvHalfLengths(shapeCount);
    for (int i = 0; i < shapeCount; ++i)
    {
        dShape2BodyFlat[i] = shape2BodyFlatHost[i];
        dShapeCenters[i] = shapeCentersHost[i];
        dShapeRotations[i] = shapeRotationsHost[i];
        dShapeHalfLengths[i] = shapeHalfLengthsHost[i];
        dShapeInvHalfLengths[i] = shapeInvHalfLengthsHost[i];
    }
    m_shape2BodyFlat.assign(dShape2BodyFlat);
    m_shapeCenters.assign(dShapeCenters);
    m_shapeRotations.assign(dShapeRotations);
    m_shapeHalfLengths.assign(dShapeHalfLengths);
    m_shapeInvHalfLengths.assign(dShapeInvHalfLengths);

    std::unordered_set<uint64_t> pairSet;
    std::vector<PairUU> shapePairsHost;
    shapePairsHost.reserve(128);

    const int totalBodies = num_envs * m_maxBodies;
    for (int q = 0; q < totalBodies && q < static_cast<int>(hContactList.size()); ++q)
    {
        const int envA = q / m_maxBodies;
        const int bodyA = q - envA * m_maxBodies;
        if (envA < 0 || envA >= num_envs)
            continue;
        if (bodyA < 0 || bodyA >= m_maxBodies || bodyA >= hBatchBodies[envA])
            continue;
        if (hShapeType(envA, bodyA) != 1)
            continue;

        auto& nbr = hContactList[q];
        for (auto it = nbr.begin(); it != nbr.end(); ++it)
        {
            const int r = *it;
            const int envB = r / m_maxBodies;
            const int bodyB = r - envB * m_maxBodies;
            if (envA != envB)
                continue;
            if (bodyB < 0 || bodyB >= m_maxBodies || bodyB >= hBatchBodies[envA])
                continue;
            if (bodyA == bodyB)
                continue;
            if (hShapeType(envA, bodyB) != 1)
                continue;

            int a = bodyA;
            int b = bodyB;
            if (a > b)
            {
                const int t = a;
                a = b;
                b = t;
            }

            if (!hIsStatic.Empty()
                && hIsStatic.AtBlock(envA, a)
                && hIsStatic.AtBlock(envA, b))
                continue;
            if (!hParentIdx.Empty())
            {
                if (hParentIdx.AtBlock(envA, a) == b
                    || hParentIdx.AtBlock(envA, b) == a)
                    continue;
            }

            const uint64_t key = (static_cast<uint64_t>(envA) << 40)
                | (static_cast<uint64_t>(a) << 20)
                | static_cast<uint64_t>(b);
            if (!pairSet.insert(key).second)
                continue;

            const int flatA = envA * m_maxBodies + a;
            const int flatB = envA * m_maxBodies + b;
            const int shapeA = flatBodyToShape[flatA];
            const int shapeB = flatBodyToShape[flatB];
            if (shapeA < 0 || shapeB < 0)
                continue;

            shapePairsHost.emplace_back(static_cast<uint>(shapeA), static_cast<uint>(shapeB));
        }
    }

    if (shapePairsHost.empty())
        return;

    CArray<PairUU> dShapePairs(static_cast<uint>(shapePairsHost.size()));
    for (uint i = 0; i < dShapePairs.size(); ++i)
        dShapePairs[i] = shapePairsHost[i];
    m_shapePairs.assign(dShapePairs);
    m_patchPairs.clear();

    cd_internal::MeshShapeView<TDataType> view{
        m_cubeTemplateTriSet->getPoints(),
        m_cubeTemplateTriSet->triangleIndices(),
        m_cubeTemplateTriSet->triangle2Edge(),
        m_cubeTemplateTriSet->edgeIndices(),
        m_cubeTemplateTriSet->edge2Triangle(),
        m_shapePairs,
        m_shape2BodyFlat,
        m_shapeCenters,
        m_shapeRotations,
        m_shapeHalfLengths,
        m_shapeInvHalfLengths,
        m_shape2PatchOffsets,
        m_shape2TriOffsets,
        m_shape2EdgeOffsets,
        m_shape2VertexOffsets,
        m_patch2Shape,
        m_patch2TriOffsets,
        m_patch2TriIndices,
        m_cubeTemplate.patchAABBs,
        m_patchPairs,
        m_triAabbsWorld,
        m_faceNormalsWorld,
        m_edgeNormalsWorld,
        m_dHat,
        m_edgeEdgeActivationMargin
    };

    const int templateTriCount = static_cast<int>(m_cubeTemplateTriSet->triangleIndices().size());
    const int templateEdgeCount = static_cast<int>(m_cubeTemplateTriSet->edgeIndices().size());
    const int triCount = shapeCount * templateTriCount;
    const int edgeCount = shapeCount * templateEdgeCount;
    if (triCount <= 0)
        return;

    m_triAabbsWorld.resize(triCount);
    m_faceNormalsWorld.resize(triCount);
    if (edgeCount > 0)
        m_edgeNormalsWorld.resize(edgeCount);

    view.triangleAabbsWorld = m_triAabbsWorld;
    view.faceNormalsWorld = m_faceNormalsWorld;
    view.edgeNormalsWorld = m_edgeNormalsWorld;

    {
        const int threads = 128;
        const int triBlocks = (triCount + threads - 1) / threads;
        cd_internal::PrepareTriangleWorldDataKernel<decltype(view)><<<triBlocks, threads>>>(view);
        if (edgeCount > 0)
        {
            const int edgeBlocks = (edgeCount + threads - 1) / threads;
            cd_internal::PrepareEdgeNormalsWorldKernel<decltype(view)><<<edgeBlocks, threads>>>(view);
        }
        cudaDeviceSynchronize();
    }

    const int shapePairCount = static_cast<int>(m_shapePairs.size());
    m_patchPairTriPairCounts.resize(shapePairCount);
    m_patchPairTriPairCounts.reset();
    {
        const int threads = 128;
        const int blocks = (shapePairCount + threads - 1) / threads;
        cd_internal::CountTriPairsPerShapePairKernel<<<blocks, threads>>>(
            m_patchPairTriPairCounts,
            m_shapePairs,
            m_shape2TriOffsets);
        cudaDeviceSynchronize();
    }

    const int totalCandidateTriPairs = shapePairCount > 0
        ? m_reduce.accumulate(m_patchPairTriPairCounts.begin(), m_patchPairTriPairCounts.size())
        : 0;
    if (totalCandidateTriPairs <= 0)
        return;

    m_patchPairTriPairOffsets.resize(shapePairCount);
    m_patchPairTriPairOffsets.assign(m_patchPairTriPairCounts);
    m_scan.exclusive(m_patchPairTriPairOffsets, true);

    m_candidateTri0.resize(totalCandidateTriPairs);
    m_candidateTri1.resize(totalCandidateTriPairs);
    m_candidatePatchPairId.resize(totalCandidateTriPairs);
    {
        const int threads = 128;
        const int blocks = (shapePairCount + threads - 1) / threads;
        cd_internal::SetTriPairsFromShapePairsKernel<<<blocks, threads>>>(
            m_candidateTri0,
            m_candidateTri1,
            m_candidatePatchPairId,
            m_patchPairTriPairOffsets,
            m_patchPairTriPairCounts,
            m_shapePairs,
            m_shape2TriOffsets);
        cudaDeviceSynchronize();
    }

    m_coarsePassCounts.resize(totalCandidateTriPairs);
    m_coarsePassCounts.reset();
    {
        const int threads = 128;
        const int blocks = (totalCandidateTriPairs + threads - 1) / threads;
        cd_internal::CountCoarsePassedTriPairsKernel<AABB, Real><<<blocks, threads>>>(
            m_coarsePassCounts,
            m_candidateTri0,
            m_candidateTri1,
            m_triAabbsWorld,
            m_dHat);
        cudaDeviceSynchronize();
    }

    const int totalFilteredTriPairs = totalCandidateTriPairs > 0
        ? m_reduce.accumulate(m_coarsePassCounts.begin(), m_coarsePassCounts.size())
        : 0;
    if (totalFilteredTriPairs <= 0)
        return;

    m_coarsePassOffsets.resize(totalCandidateTriPairs);
    m_coarsePassOffsets.assign(m_coarsePassCounts);
    m_scan.exclusive(m_coarsePassOffsets, true);

    m_filteredTri0.resize(totalFilteredTriPairs);
    m_filteredTri1.resize(totalFilteredTriPairs);
    m_filteredPatchPairId.resize(totalFilteredTriPairs);
    {
        const int threads = 128;
        const int blocks = (totalCandidateTriPairs + threads - 1) / threads;
        cd_internal::SetCoarsePassedTriPairsKernel<<<blocks, threads>>>(
            m_filteredTri0,
            m_filteredTri1,
            m_filteredPatchPairId,
            m_candidateTri0,
            m_candidateTri1,
            m_candidatePatchPairId,
            m_coarsePassOffsets,
            m_coarsePassCounts);
        cudaDeviceSynchronize();
    }

    const int primitivePassSlotCount = totalFilteredTriPairs * cd_internal::MESH_PASS_COUNT;
    m_primitivePassCounts.resize(primitivePassSlotCount);
    m_primitivePassCounts.reset();
    {
        const int threads = 128;
        const int blocks = (primitivePassSlotCount + threads - 1) / threads;
        cd_internal::CountPrimitiveCandidatesPerPassKernel<decltype(view)><<<blocks, threads>>>(
            m_primitivePassCounts,
            m_filteredTri0,
            m_filteredTri1,
            m_filteredPatchPairId,
            view);
        cudaDeviceSynchronize();
    }

    const int totalPrimitiveCandidates = primitivePassSlotCount > 0
        ? m_reduce.accumulate(m_primitivePassCounts.begin(), m_primitivePassCounts.size())
        : 0;

    m_primitivePassOffsets.resize(primitivePassSlotCount);
    if (primitivePassSlotCount > 0)
    {
        m_primitivePassOffsets.assign(m_primitivePassCounts);
        m_scan.exclusive(m_primitivePassOffsets, true);
    }

    if (totalPrimitiveCandidates > 0)
    {
        m_primitiveCandidateContacts.resize(totalPrimitiveCandidates);
        m_primitiveCandidateKeys.resize(totalPrimitiveCandidates);
        m_primitiveCandidateSortedIndices.resize(totalPrimitiveCandidates);
        m_primitiveCandidateKeepFlags.resize(totalPrimitiveCandidates);
        m_primitiveCandidateKeepFlags.reset();

        const int threads = 128;
        const int passBlocks = (primitivePassSlotCount + threads - 1) / threads;
        cd_internal::SetPrimitiveCandidatesPerPassKernel<decltype(view)><<<passBlocks, threads>>>(
            m_primitiveCandidateContacts,
            m_primitiveCandidateKeys,
            m_primitivePassOffsets,
            m_primitivePassCounts,
            m_filteredTri0,
            m_filteredTri1,
            m_filteredPatchPairId,
            view);
        cudaDeviceSynchronize();

        const int candBlocks = (totalPrimitiveCandidates + threads - 1) / threads;
        cd_internal::InitPrimitiveCandidateIndicesKernel<<<candBlocks, threads>>>(
            m_primitiveCandidateSortedIndices);
        cudaDeviceSynchronize();

        thrust::stable_sort_by_key(
            thrust::device,
            m_primitiveCandidateKeys.begin(),
            m_primitiveCandidateKeys.begin() + m_primitiveCandidateKeys.size(),
            m_primitiveCandidateSortedIndices.begin());

        cd_internal::MarkMinDepthCandidatesPerPrimitiveKeyKernel<ContactPair, Real><<<candBlocks, threads>>>(
            m_primitiveCandidateKeepFlags,
            m_primitiveCandidateKeys,
            m_primitiveCandidateSortedIndices,
            m_primitiveCandidateContacts,
            Real(1e-6),
            Real(1e-4));
        cudaDeviceSynchronize();

        const Real crossTypePositionEps = m_edgeEdgeActivationMargin > Real(2e-4)
            ? m_edgeEdgeActivationMargin * Real(0.5)
            : Real(1e-4);
        const int filteredBlocks = (totalFilteredTriPairs + threads - 1) / threads;
        cd_internal::SuppressRedundantEdgeFaceAgainstVertexFaceKernel<ContactPair, Real><<<filteredBlocks, threads>>>(
            m_primitiveCandidateKeepFlags,
            m_primitivePassCounts,
            m_primitivePassOffsets,
            m_primitiveCandidateContacts,
            crossTypePositionEps,
            Real(1e-4));
        cudaDeviceSynchronize();
    }

    m_selectedPrimitiveCounts.resize(totalFilteredTriPairs);
    m_selectedPrimitiveCounts.reset();
    if (totalPrimitiveCandidates > 0)
    {
        const int threads = 128;
        const int blocks = (totalFilteredTriPairs + threads - 1) / threads;
        cd_internal::CountSelectedPrimitiveContactsPerTriPairKernel<<<blocks, threads>>>(
            m_selectedPrimitiveCounts,
            m_primitivePassCounts,
            m_primitivePassOffsets,
            m_primitiveCandidateKeepFlags);
        cudaDeviceSynchronize();
    }

    m_triPairContactCounts.resize(totalFilteredTriPairs);
    m_triPairContactCounts.reset();
    {
        const int threads = 128;
        const int blocks = (totalFilteredTriPairs + threads - 1) / threads;
        cd_internal::SetFinalContactCountsKernel<<<blocks, threads>>>(
            m_triPairContactCounts,
            m_selectedPrimitiveCounts);
        cudaDeviceSynchronize();
    }

    const int totalContacts = totalFilteredTriPairs > 0
        ? m_reduce.accumulate(m_triPairContactCounts.begin(), m_triPairContactCounts.size())
        : 0;
    if (totalContacts <= 0)
        return;

    m_triPairContactOffsets.resize(totalFilteredTriPairs);
    m_triPairContactOffsets.assign(m_triPairContactCounts);
    m_scan.exclusive(m_triPairContactOffsets, true);
    m_meshContacts.resize(totalContacts);
    {
        const int threads = 128;
        const int blocks = (totalFilteredTriPairs + threads - 1) / threads;
        cd_internal::SetFinalContactsPerTriPairKernel<ContactPair><<<blocks, threads>>>(
            m_meshContacts,
            m_triPairContactOffsets,
            m_primitivePassCounts,
            m_primitivePassOffsets,
            m_primitiveCandidateKeepFlags,
            m_primitiveCandidateContacts,
            m_selectedPrimitiveCounts);
        cudaDeviceSynchronize();
    }

    {
        const int threads = 128;
        const int blocks = (totalContacts + threads - 1) / threads;
        CD_AppendMeshContactsKernel<TDataType><<<blocks, threads>>>(
            out,
            m_meshContacts,
            rb.batch_bodies,
            rb.friction_mu,
            m_maxBodies,
            num_envs);
        cudaDeviceSynchronize();
    }
}

template<typename TDataType>
bool MeshCollisionDetector<TDataType>::broad_phase(
    const RigidBody<TDataType>& rb,
    int num_envs)
{
    CArray<int> hBatchBodies;
    hBatchBodies.assign(rb.batch_bodies);

    int totalBodies = 0;
    for (int env_id = 0; env_id < num_envs && env_id < static_cast<int>(hBatchBodies.size()); ++env_id)
        totalBodies += hBatchBodies[env_id];

    if (totalBodies <= 0)
    {
        m_bodyAABBs.resize(0);
        m_bodyContactPairs.resize(0);
        return false;
    }

    if (m_bodyAABBs.size() != static_cast<uint>(totalBodies))
        m_bodyAABBs.resize(totalBodies);

    {
        const int threads = 128;
        const int blocks = (totalBodies + threads - 1) / threads;
        CD_ComputeBodyAABBsKernel<TDataType><<<blocks, threads>>>(
            m_bodyAABBs,
            rb.batch_bodies,
            rb.batch_body_offset,
            rb.batch_pos,
            rb.batch_rot,
            rb.shape_type,
            rb.shape_idx,
            rb.boxes,
            rb.spheres,
            rb.capsules,
            m_dHat,
            num_envs,
            totalBodies);
        cudaDeviceSynchronize();
    }

    m_bodyBroadPhase->inSource()->assign(m_bodyAABBs);
    m_bodyBroadPhase->inTarget()->assign(m_bodyAABBs);
    m_bodyBroadPhase->update();

    auto& contactList = m_bodyBroadPhase->outContactList()->getData();
    if (contactList.size() == 0 || contactList.elementSize() == 0)
    {
        m_bodyContactPairs.resize(0);
        return true;
    }

    DArray<int> count(contactList.size());
    {
        const int threads = 128;
        const int blocks = (contactList.size() + threads - 1) / threads;
        CD_CountBodyContactListSize<<<blocks, threads>>>(
            count,
            contactList,
            rb.batch_bodies,
            rb.batch_body_offset,
            num_envs);
        cudaDeviceSynchronize();
    }

    const int totalSize = count.size() > 0
        ? m_reduce.accumulate(count.begin(), count.size())
        : 0;
    if (totalSize <= 0)
    {
        m_bodyContactPairs.resize(0);
        return true;
    }

    m_scan.exclusive(count);

    m_bodyContactPairs.resize(totalSize);
    {
        const int threads = 128;
        const int blocks = (contactList.size() + threads - 1) / threads;
        CD_SetupBodyContactIds<<<blocks, threads>>>(
            m_bodyContactPairs,
            count,
            contactList,
            rb.batch_bodies,
            rb.batch_body_offset,
            num_envs);
        cudaDeviceSynchronize();
    }

    return true;
}

template<typename TDataType>
bool MeshCollisionDetector<TDataType>::middle_phase(
    const RigidBody<TDataType>& rb,
    int num_envs,
    std::vector<BodyPair>& bodyPairsHost)
{
    const int totalBodies = num_envs * m_maxBodies;
    if (totalBodies <= 0)
        return false;

    CArray<int> hBatchBodies;
    CArrayList<int> hContactList;

    hBatchBodies.assign(rb.batch_bodies);
    hContactList.assign(m_bodyBroadPhase->outContactList()->getData());

    bodyPairsHost.clear();
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

    if (bodyPairsHost.empty())
    {
        m_bodyPairs.resize(0);
        return false;
    }

    CArray<BodyPair> hPairs(static_cast<uint>(bodyPairsHost.size()));
    for (uint i = 0; i < hPairs.size(); ++i)
        hPairs[i] = bodyPairsHost[i];
    m_bodyPairs.assign(hPairs);

    return true;
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::narrow_phase(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    // detectMeshMeshInternal(rb, out, num_envs);

    if (m_bodyContactPairs.size() == 0)
        return;

    CArray<int> hBatchBodies;
    CArray<BodyContactId> hBodyContactPairs;
    hBatchBodies.assign(rb.batch_bodies);
    hBodyContactPairs.assign(m_bodyContactPairs);

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

    HostBlockVector<int> hIsStatic;
    HostBlockVector<int> hParentIdx;
    rb.is_static.Download(hIsStatic);
    rb.parent_idx.Download(hParentIdx);

    std::unordered_set<uint64_t> pairSet;
    std::vector<PairUU> shapePairsHost;
    shapePairsHost.reserve(128);

    for (int q = 0; q < static_cast<int>(hBodyContactPairs.size()); ++q)
    {
        const BodyContactId pair = hBodyContactPairs[q];
        const int envA = pair.env_id;
        if (envA < 0 || envA >= num_envs)
            continue;
        const int bodyA = pair.body_id_1;
        const int bodyB = pair.body_id_2;
        if (bodyA < 0 || bodyA >= m_maxBodies || bodyB < 0 || bodyB >= m_maxBodies)
            continue;
        if (bodyA >= hBatchBodies[envA] || bodyB >= hBatchBodies[envA])
            continue;
        if (bodyA == bodyB)
            continue;
        if (hShapeType(envA, bodyA) != 1 || hShapeType(envA, bodyB) != 1)
            continue;

        int a = bodyA;
        int b = bodyB;
        if (a > b)
        {
            const int t = a;
            a = b;
            b = t;
        }

        if (!hIsStatic.Empty()
            && hIsStatic.AtBlock(envA, a)
            && hIsStatic.AtBlock(envA, b))
            continue;
        if (!hParentIdx.Empty())
        {
            if (hParentIdx.AtBlock(envA, a) == b
                || hParentIdx.AtBlock(envA, b) == a)
                continue;
        }

        const uint64_t key = (static_cast<uint64_t>(envA) << 40)
            | (static_cast<uint64_t>(a) << 20)
            | static_cast<uint64_t>(b);
        if (!pairSet.insert(key).second)
            continue;

        const int shapeA = envA * m_maxBodies + a;
        const int shapeB = envA * m_maxBodies + b;
        shapePairsHost.emplace_back(static_cast<uint>(shapeA), static_cast<uint>(shapeB));
    }

    if (shapePairsHost.empty())
        return;

    const int shapeCount = num_envs * m_maxBodies;
    if (shapeCount <= 0)
        return;

    refreshMeshShapeLayoutCache(shapeCount);

    CArray<int> dShape2BodyFlat(shapeCount);
    CArray<Coord> dShapeCenters(shapeCount);
    CArray<Matrix> dShapeRotations(shapeCount);
    CArray<Coord> dShapeHalfLengths(shapeCount);
    CArray<Coord> dShapeInvHalfLengths(shapeCount);
    for (int i = 0; i < shapeCount; ++i)
    {
        dShape2BodyFlat[i] = i;
        dShapeCenters[i] = Coord(0);
        dShapeRotations[i] = Matrix::identityMatrix();
        dShapeHalfLengths[i] = Coord(0);
        dShapeInvHalfLengths[i] = Coord(0);
    }

    for (int env = 0; env < num_envs; ++env)
    {
        const int bodyCount = hBatchBodies[env] < m_maxBodies ? hBatchBodies[env] : m_maxBodies;
        for (int b = 0; b < bodyCount; ++b)
        {
            if (hShapeType(env, b) != 1)
                continue;

            const int shapeId = env * m_maxBodies + b;
            const int sidx = hShapeIdx(env, b);
            const BoxInfo box = hBoxes(env, sidx);
            const Coord bodyPos = hPos(env, b);
            const Matrix bodyRot = hRot(env, b);

            dShapeCenters[shapeId] = bodyPos + bodyRot * box.center;
            dShapeRotations[shapeId] = bodyRot * box.rot.toMatrix3x3();
            dShapeHalfLengths[shapeId] = box.halfLength;
            dShapeInvHalfLengths[shapeId] = Coord(
                box.halfLength[0] != Real(0) ? Real(1) / box.halfLength[0] : Real(0),
                box.halfLength[1] != Real(0) ? Real(1) / box.halfLength[1] : Real(0),
                box.halfLength[2] != Real(0) ? Real(1) / box.halfLength[2] : Real(0));
        }
    }

    m_shape2BodyFlat.assign(dShape2BodyFlat);
    m_shapeCenters.assign(dShapeCenters);
    m_shapeRotations.assign(dShapeRotations);
    m_shapeHalfLengths.assign(dShapeHalfLengths);
    m_shapeInvHalfLengths.assign(dShapeInvHalfLengths);

    CArray<PairUU> dShapePairs(static_cast<uint>(shapePairsHost.size()));
    for (uint i = 0; i < dShapePairs.size(); ++i)
        dShapePairs[i] = shapePairsHost[i];
    m_shapePairs.assign(dShapePairs);
    m_patchPairs.clear();

    cd_internal::MeshShapeView<TDataType> view{
        m_cubeTemplateTriSet->getPoints(),
        m_cubeTemplateTriSet->triangleIndices(),
        m_cubeTemplateTriSet->triangle2Edge(),
        m_cubeTemplateTriSet->edgeIndices(),
        m_cubeTemplateTriSet->edge2Triangle(),
        m_shapePairs,
        m_shape2BodyFlat,
        m_shapeCenters,
        m_shapeRotations,
        m_shapeHalfLengths,
        m_shapeInvHalfLengths,
        m_shape2PatchOffsets,
        m_shape2TriOffsets,
        m_shape2EdgeOffsets,
        m_shape2VertexOffsets,
        m_patch2Shape,
        m_patch2TriOffsets,
        m_patch2TriIndices,
        m_cubeTemplate.patchAABBs,
        m_patchPairs,
        m_triAabbsWorld,
        m_faceNormalsWorld,
        m_edgeNormalsWorld,
        m_dHat,
        m_edgeEdgeActivationMargin
    };

    const int templateTriCount = static_cast<int>(m_cubeTemplateTriSet->triangleIndices().size());
    const int templateEdgeCount = static_cast<int>(m_cubeTemplateTriSet->edgeIndices().size());
    const int triCount = shapeCount * templateTriCount;
    const int edgeCount = shapeCount * templateEdgeCount;
    if (triCount <= 0)
        return;

    m_triAabbsWorld.resize(triCount);
    m_faceNormalsWorld.resize(triCount);
    if (edgeCount > 0)
        m_edgeNormalsWorld.resize(edgeCount);

    view.triangleAabbsWorld = m_triAabbsWorld;
    view.faceNormalsWorld = m_faceNormalsWorld;
    view.edgeNormalsWorld = m_edgeNormalsWorld;

    {
        const int threads = 128;
        const int triBlocks = (triCount + threads - 1) / threads;
        cd_internal::PrepareTriangleWorldDataKernel<decltype(view)><<<triBlocks, threads>>>(view);
        if (edgeCount > 0)
        {
            const int edgeBlocks = (edgeCount + threads - 1) / threads;
            cd_internal::PrepareEdgeNormalsWorldKernel<decltype(view)><<<edgeBlocks, threads>>>(view);
        }
        cudaDeviceSynchronize();
    }

    const int shapePairCount = static_cast<int>(m_shapePairs.size());
    if (shapePairCount <= 0)
        return;

    m_patchPairTriPairCounts.resize(shapePairCount);
    m_patchPairTriPairCounts.reset();

    cd_internal::CountTriPairsPerShapePairKernel<<<(shapePairCount + 127) / 128, 128>>>(
        m_patchPairTriPairCounts,
        m_shapePairs,
        m_shape2TriOffsets);
    cudaDeviceSynchronize();

    const int totalCandidateTriPairs = m_reduce.accumulate(
        m_patchPairTriPairCounts.begin(),
        m_patchPairTriPairCounts.size());
    if (totalCandidateTriPairs <= 0)
        return;

    m_patchPairTriPairOffsets.resize(shapePairCount);
    m_patchPairTriPairOffsets.assign(m_patchPairTriPairCounts);
    m_scan.exclusive(m_patchPairTriPairOffsets, true);

    m_candidateTri0.resize(totalCandidateTriPairs);
    m_candidateTri1.resize(totalCandidateTriPairs);
    m_candidatePatchPairId.resize(totalCandidateTriPairs);

    cd_internal::SetTriPairsFromShapePairsKernel<<<(shapePairCount + 127) / 128, 128>>>(
        m_candidateTri0,
        m_candidateTri1,
        m_candidatePatchPairId,
        m_patchPairTriPairOffsets,
        m_patchPairTriPairCounts,
        m_shapePairs,
        m_shape2TriOffsets);
    cudaDeviceSynchronize();

    m_coarsePassCounts.resize(totalCandidateTriPairs);
    m_coarsePassCounts.reset();

    cd_internal::CountCoarsePassedTriPairsKernel<AABB, Real><<<(totalCandidateTriPairs + 127) / 128, 128>>>(
        m_coarsePassCounts,
        m_candidateTri0,
        m_candidateTri1,
        m_triAabbsWorld,
        m_dHat);
    cudaDeviceSynchronize();

    const int totalFilteredTriPairs = m_reduce.accumulate(
        m_coarsePassCounts.begin(),
        m_coarsePassCounts.size());
    if (totalFilteredTriPairs <= 0)
        return;

    m_coarsePassOffsets.resize(totalCandidateTriPairs);
    m_coarsePassOffsets.assign(m_coarsePassCounts);
    m_scan.exclusive(m_coarsePassOffsets, true);

    m_filteredTri0.resize(totalFilteredTriPairs);
    m_filteredTri1.resize(totalFilteredTriPairs);
    m_filteredPatchPairId.resize(totalFilteredTriPairs);

    cd_internal::SetCoarsePassedTriPairsKernel<<<(totalCandidateTriPairs + 127) / 128, 128>>>(
        m_filteredTri0,
        m_filteredTri1,
        m_filteredPatchPairId,
        m_candidateTri0,
        m_candidateTri1,
        m_candidatePatchPairId,
        m_coarsePassOffsets,
        m_coarsePassCounts);
    cudaDeviceSynchronize();

    const int primitivePassSlotCount = totalFilteredTriPairs * cd_internal::MESH_PASS_COUNT;
    if (primitivePassSlotCount <= 0)
        return;

    m_primitivePassCounts.resize(primitivePassSlotCount);
    m_primitivePassCounts.reset();

    cd_internal::CountPrimitiveCandidatesPerPassKernel<decltype(view)><<<(primitivePassSlotCount + 127) / 128, 128>>>(
        m_primitivePassCounts,
        m_filteredTri0,
        m_filteredTri1,
        m_filteredPatchPairId,
        view);
    cudaDeviceSynchronize();

    const int totalPrimitiveCandidates = m_reduce.accumulate(
        m_primitivePassCounts.begin(),
        m_primitivePassCounts.size());

    m_primitivePassOffsets.resize(primitivePassSlotCount);
    m_primitivePassOffsets.assign(m_primitivePassCounts);
    m_scan.exclusive(m_primitivePassOffsets, true);

    if (totalPrimitiveCandidates <= 0)
        return;

    m_primitiveCandidateContacts.resize(totalPrimitiveCandidates);
    m_primitiveCandidateKeys.resize(totalPrimitiveCandidates);
    m_primitiveCandidateSortedIndices.resize(totalPrimitiveCandidates);
    m_primitiveCandidateKeepFlags.resize(totalPrimitiveCandidates);
    m_primitiveCandidateKeepFlags.reset();

    cd_internal::SetPrimitiveCandidatesPerPassKernel<decltype(view)><<<(primitivePassSlotCount + 127) / 128, 128>>>(
        m_primitiveCandidateContacts,
        m_primitiveCandidateKeys,
        m_primitivePassOffsets,
        m_primitivePassCounts,
        m_filteredTri0,
        m_filteredTri1,
        m_filteredPatchPairId,
        view);
    cudaDeviceSynchronize();

    cd_internal::InitPrimitiveCandidateIndicesKernel<<<(totalPrimitiveCandidates + 127) / 128, 128>>>(
        m_primitiveCandidateSortedIndices);
    cudaDeviceSynchronize();

    thrust::stable_sort_by_key(
        thrust::device,
        m_primitiveCandidateKeys.begin(),
        m_primitiveCandidateKeys.begin() + m_primitiveCandidateKeys.size(),
        m_primitiveCandidateSortedIndices.begin());

    cd_internal::MarkMinDepthCandidatesPerPrimitiveKeyKernel<ContactPair, Real><<<(totalPrimitiveCandidates + 127) / 128, 128>>>(
        m_primitiveCandidateKeepFlags,
        m_primitiveCandidateKeys,
        m_primitiveCandidateSortedIndices,
        m_primitiveCandidateContacts,
        Real(1e-6),
        Real(1e-4));
    cudaDeviceSynchronize();

    const Real crossTypePositionEps = m_edgeEdgeActivationMargin > Real(2e-4)
        ? m_edgeEdgeActivationMargin * Real(0.5)
        : Real(1e-4);
    cd_internal::SuppressRedundantEdgeFaceAgainstVertexFaceKernel<ContactPair, Real><<<(totalFilteredTriPairs + 127) / 128, 128>>>(
        m_primitiveCandidateKeepFlags,
        m_primitivePassCounts,
        m_primitivePassOffsets,
        m_primitiveCandidateContacts,
        crossTypePositionEps,
        Real(1e-4));
    cudaDeviceSynchronize();

    m_selectedPrimitiveCounts.resize(totalFilteredTriPairs);
    m_selectedPrimitiveCounts.reset();
    cd_internal::CountSelectedPrimitiveContactsPerTriPairKernel<<<(totalFilteredTriPairs + 127) / 128, 128>>>(
        m_selectedPrimitiveCounts,
        m_primitivePassCounts,
        m_primitivePassOffsets,
        m_primitiveCandidateKeepFlags);
    cudaDeviceSynchronize();

    m_triPairContactCounts.resize(totalFilteredTriPairs);
    m_triPairContactCounts.reset();
    cd_internal::SetFinalContactCountsKernel<<<(totalFilteredTriPairs + 127) / 128, 128>>>(
        m_triPairContactCounts,
        m_selectedPrimitiveCounts);
    cudaDeviceSynchronize();

    const int totalContacts = m_reduce.accumulate(
        m_triPairContactCounts.begin(),
        m_triPairContactCounts.size());
    if (totalContacts <= 0)
        return;

    m_triPairContactOffsets.resize(totalFilteredTriPairs);
    m_triPairContactOffsets.assign(m_triPairContactCounts);
    m_scan.exclusive(m_triPairContactOffsets, true);

    m_meshContacts.resize(totalContacts);
    cd_internal::SetFinalContactsPerTriPairKernel<ContactPair><<<(totalFilteredTriPairs + 127) / 128, 128>>>(
        m_meshContacts,
        m_triPairContactOffsets,
        m_primitivePassCounts,
        m_primitivePassOffsets,
        m_primitiveCandidateKeepFlags,
        m_primitiveCandidateContacts,
        m_selectedPrimitiveCounts);
    cudaDeviceSynchronize();

    CD_AppendMeshContactsKernel<TDataType><<<(totalContacts + 127) / 128, 128>>>(
        out,
        m_meshContacts,
        rb.batch_bodies,
        rb.friction_mu,
        m_maxBodies,
        num_envs);
    cudaDeviceSynchronize();

#if 0
    if (m_bodyPairs.size() > 0)
    {
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
            m_cubeTemplateTriSet->getPoints(),
            m_cubeTemplateTriSet->triangleIndices(),
            m_dHat);
        cudaDeviceSynchronize();
    }

    detectMeshMeshInternal(bodyPairsHost, rb, out, num_envs);
#endif
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

    if (!broad_phase(rb, num_envs))
        return;

    // std::vector<BodyPair> bodyPairsHost;
    // if (!middle_phase(rb, num_envs, bodyPairsHost))
    //     return;

    narrow_phase(rb, out, num_envs);
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
