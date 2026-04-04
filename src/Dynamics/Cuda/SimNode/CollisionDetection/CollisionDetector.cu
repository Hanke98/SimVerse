#include "CollisionDetector.h"
#include "CubeToMesh.h"

#include "PhysicalField/RigidBody/RigidBody.h"
#include "Collision/CollisionDetectionAlgorithm.h"
#include "Topology/LinearBVH.h"

#include <spdlog/spdlog.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

namespace dyno {

// ============================================================================
// Device helpers
// ============================================================================

// Build an OBox3D from a box body's pose + BoxInfo
__device__ inline TOrientedBox3D<Real> MakeOBox(
    const Vec3f& pos, const Mat3f& rot, const BoxInfo& box)
{
    TOrientedBox3D<Real> obb;
    obb.center = pos + rot * box.center;
    obb.u = Vec3f(rot(0, 0), rot(1, 0), rot(2, 0));
    obb.v = Vec3f(rot(0, 1), rot(1, 1), rot(2, 1));
    obb.w = Vec3f(rot(0, 2), rot(1, 2), rot(2, 2));
    obb.extent = box.halfLength;
    return obb;
}

// Build a Sphere3D from a sphere body's pose + SphereInfo
__device__ inline Sphere3D MakeSphere(
    const Vec3f& pos, const Mat3f& rot, const SphereInfo& info)
{
    Sphere3D s;
    s.center = pos + rot * info.center;
    s.radius = info.radius;
    return s;
}

// Build a Capsule3D from a capsule body's pose + CapsuleInfo
__device__ inline Capsule3D MakeCapsule(
    const Vec3f& pos, const Mat3f& rot, const CapsuleInfo& info)
{
    Capsule3D c;
    c.center = pos + rot * info.center;
    // The capsule axis is along local Y by default
    c.rotation = Quat<Real>(rot);
    c.halfLength = info.halfLength;
    c.radius = info.radius;
    return c;
}

// Build a world-space triangle from template + body transform
__device__ inline Triangle3D MakeWorldTriangle(
    const Vec3f* templateVerts, int v0, int v1, int v2,
    const Vec3f& halfLen, const Vec3f& pos, const Mat3f& rot)
{
    Vec3f p0 = pos + rot * Vec3f(templateVerts[v0][0] * halfLen[0],
                                  templateVerts[v0][1] * halfLen[1],
                                  templateVerts[v0][2] * halfLen[2]);
    Vec3f p1 = pos + rot * Vec3f(templateVerts[v1][0] * halfLen[0],
                                  templateVerts[v1][1] * halfLen[1],
                                  templateVerts[v1][2] * halfLen[2]);
    Vec3f p2 = pos + rot * Vec3f(templateVerts[v2][0] * halfLen[0],
                                  templateVerts[v2][1] * halfLen[1],
                                  templateVerts[v2][2] * halfLen[2]);
    return Triangle3D(p0, p1, p2);
}

// Write one contact to BatchCollisionConstraints (with bounds check)
__device__ inline void WriteContact(
    BatchCollisionConstraints& out,
    int env_id, int bodyA, int bodyB,
    Real depth, const Vec3f& normal, const Vec3f& point, Real mu,
    int maxContacts = 1024)
{
    int idx = atomicAdd(&out.collision_nums[env_id], 1);
    if (idx < maxContacts)
    {
        out.body_idxs(env_id, idx) = Pair<int,int>(bodyA, bodyB);
        out.depth(env_id, idx) = depth;
        out.normal(env_id, idx) = normal;
        out.point(env_id, idx) = point;
        out.mu(env_id, idx) = mu;
    }
    else
    {
        atomicSub(&out.collision_nums[env_id], 1);  // revert if over capacity
    }
}

// ============================================================================
// Narrow phase: primitive-primitive
// ============================================================================
__device__ void NarrowPhasePrimPrim(
    int env_id, int bodyA, int bodyB,
    int typeA, int typeB, int idxA, int idxB,
    const Vec3f& posA, const Mat3f& rotA,
    const Vec3f& posB, const Mat3f& rotB,
    DArray2D<BoxInfo>& boxes,
    DArray2D<SphereInfo>& spheres,
    DArray2D<CapsuleInfo>& capsules,
    Real mu, Real dHat,
    BatchCollisionConstraints& out)
{
    TManifold<Real> manifold;

    // Sphere(0) - Sphere(0)
    if (typeA == 0 && typeB == 0)
    {
        auto sA = MakeSphere(posA, rotA, spheres(env_id, idxA));
        auto sB = MakeSphere(posB, rotB, spheres(env_id, idxB));
        CollisionDetection<Real>::request(manifold, sA, sB, dHat, dHat);
    }
    // Sphere(0) - Capsule(2)
    else if (typeA == 0 && typeB == 2)
    {
        auto sA = MakeSphere(posA, rotA, spheres(env_id, idxA));
        auto cB = MakeCapsule(posB, rotB, capsules(env_id, idxB));
        CollisionDetection<Real>::request(manifold, sA, cB);
    }
    else if (typeA == 2 && typeB == 0)
    {
        auto cA = MakeCapsule(posA, rotA, capsules(env_id, idxA));
        auto sB = MakeSphere(posB, rotB, spheres(env_id, idxB));
        CollisionDetection<Real>::request(manifold, cA, sB);
    }
    // Capsule(2) - Capsule(2)
    else if (typeA == 2 && typeB == 2)
    {
        auto cA = MakeCapsule(posA, rotA, capsules(env_id, idxA));
        auto cB = MakeCapsule(posB, rotB, capsules(env_id, idxB));
        CollisionDetection<Real>::request(manifold, cA, cB);
    }

    for (int c = 0; c < manifold.contactCount; c++)
    {
        WriteContact(out, env_id, bodyA, bodyB,
            -manifold.contacts[c].penetration,
            manifold.normal,
            manifold.contacts[c].position,
            mu);
    }
}

// ============================================================================
// Narrow phase: primitive vs mesh (box treated as mesh)
// ============================================================================
__device__ void NarrowPhasePrimMesh(
    int env_id, int primBody, int meshBody,
    int primType, int primIdx,
    const Vec3f& primPos, const Mat3f& primRot,
    const Vec3f& meshPos, const Mat3f& meshRot,
    DArray2D<SphereInfo>& spheres,
    DArray2D<CapsuleInfo>& capsules,
    DArray2D<BoxInfo>& boxes,
    int meshShapeIdx,
    const Vec3f* templateVerts, const int* templateTriIndices, int numTris,
    Real mu, Real dHat,
    BatchCollisionConstraints& out)
{
    const Vec3f& halfLen = boxes(env_id, meshShapeIdx).halfLength;

    for (int t = 0; t < numTris; t++)
    {
        int v0 = templateTriIndices[t * 3 + 0];
        int v1 = templateTriIndices[t * 3 + 1];
        int v2 = templateTriIndices[t * 3 + 2];
        Triangle3D tri = MakeWorldTriangle(templateVerts, v0, v1, v2,
                                            halfLen, meshPos, meshRot);

        TManifold<Real> manifold;

        if (primType == 0)  // sphere vs triangle
        {
            auto sphere = MakeSphere(primPos, primRot, spheres(env_id, primIdx));
            CollisionDetection<Real>::request(manifold, sphere, tri, dHat, dHat);
        }
        else if (primType == 2)  // capsule vs triangle
        {
            auto cap = MakeCapsule(primPos, primRot, capsules(env_id, primIdx));
            Segment3D seg = cap.centerline();
            Real radius = cap.radius;
            CollisionDetection<Real>::request(manifold, seg, tri, radius + dHat, dHat);
        }

        for (int c = 0; c < manifold.contactCount; c++)
        {
            WriteContact(out, env_id, primBody, meshBody,
                -manifold.contacts[c].penetration,
                manifold.normal,
                manifold.contacts[c].position,
                mu);
        }
    }
}

// ============================================================================
// Narrow phase: mesh vs mesh (both boxes treated as cube meshes)
// ============================================================================
__device__ void NarrowPhaseMeshMesh(
    int env_id, int bodyA, int bodyB,
    int shapeIdxA, int shapeIdxB,
    const Vec3f& posA, const Mat3f& rotA,
    const Vec3f& posB, const Mat3f& rotB,
    DArray2D<BoxInfo>& boxes,
    const Vec3f* templateVerts, const int* templateTriIndices, int numTris,
    Real mu, Real dHat,
    BatchCollisionConstraints& out)
{
    const Vec3f& halfLenA = boxes(env_id, shapeIdxA).halfLength;
    const Vec3f& halfLenB = boxes(env_id, shapeIdxB).halfLength;

    // Test all triangle pairs between the two cube meshes
    for (int tA = 0; tA < numTris; tA++)
    {
        int a0 = templateTriIndices[tA * 3 + 0];
        int a1 = templateTriIndices[tA * 3 + 1];
        int a2 = templateTriIndices[tA * 3 + 2];
        Triangle3D triA = MakeWorldTriangle(templateVerts, a0, a1, a2,
                                             halfLenA, posA, rotA);

        // Compute AABB of triA for coarse filtering
        Vec3f aabbA_min = triA.v[0]; Vec3f aabbA_max = triA.v[0];
        for (int k = 1; k < 3; k++) {
            aabbA_min = Vec3f(min(aabbA_min[0], triA.v[k][0]),
                              min(aabbA_min[1], triA.v[k][1]),
                              min(aabbA_min[2], triA.v[k][2]));
            aabbA_max = Vec3f(max(aabbA_max[0], triA.v[k][0]),
                              max(aabbA_max[1], triA.v[k][1]),
                              max(aabbA_max[2], triA.v[k][2]));
        }
        aabbA_min -= dHat;
        aabbA_max += dHat;

        for (int tB = 0; tB < numTris; tB++)
        {
            int b0 = templateTriIndices[tB * 3 + 0];
            int b1 = templateTriIndices[tB * 3 + 1];
            int b2 = templateTriIndices[tB * 3 + 2];
            Triangle3D triB = MakeWorldTriangle(templateVerts, b0, b1, b2,
                                                 halfLenB, posB, rotB);

            // Coarse AABB overlap check
            Vec3f aabbB_min = triB.v[0]; Vec3f aabbB_max = triB.v[0];
            for (int k = 1; k < 3; k++) {
                aabbB_min = Vec3f(min(aabbB_min[0], triB.v[k][0]),
                                  min(aabbB_min[1], triB.v[k][1]),
                                  min(aabbB_min[2], triB.v[k][2]));
                aabbB_max = Vec3f(max(aabbB_max[0], triB.v[k][0]),
                                  max(aabbB_max[1], triB.v[k][1]),
                                  max(aabbB_max[2], triB.v[k][2]));
            }
            aabbB_min -= dHat;
            aabbB_max += dHat;

            // AABB overlap test
            if (aabbA_max[0] < aabbB_min[0] || aabbB_max[0] < aabbA_min[0] ||
                aabbA_max[1] < aabbB_min[1] || aabbB_max[1] < aabbA_min[1] ||
                aabbA_max[2] < aabbB_min[2] || aabbB_max[2] < aabbA_min[2])
                continue;

            // Triangle-triangle collision detection
            TManifold<Real> manifold;
            CollisionDetection<Real>::request(manifold, triA, triB, dHat, dHat);

            for (int c = 0; c < manifold.contactCount; c++)
            {
                WriteContact(out, env_id, bodyA, bodyB,
                    -manifold.contacts[c].penetration,
                    manifold.normal,
                    manifold.contacts[c].position,
                    mu);
            }
        }
    }
}

// ============================================================================
// Main per-env collision detection kernel
// ============================================================================

// Unit cube template vertices (8 vertices) - stored in constant memory
__constant__ Vec3f c_cubeVerts[8];
// Unit cube template triangle indices (12 triangles * 3 indices = 36 ints)
__constant__ int c_cubeTriIndices[36];
__constant__ int c_numCubeTriangles;

template<typename TDataType>
__global__ void CollisionDetectKernel(
    BatchCollisionConstraints collision_constraints,
    DArray<int> batch_bodies,
    DevArr2D<int> is_static,
    DevArr2D<int> parent_idx,
    DArray2D<Vec3f> batch_pos,
    DArray2D<Mat3f> batch_rot,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    DevArr2D<Real> friction_mu,
    Real dHat,
    int num_envs)
{
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs) return;

    const int num_bodies = batch_bodies[env_id];
    const int numTris = c_numCubeTriangles;

    for (int i = 0; i < num_bodies; i++)
    {
        for (int j = i + 1; j < num_bodies; j++)
        {
            // Filter: skip if both static
            if (is_static(env_id, i) && is_static(env_id, j))
                continue;

            // Filter: skip if joint-connected (direct parent-child)
            if (parent_idx(env_id, i) == j || parent_idx(env_id, j) == i)
                continue;

            int typeI = shape_type(env_id, i);
            int typeJ = shape_type(env_id, j);
            int idxI = shape_idx(env_id, i);
            int idxJ = shape_idx(env_id, j);

            const Vec3f& posI = batch_pos(env_id, i);
            const Mat3f& rotI = batch_rot(env_id, i);
            const Vec3f& posJ = batch_pos(env_id, j);
            const Mat3f& rotJ = batch_rot(env_id, j);

            Real muI = friction_mu(env_id, i);
            Real muJ = friction_mu(env_id, j);
            Real mu = sqrtf(muI * muJ);

            bool isMeshI = (typeI == 1);  // box treated as mesh
            bool isMeshJ = (typeJ == 1);  // box treated as mesh

            if (isMeshI && isMeshJ)
            {
                // Mesh-mesh: triangle-pair collision
                NarrowPhaseMeshMesh(
                    env_id, i, j, idxI, idxJ,
                    posI, rotI, posJ, rotJ,
                    boxes, c_cubeVerts, c_cubeTriIndices, numTris,
                    mu, dHat, collision_constraints);
            }
            else if (!isMeshI && !isMeshJ)
            {
                // Primitive-primitive
                NarrowPhasePrimPrim(
                    env_id, i, j, typeI, typeJ, idxI, idxJ,
                    posI, rotI, posJ, rotJ,
                    boxes, spheres, capsules,
                    mu, dHat, collision_constraints);
            }
            else
            {
                // Primitive-mesh: determine which is which
                int primBody, meshBody, primType, primShapeIdx, meshShapeIdx;
                Vec3f primPos, meshPos;
                Mat3f primRot, meshRot;

                if (isMeshJ)
                {
                    primBody = i; meshBody = j;
                    primType = typeI; primShapeIdx = idxI; meshShapeIdx = idxJ;
                    primPos = posI; primRot = rotI;
                    meshPos = posJ; meshRot = rotJ;
                }
                else
                {
                    primBody = j; meshBody = i;
                    primType = typeJ; primShapeIdx = idxJ; meshShapeIdx = idxI;
                    primPos = posJ; primRot = rotJ;
                    meshPos = posI; meshRot = rotI;
                }

                NarrowPhasePrimMesh(
                    env_id, primBody, meshBody,
                    primType, primShapeIdx,
                    primPos, primRot,
                    meshPos, meshRot,
                    spheres, capsules, boxes,
                    meshShapeIdx,
                    c_cubeVerts, c_cubeTriIndices, numTris,
                    mu, dHat, collision_constraints);
            }
        }
    }
}

// ============================================================================
// Ground collision kernel (extended for all shape types)
// ============================================================================
template<typename TDataType>
__global__ void GroundCollisionKernel(
    BatchCollisionConstraints collision_constraints,
    DArray<int> batch_bodies,
    DevArr2D<int> is_static,
    DArray2D<Vec3f> batch_pos,
    DArray2D<Mat3f> batch_rot,
    DArray2D<int> shape_type,
    DArray2D<int> shape_idx,
    DArray2D<BoxInfo> boxes,
    DArray2D<SphereInfo> spheres,
    DArray2D<CapsuleInfo> capsules,
    int num_envs)
{
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs) return;

    Vec3f ground_normal = Vec3f(0.f, 1.f, 0.f);
    const int num_bodies = batch_bodies[env_id];

    for (int bid = 0; bid < num_bodies; bid++)
    {
        if (is_static(env_id, bid))
            continue;

        const Vec3f& pos = batch_pos(env_id, bid);
        const Mat3f& rot = batch_rot(env_id, bid);
        int type = shape_type(env_id, bid);
        int sidx = shape_idx(env_id, bid);

        if (type == 1)  // Box
        {
            const BoxInfo& box = boxes(env_id, sidx);
            for (int i = 0; i < 8; i++)
            {
                int sx = (i & 1) ? 1 : -1;
                int sy = (i & 2) ? 1 : -1;
                int sz = (i & 4) ? 1 : -1;
                Vec3f vertex = rot * Vec3f(sx * box.halfLength.x, sy * box.halfLength.y, sz * box.halfLength.z) + pos;

                if (vertex.y < 0.f)
                {
                    WriteContact(collision_constraints, env_id, bid, -1,
                        -vertex.y, ground_normal,
                        Vec3f(vertex.x, 0.5f * vertex.y, vertex.z),
                        0.6f);
                }
            }
        }
        else if (type == 0)  // Sphere
        {
            const SphereInfo& sph = spheres(env_id, sidx);
            Vec3f center = pos + rot * sph.center;
            Real penetration = sph.radius - center.y;
            if (penetration > 0.f)
            {
                WriteContact(collision_constraints, env_id, bid, -1,
                    penetration, ground_normal,
                    Vec3f(center.x, center.y - 0.5f * penetration, center.z),
                    0.6f);
            }
        }
        else if (type == 2)  // Capsule
        {
            const CapsuleInfo& cap = capsules(env_id, sidx);
            Vec3f center = pos + rot * cap.center;
            // Capsule axis is along local Y
            Vec3f axis = rot * Vec3f(0.f, cap.halfLength, 0.f);

            Vec3f endpoints[2] = { center - axis, center + axis };
            for (int e = 0; e < 2; e++)
            {
                Real penetration = cap.radius - endpoints[e].y;
                if (penetration > 0.f)
                {
                    WriteContact(collision_constraints, env_id, bid, -1,
                        penetration, ground_normal,
                        Vec3f(endpoints[e].x, endpoints[e].y - 0.5f * penetration, endpoints[e].z),
                        0.6f);
                }
            }
        }
    }
}

// ============================================================================
// MeshCollisionDetector implementation
// ============================================================================

template<typename TDataType>
void MeshCollisionDetector<TDataType>::Initialize(
    int num_envs, int max_bodies, const RigidBody<TDataType>& rb)
{
    m_numEnvs = num_envs;
    m_maxBodies = max_bodies;

    // Generate cube mesh template
    GenerateUnitCubeMesh(m_cubeTemplate);

    // Store host copies of template geometry
    m_cubeVerticesHost.resize(8);
    for (int i = 0; i < 8; i++)
    {
        Real x = (i & 1) ? Real(1) : Real(-1);
        Real y = (i & 2) ? Real(1) : Real(-1);
        Real z = (i & 4) ? Real(1) : Real(-1);
        m_cubeVerticesHost[i] = Coord(x, y, z);
    }

    m_cubeTrianglesHost = {
        // -Z face
        Triangle(0, 3, 1), Triangle(0, 2, 3),
        // +Z face
        Triangle(4, 5, 7), Triangle(4, 7, 6),
        // -Y face
        Triangle(0, 1, 5), Triangle(0, 5, 4),
        // +Y face
        Triangle(2, 7, 3), Triangle(2, 6, 7),
        // -X face
        Triangle(0, 4, 6), Triangle(0, 6, 2),
        // +X face
        Triangle(1, 3, 7), Triangle(1, 7, 5),
    };

    // Upload template to GPU constant memory
    Vec3f vertsHost[8];
    for (int i = 0; i < 8; i++)
        vertsHost[i] = m_cubeVerticesHost[i];
    cudaMemcpyToSymbol(c_cubeVerts, vertsHost, sizeof(Vec3f) * 8);

    int triIndicesHost[36];
    for (int t = 0; t < 12; t++)
    {
        triIndicesHost[t * 3 + 0] = m_cubeTrianglesHost[t][0];
        triIndicesHost[t * 3 + 1] = m_cubeTrianglesHost[t][1];
        triIndicesHost[t * 3 + 2] = m_cubeTrianglesHost[t][2];
    }
    cudaMemcpyToSymbol(c_cubeTriIndices, triIndicesHost, sizeof(int) * 36);

    int numTris = 12;
    cudaMemcpyToSymbol(c_numCubeTriangles, &numTris, sizeof(int));

    m_initialized = true;
    spdlog::info("[MeshCollisionDetector] Initialized with {} envs, {} max bodies, cube template: 8 verts, 12 tris",
                 num_envs, max_bodies);
}

template<typename TDataType>
void MeshCollisionDetector<TDataType>::Detect(
    const RigidBody<TDataType>& rb,
    BatchCollisionConstraints& out,
    int num_envs)
{
    if (!m_initialized)
    {
        spdlog::warn("[MeshCollisionDetector] Not initialized, skipping detection.");
        return;
    }

    // Reset collision counts
    out.collision_nums.reset();

    // Launch body-body collision detection
    const int threads = 128;
    const int blocks = (num_envs + threads - 1) / threads;

    CollisionDetectKernel<TDataType><<<blocks, threads>>>(
        out,
        rb.batch_bodies,
        rb.is_static,
        rb.parent_idx,
        rb.batch_pos,
        rb.batch_rot,
        rb.shape_type,
        rb.shape_idx,
        rb.boxes,
        rb.spheres,
        rb.capsules,
        rb.friction_mu,
        m_dHat,
        num_envs);
    cudaDeviceSynchronize();

    // Launch ground collision detection
    GroundCollisionKernel<TDataType><<<blocks, threads>>>(
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

template<typename TDataType>
void MeshCollisionDetector<TDataType>::resetQueryStaticMappingIfNeeded(
    int shapeCount,
    const std::vector<int>& shape2PatchOffsets,
    const std::vector<uint>& patch2Shape)
{
    if (m_cachedMeshShapeCount == shapeCount)
        return;

    // Reserved for future NeighborMeshLevelQuery integration
    // When real mesh loading is available, this will call:
    //   m_meshNarrowQuery->setStaticShape2PatchOffsets(shape2PatchOffsets);
    //   m_meshNarrowQuery->setStaticPatch2Shape(patch2Shape);

    m_cachedMeshShapeCount = shapeCount;
    spdlog::info("[MeshCollisionDetector] Updated static mappings for {} shapes", shapeCount);
}

// Explicit template instantiation
template class MeshCollisionDetector<DataType3f>;

} // namespace dyno
