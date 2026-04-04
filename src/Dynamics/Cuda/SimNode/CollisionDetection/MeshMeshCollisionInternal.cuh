#pragma once

#include <limits>

#include <Array/ArrayList.h>

#include "MeshCollisionTypes.h"

namespace dyno {
namespace cd_internal {

enum MeshRegionType
{
    MESH_REGION_INVALID = 0,
    MESH_REGION_FACE = 1,
    MESH_REGION_EDGE = 2,
    MESH_REGION_VERTEX = 3,
};

enum MeshPrimitivePassType
{
    MESH_PASS_TRI0_VERTEX = 0,
    MESH_PASS_TRI0_EDGE = 1,
    MESH_PASS_TRI1_VERTEX = 2,
    MESH_PASS_TRI1_EDGE = 3,
    MESH_PASS_COUNT = 4,
};

static constexpr unsigned long long MeshEdgePrimitiveKeyMask = 1ull << 63;

template<typename TDataType>
struct MeshShapeView
{
    using Real = typename TDataType::Real;
    using Coord = typename TDataType::Coord;
    using Matrix = typename TDataType::Matrix;
    using AABB = TAlignedBox3D<Real>;
    using Triangle = TopologyModule::Triangle;
    using Edge = TopologyModule::Edge;
    using Tri2Edg = TopologyModule::Tri2Edg;
    using Edg2Tri = TopologyModule::Edg2Tri;
    using ContactPair = TContactPair<Real>;

    DArray<Coord> templateVertices;
    DArray<Triangle> templateTriangles;
    DArray<Tri2Edg> triangleEdges;
    DArray<Edge> edgeVertices;
    DArray<Edg2Tri> edgeAdjacentFaces;
    DArray<Pair<uint, uint>> shapePairs;

    DArray<int> shape2BodyFlat;
    DArray<Coord> shapeCenters;
    DArray<Matrix> shapeRotations;
    DArray<Coord> shapeHalfLengths;
    DArray<Coord> shapeInvHalfLengths;

    DArray<int> shape2PatchOffsets;
    DArray<int> shape2TriOffsets;
    DArray<int> shape2EdgeOffsets;
    DArray<int> shape2VertexOffsets;
    DArray<int> patch2Shape;
    DArray<int> patch2TriOffsets;
    DArray<int> patch2TriIndices;

    DArray<AABB> templatePatchAabbs;
    DArray<PatchPair> patchPairs;

    DArray<AABB> triangleAabbsWorld;
    DArray<Coord> faceNormalsWorld;
    DArray<Coord> edgeNormalsWorld;

    Real dHat = Real(0);
    Real edgeEdgeActivationMargin = Real(0);
};

template<typename View>
struct TriPairContext
{
    using Real = typename View::Real;

    int tri0 = -1;
    int tri1 = -1;
    int bodyId1 = -1;
    int bodyId2 = -1;
    int tri0Shape = -1;
    int tri1Shape = -1;
    TTriangle3D<Real> triangle0;
    TTriangle3D<Real> triangle1;
};

template<typename Real, typename Coord>
DYN_FUNC inline Coord scalePoint(const Coord& p, const Coord& s)
{
    return Coord(p[0] * s[0], p[1] * s[1], p[2] * s[2]);
}

template<typename Coord>
DYN_FUNC inline Coord stablePerpendicular(const Coord& direction)
{
    using Real = typename Coord::VarType;

    const Real epsSqr = Real(1e-12);
    Coord dir = direction;
    if (dir.normSquared() <= epsSqr)
        return Coord(1, 0, 0);

    dir.normalize();

    const Real ax = dir[0] < Real(0) ? -dir[0] : dir[0];
    const Real ay = dir[1] < Real(0) ? -dir[1] : dir[1];
    const Real az = dir[2] < Real(0) ? -dir[2] : dir[2];

    Coord axis(1, 0, 0);
    if (ay <= ax && ay <= az)
        axis = Coord(0, 1, 0);
    else if (az <= ax && az <= ay)
        axis = Coord(0, 0, 1);

    Coord normal = dir.cross(axis);
    if (normal.normSquared() <= epsSqr)
    {
        axis = axis[0] == Real(1) ? Coord(0, 1, 0) : Coord(1, 0, 0);
        normal = dir.cross(axis);
    }

    if (normal.normSquared() <= epsSqr)
        return Coord(1, 0, 0);

    normal.normalize();
    return normal;
}

template<typename Coord>
DYN_FUNC inline Coord normalizeOrFallback(const Coord& value, const Coord& fallback)
{
    using Real = typename Coord::VarType;
    const Real epsSqr = Real(1e-12);

    Coord out = value;
    if (out.normSquared() > epsSqr)
    {
        out.normalize();
        return out;
    }

    out = fallback;
    if (out.normSquared() > epsSqr)
    {
        out.normalize();
        return out;
    }

    return Coord(1, 0, 0);
}

template<typename Coord>
DYN_FUNC inline Coord buildRobustFaceNormal(const Coord& p0, const Coord& p1, const Coord& p2)
{
    using Real = typename Coord::VarType;

    const Real epsSqr = Real(1e-12);
    Coord normal = (p1 - p0).cross(p2 - p0);
    if (normal.normSquared() > epsSqr)
    {
        normal.normalize();
        return normal;
    }

    Coord longestEdge = p1 - p0;
    Real longestEdgeLen = longestEdge.normSquared();

    Coord edge1 = p2 - p1;
    Real edge1Len = edge1.normSquared();
    if (edge1Len > longestEdgeLen)
    {
        longestEdge = edge1;
        longestEdgeLen = edge1Len;
    }

    Coord edge2 = p0 - p2;
    if (edge2.normSquared() > longestEdgeLen)
        longestEdge = edge2;

    return stablePerpendicular(longestEdge);
}

template<typename Real>
DYN_FUNC inline Real absValue(Real v)
{
    return v < Real(0) ? -v : v;
}

template<typename View>
DYN_FUNC inline bool getWorldVertex(
    const View& view,
    int globalVertexId,
    int shapeId,
    typename View::Coord& p)
{
    if (shapeId < 0 || shapeId >= view.shapeCenters.size()
        || shapeId >= view.shapeRotations.size()
        || shapeId >= view.shapeHalfLengths.size()
        || shapeId + 1 >= view.shape2VertexOffsets.size())
        return false;

    const int begin = view.shape2VertexOffsets[shapeId];
    const int end = view.shape2VertexOffsets[shapeId + 1];
    if (globalVertexId < begin || globalVertexId >= end)
        return false;

    const int localVertexId = globalVertexId - begin;
    if (localVertexId < 0 || localVertexId >= view.templateVertices.size())
        return false;

    const auto& local = view.templateVertices[localVertexId];
    const auto scaled = scalePoint<typename View::Real, typename View::Coord>(local, view.shapeHalfLengths[shapeId]);
    p = view.shapeCenters[shapeId] + view.shapeRotations[shapeId] * scaled;
    return true;
}

template<typename View>
DYN_FUNC inline bool getWorldTriangle(
    const View& view,
    int globalTriId,
    int shapeId,
    typename View::Coord& p0,
    typename View::Coord& p1,
    typename View::Coord& p2)
{
    if (shapeId < 0 || shapeId + 1 >= view.shape2TriOffsets.size())
        return false;

    const int begin = view.shape2TriOffsets[shapeId];
    const int end = view.shape2TriOffsets[shapeId + 1];
    if (globalTriId < begin || globalTriId >= end)
        return false;

    const int localTriId = globalTriId - begin;
    if (localTriId < 0 || localTriId >= view.templateTriangles.size())
        return false;

    const auto tri = view.templateTriangles[localTriId];
    const int vertexBase = view.shape2VertexOffsets[shapeId];
    return getWorldVertex(view, vertexBase + tri[0], shapeId, p0)
        && getWorldVertex(view, vertexBase + tri[1], shapeId, p1)
        && getWorldVertex(view, vertexBase + tri[2], shapeId, p2);
}

template<typename View>
DYN_FUNC inline bool getWorldEdge(
    const View& view,
    int globalEdgeId,
    int shapeId,
    TSegment3D<typename View::Real>& segment)
{
    if (shapeId < 0 || shapeId + 1 >= view.shape2EdgeOffsets.size())
        return false;

    const int begin = view.shape2EdgeOffsets[shapeId];
    const int end = view.shape2EdgeOffsets[shapeId + 1];
    if (globalEdgeId < begin || globalEdgeId >= end)
        return false;

    const int localEdgeId = globalEdgeId - begin;
    if (localEdgeId < 0 || localEdgeId >= view.edgeVertices.size())
        return false;

    const auto edge = view.edgeVertices[localEdgeId];
    typename View::Coord p0;
    typename View::Coord p1;
    const int vertexBase = view.shape2VertexOffsets[shapeId];
    if (!getWorldVertex(view, vertexBase + edge[0], shapeId, p0)
        || !getWorldVertex(view, vertexBase + edge[1], shapeId, p1))
        return false;

    segment = TSegment3D<typename View::Real>(p0, p1);
    return true;
}

template<typename View>
DYN_FUNC inline typename View::Coord transformWorldPointToTargetRest(
    const View& view,
    const typename View::Coord& pWorld,
    int targetShapeId)
{
    const auto local = view.shapeRotations[targetShapeId].transpose() * (pWorld - view.shapeCenters[targetShapeId]);
    return scalePoint<typename View::Real, typename View::Coord>(local, view.shapeInvHalfLengths[targetShapeId]);
}

template<typename View>
DYN_FUNC inline bool buildSourcePatchAabbInTargetRest(
    const View& view,
    int sourcePatchId,
    int targetShapeId,
    typename View::AABB& outAabb)
{
    using Coord = typename View::Coord;
    using Real = typename View::Real;

    if (sourcePatchId < 0 || sourcePatchId + 1 >= view.patch2TriOffsets.size())
        return false;

    const int sourceShapeId = view.patch2Shape[sourcePatchId];
    if (sourceShapeId < 0 || sourceShapeId >= view.shapeCenters.size()
        || targetShapeId < 0 || targetShapeId >= view.shapeCenters.size())
        return false;

    const int triBegin = view.patch2TriOffsets[sourcePatchId];
    const int triEnd = view.patch2TriOffsets[sourcePatchId + 1];
    if (triEnd <= triBegin)
        return false;

    Coord vmin(Real(1e30), Real(1e30), Real(1e30));
    Coord vmax(Real(-1e30), Real(-1e30), Real(-1e30));

    for (int i = triBegin; i < triEnd; ++i)
    {
        if (i < 0 || i >= view.patch2TriIndices.size())
            continue;

        const int globalTriId = view.patch2TriIndices[i];
        Coord p0, p1, p2;
        if (!getWorldTriangle(view, globalTriId, sourceShapeId, p0, p1, p2))
            continue;

        const Coord r0 = transformWorldPointToTargetRest(view, p0, targetShapeId);
        const Coord r1 = transformWorldPointToTargetRest(view, p1, targetShapeId);
        const Coord r2 = transformWorldPointToTargetRest(view, p2, targetShapeId);
        vmin = vmin.minimum(r0).minimum(r1).minimum(r2);
        vmax = vmax.maximum(r0).maximum(r1).maximum(r2);
    }

    outAabb.v0 = vmin;
    outAabb.v1 = vmax;
    return true;
}

template<typename TDataType, typename View>
__global__ void CountPatchPairHitsKernel(
    DArray<int> counts,
    DArray<int> sourcePatchIds,
    DArray<int> sourceTargetShapeIds,
    DArray<typename View::AABB> templatePatchAabbs,
    LinearBVH<TDataType> templatePatchBvh,
    View view)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= counts.size() || sourceId >= sourcePatchIds.size() || sourceId >= sourceTargetShapeIds.size())
        return;

    const int sourcePatchId = sourcePatchIds[sourceId];
    const int targetShapeId = sourceTargetShapeIds[sourceId];
    if (targetShapeId < 0 || targetShapeId + 1 >= view.shape2PatchOffsets.size())
    {
        counts[sourceId] = 0;
        return;
    }

    const int targetPatchCount = view.shape2PatchOffsets[targetShapeId + 1] - view.shape2PatchOffsets[targetShapeId];
    if (targetPatchCount <= 0)
    {
        counts[sourceId] = 0;
        return;
    }

    typename View::AABB sourceAabb;
    if (!buildSourcePatchAabbInTargetRest(view, sourcePatchId, targetShapeId, sourceAabb))
    {
        counts[sourceId] = 0;
        return;
    }

    if (targetPatchCount == 1)
    {
        counts[sourceId] = (templatePatchAabbs.size() > 0 && sourceAabb.checkOverlap(templatePatchAabbs[0])) ? 1 : 0;
        return;
    }

    int hitCount = 0;
    for (int localPatchId = 0; localPatchId < targetPatchCount; ++localPatchId)
    {
        if (localPatchId < templatePatchAabbs.size() && sourceAabb.checkOverlap(templatePatchAabbs[localPatchId]))
            ++hitCount;
    }
    counts[sourceId] = hitCount;
}

template<typename TDataType, typename View>
__global__ void RequestPatchPairHitsKernel(
    DArrayList<int> hitLists,
    DArray<int> sourcePatchIds,
    DArray<int> sourceTargetShapeIds,
    DArray<typename View::AABB> templatePatchAabbs,
    LinearBVH<TDataType> templatePatchBvh,
    View view)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= hitLists.size() || sourceId >= sourcePatchIds.size() || sourceId >= sourceTargetShapeIds.size())
        return;

    auto& list = hitLists[sourceId];
    list.clear();

    const int sourcePatchId = sourcePatchIds[sourceId];
    const int targetShapeId = sourceTargetShapeIds[sourceId];
    if (targetShapeId < 0 || targetShapeId + 1 >= view.shape2PatchOffsets.size())
        return;

    const int targetPatchCount = view.shape2PatchOffsets[targetShapeId + 1] - view.shape2PatchOffsets[targetShapeId];
    if (targetPatchCount <= 0)
        return;

    typename View::AABB sourceAabb;
    if (!buildSourcePatchAabbInTargetRest(view, sourcePatchId, targetShapeId, sourceAabb))
        return;

    if (targetPatchCount == 1)
    {
        if (templatePatchAabbs.size() > 0 && sourceAabb.checkOverlap(templatePatchAabbs[0]))
            list.insert(0);
        return;
    }

    for (int localPatchId = 0; localPatchId < targetPatchCount; ++localPatchId)
    {
        if (localPatchId < templatePatchAabbs.size() && sourceAabb.checkOverlap(templatePatchAabbs[localPatchId]))
            list.insert(localPatchId);
    }
}

template<typename View>
__global__ void SetPatchPairsFromHitListsKernel(
    DArray<PatchPair> patchPairs,
    DArrayList<int> hitLists,
    DArray<int> offsets,
    DArray<int> counts,
    DArray<int> sourcePatchIds,
    DArray<int> sourceTargetShapeIds,
    DArray<int> patch2Shape,
    DArray<int> shape2PatchOffsets,
    DArray<int> shape2BodyFlat,
    int maxBodies)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= hitLists.size() || sourceId >= offsets.size() || sourceId >= counts.size())
        return;

    const int count = counts[sourceId];
    if (count <= 0)
        return;

    const int sourcePatchId = sourcePatchIds[sourceId];
    const int targetShapeId = sourceTargetShapeIds[sourceId];
    const int sourceShapeId = sourcePatchId >= 0 && sourcePatchId < patch2Shape.size() ? patch2Shape[sourcePatchId] : -1;
    if (sourceShapeId < 0 || targetShapeId < 0
        || sourceShapeId >= shape2BodyFlat.size() || targetShapeId >= shape2BodyFlat.size()
        || targetShapeId + 1 >= shape2PatchOffsets.size())
        return;

    const int sourceBodyFlat = shape2BodyFlat[sourceShapeId];
    const int targetBodyFlat = shape2BodyFlat[targetShapeId];
    const int envId = sourceBodyFlat / maxBodies;
    const int bodyA = sourceBodyFlat - envId * maxBodies;
    const int bodyB = targetBodyFlat - envId * maxBodies;
    const int targetPatchBase = shape2PatchOffsets[targetShapeId];

    const int writeBase = offsets[sourceId];
    auto& list = hitLists[sourceId];
    int written = 0;
    for (int i = 0; i < list.size() && written < count; ++i)
    {
        const int localPatchId = list[i];
        const int outIdx = writeBase + written;
        if (outIdx < 0 || outIdx >= patchPairs.size())
            break;

        PatchPair pair;
        pair.env_id = envId;
        pair.body_a = bodyA;
        pair.body_b = bodyB;
        pair.patch_a = sourcePatchId;
        pair.patch_b = targetPatchBase + localPatchId;
        pair.type = MESH_MESH;
        patchPairs[outIdx] = pair;
        ++written;
    }
}

template<typename View>
__global__ void PrepareTriangleWorldDataKernel(View view)
{
    using Coord = typename View::Coord;
    using Real = typename View::Real;

    int triId = threadIdx.x + blockIdx.x * blockDim.x;
    if (triId >= view.triangleAabbsWorld.size() || triId >= view.faceNormalsWorld.size())
        return;

    const int triPerShape = view.templateTriangles.size();
    if (triPerShape <= 0)
        return;

    const int shapeId = triId / triPerShape;
    Coord p0, p1, p2;
    if (!getWorldTriangle(view, triId, shapeId, p0, p1, p2))
    {
        typename View::AABB box;
        box.v0 = Coord(0);
        box.v1 = Coord(0);
        view.triangleAabbsWorld[triId] = box;
        view.faceNormalsWorld[triId] = Coord(1, 0, 0);
        return;
    }

    typename View::AABB box;
    box.v0 = p0.minimum(p1).minimum(p2);
    box.v1 = p0.maximum(p1).maximum(p2);
    view.triangleAabbsWorld[triId] = box;
    view.faceNormalsWorld[triId] = buildRobustFaceNormal(p0, p1, p2);
}

template<typename View>
__global__ void PrepareEdgeNormalsWorldKernel(View view)
{
    using Coord = typename View::Coord;
    using Real = typename View::Real;

    int edgeId = threadIdx.x + blockIdx.x * blockDim.x;
    if (edgeId >= view.edgeNormalsWorld.size())
        return;

    const int edgePerShape = view.edgeVertices.size();
    if (edgePerShape <= 0)
        return;

    const int shapeId = edgeId / edgePerShape;
    const int localEdgeId = edgeId - shapeId * edgePerShape;
    const Real epsSqr = Real(1e-12);

    Coord edgeNormal(0);
    if (localEdgeId >= 0 && localEdgeId < view.edgeAdjacentFaces.size())
    {
        const auto adjacentFaces = view.edgeAdjacentFaces[localEdgeId];
        const int triBase = view.shape2TriOffsets[shapeId];
        const int face0 = adjacentFaces[0] != -1 ? triBase + adjacentFaces[0] : -1;
        const int face1 = adjacentFaces[1] != -1 ? triBase + adjacentFaces[1] : -1;
        if (face0 != -1 && face0 < view.faceNormalsWorld.size()
            && face1 != -1 && face1 < view.faceNormalsWorld.size())
        {
            edgeNormal = view.faceNormalsWorld[face0] + view.faceNormalsWorld[face1];
            if (edgeNormal.normSquared() > epsSqr)
                edgeNormal.normalize();
            else
                edgeNormal = view.faceNormalsWorld[face0];
        }
        else if (face0 != -1 && face0 < view.faceNormalsWorld.size())
        {
            edgeNormal = view.faceNormalsWorld[face0];
        }
        else if (face1 != -1 && face1 < view.faceNormalsWorld.size())
        {
            edgeNormal = view.faceNormalsWorld[face1];
        }
    }

    if (edgeNormal.normSquared() <= epsSqr)
    {
        TSegment3D<Real> edgeSegment;
        if (getWorldEdge(view, edgeId, shapeId, edgeSegment))
            edgeNormal = stablePerpendicular(edgeSegment.direction());
        else
            edgeNormal = Coord(1, 0, 0);
    }

    view.edgeNormalsWorld[edgeId] = normalizeOrFallback(edgeNormal, Coord(1, 0, 0));
}

DYN_FUNC inline int getPairPassSlot(int pairId, int passType)
{
    return pairId * MESH_PASS_COUNT + passType;
}

DYN_FUNC inline unsigned long long encodeVertexPrimitiveKey(int vertexId)
{
    return static_cast<unsigned long long>(vertexId);
}

DYN_FUNC inline unsigned long long encodeEdgePrimitiveKey(int edgeId)
{
    return MeshEdgePrimitiveKeyMask | static_cast<unsigned long long>(edgeId);
}

DYN_FUNC inline bool isEdgePrimitiveKey(unsigned long long key)
{
    return (key & MeshEdgePrimitiveKeyMask) != 0;
}

DYN_FUNC inline bool isPreferredEdgeContactType(
    bool edgePrimitive,
    bool preferEdgeFace,
    ContactType type)
{
    if (!edgePrimitive)
        return true;
    return preferEdgeFace ? type == CT_EDGE_FACE : type == CT_EDGE_EDGE;
}

template<typename Real>
DYN_FUNC inline int localEdgeIdFromBarycentric(Real b0, Real b1, Real b2)
{
    if (b0 <= b1 && b0 <= b2)
        return 1;
    if (b1 <= b0 && b1 <= b2)
        return 2;
    return 0;
}

template<typename Real>
DYN_FUNC inline int localVertexIdFromBarycentric(Real b0, Real b1, Real b2)
{
    if (b0 >= b1 && b0 >= b2)
        return 0;
    if (b1 >= b0 && b1 >= b2)
        return 1;
    return 2;
}

template<typename Real>
DYN_FUNC inline bool classifyTriangleRegion(
    const TTriangle3D<Real>& triangle,
    const typename TTriangle3D<Real>::Coord3D& r,
    Real epsBary,
    int& regionType,
    int& localEdgeId,
    int& localVertexId,
    Real bary[3])
{
    typename TTriangle3D<Real>::Param param;
    if (!triangle.computeBarycentrics(r, param))
        return false;

    bary[0] = param.u;
    bary[1] = param.v;
    bary[2] = param.w;

    int smallCount = 0;
    if (bary[0] <= epsBary) ++smallCount;
    if (bary[1] <= epsBary) ++smallCount;
    if (bary[2] <= epsBary) ++smallCount;

    localEdgeId = -1;
    localVertexId = -1;
    if (smallCount <= 0)
    {
        regionType = MESH_REGION_FACE;
        return true;
    }
    if (smallCount == 1)
    {
        regionType = MESH_REGION_EDGE;
        localEdgeId = localEdgeIdFromBarycentric(bary[0], bary[1], bary[2]);
        return true;
    }

    regionType = MESH_REGION_VERTEX;
    localVertexId = localVertexIdFromBarycentric(bary[0], bary[1], bary[2]);
    return true;
}

template<typename Tri2Edg>
DYN_FUNC inline bool getLocalIncidentEdges(
    const Tri2Edg& triEdges,
    int localVertexId,
    int& edge0,
    int& edge1)
{
    edge0 = -1;
    edge1 = -1;

    switch (localVertexId)
    {
    case 0:
        edge0 = triEdges[0];
        edge1 = triEdges[2];
        return true;
    case 1:
        edge0 = triEdges[0];
        edge1 = triEdges[1];
        return true;
    case 2:
        edge0 = triEdges[1];
        edge1 = triEdges[2];
        return true;
    default:
        return false;
    }
}

template<typename View>
DYN_FUNC inline bool buildEdgeEdgeContact(
    const View& view,
    int sourceEdgeId,
    int sourceShapeId,
    int targetEdgeId,
    int targetShapeId,
    typename View::Coord& contactPoint,
    typename View::Coord& nTarget,
    typename View::Real& depth)
{
    using Real = typename View::Real;
    using Coord = typename View::Coord;

    const Real epsSqr = Real(1e-12);
    TSegment3D<Real> sourceSegment;
    TSegment3D<Real> targetSegment;
    if (!getWorldEdge(view, sourceEdgeId, sourceShapeId, sourceSegment)
        || !getWorldEdge(view, targetEdgeId, targetShapeId, targetSegment))
        return false;

    Coord sourceDir = sourceSegment.direction();
    Coord targetDir = targetSegment.direction();
    if (sourceDir.normSquared() <= epsSqr || targetDir.normSquared() <= epsSqr)
        return false;

    auto pq = sourceSegment.proximity(targetSegment);
    Coord cSource = pq.startPoint();
    Coord cTarget = pq.endPoint();
    Coord pqVec = cTarget - cSource;
    Real gap = pqVec.norm();
    Real activation = view.edgeEdgeActivationMargin + view.dHat;
    if (activation < Real(0))
        activation = Real(0);
    if (gap > activation)
        return false;

    if (targetEdgeId < 0 || targetEdgeId >= view.edgeNormalsWorld.size())
        return false;

    Coord nTargetEdge = view.edgeNormalsWorld[targetEdgeId];
    if (nTargetEdge.normSquared() <= epsSqr)
        return false;
    nTargetEdge.normalize();

    if (nTargetEdge.dot(pqVec) <= Real(0))
        return false;

    sourceDir.normalize();
    targetDir.normalize();
    nTarget = sourceDir.cross(targetDir);
    if (nTarget.normSquared() <= epsSqr)
        return false;
    nTarget.normalize();
    if (nTarget.dot(nTargetEdge) <= Real(0))
        nTarget = -nTarget;

    contactPoint = Real(0.5) * (cSource + cTarget);
    depth = view.dHat - gap;
    if (depth < Real(0))
        depth = Real(0);
    return true;
}

template<typename View>
DYN_FUNC inline bool tryVertexTriangleContact(
    const View& view,
    int sourceShapeId,
    int sourceVertexId,
    int targetTriId,
    const TTriangle3D<typename View::Real>& targetTriangle,
    typename View::Coord& contactPoint,
    typename View::Coord& nTarget,
    typename View::Real& depth,
    ContactType& contactType)
{
    using Real = typename View::Real;
    using Coord = typename View::Coord;

    const Real epsBary = Real(1e-5);
    Coord p;
    if (!getWorldVertex(view, sourceVertexId, sourceShapeId, p))
        return false;

    Coord r = TPoint3D<Real>(p).project(targetTriangle).origin;
    int regionType = MESH_REGION_INVALID;
    int localEdgeId = -1;
    int localVertexId = -1;
    Real bary[3] = { Real(0), Real(0), Real(0) };
    if (!classifyTriangleRegion(targetTriangle, r, epsBary, regionType, localEdgeId, localVertexId, bary))
        return false;

    if (regionType != MESH_REGION_FACE)
        return false;

    Coord faceNormal = targetTriId >= 0 && targetTriId < view.faceNormalsWorld.size()
        ? view.faceNormalsWorld[targetTriId]
        : buildRobustFaceNormal(targetTriangle.v[0], targetTriangle.v[1], targetTriangle.v[2]);
    nTarget = normalizeOrFallback(faceNormal, stablePerpendicular(targetTriangle.v[1] - targetTriangle.v[0]));

    Real signedDistance = (p - targetTriangle.v[0]).dot(nTarget);
    if (signedDistance > view.dHat || signedDistance < Real(-0.5))
        return false;

    contactPoint = r;
    depth = signedDistance < Real(0) ? -signedDistance : Real(0);
    contactType = CT_VERTEX_FACE;
    return true;
}

template<typename View>
DYN_FUNC inline bool tryEdgeTriangleContact(
    const View& view,
    int sourceShapeId,
    int sourceEdgeId,
    int targetTriId,
    int targetShapeId,
    const TTriangle3D<typename View::Real>& targetTriangle,
    typename View::Coord& contactPoint,
    typename View::Coord& nTarget,
    typename View::Real& depth,
    ContactType& contactType)
{
    using Real = typename View::Real;
    using Coord = typename View::Coord;

    const Real epsBary = Real(1e-5);
    TSegment3D<Real> sourceSegment;
    if (!getWorldEdge(view, sourceEdgeId, sourceShapeId, sourceSegment))
        return false;

    auto pq = sourceSegment.proximity(targetTriangle);
    Coord cTarget = pq.endPoint();
    int regionType = MESH_REGION_INVALID;
    int localEdgeId = -1;
    int localVertexId = -1;
    Real bary[3] = { Real(0), Real(0), Real(0) };
    if (!classifyTriangleRegion(targetTriangle, cTarget, epsBary, regionType, localEdgeId, localVertexId, bary))
        return false;

    const int targetLocalTriId = targetShapeId >= 0 && targetShapeId + 1 < view.shape2TriOffsets.size()
        ? targetTriId - view.shape2TriOffsets[targetShapeId]
        : -1;
    if (targetLocalTriId < 0 || targetLocalTriId >= view.triangleEdges.size())
        return false;

    if (regionType == MESH_REGION_FACE)
    {
        Coord faceNormal = targetTriId >= 0 && targetTriId < view.faceNormalsWorld.size()
            ? view.faceNormalsWorld[targetTriId]
            : buildRobustFaceNormal(targetTriangle.v[0], targetTriangle.v[1], targetTriangle.v[2]);
        nTarget = normalizeOrFallback(faceNormal, stablePerpendicular(targetTriangle.v[1] - targetTriangle.v[0]));

        Coord p0 = sourceSegment.startPoint();
        Coord p1 = sourceSegment.endPoint();
        Real d0 = (p0 - targetTriangle.v[0]).dot(nTarget);
        Real d1 = (p1 - targetTriangle.v[0]).dot(nTarget);
        Real minSignedDistance = d0 < d1 ? d0 : d1;
        Real edgeActivation = view.edgeEdgeActivationMargin + view.dHat;
        if (edgeActivation < Real(0))
            edgeActivation = Real(0);
        if (minSignedDistance > edgeActivation)
            return false;

        contactPoint = cTarget;
        depth = Real(0);
        contactType = CT_EDGE_FACE;
        return true;
    }

    if (regionType == MESH_REGION_EDGE)
    {
        const int targetEdgeId = view.shape2EdgeOffsets[targetShapeId] + view.triangleEdges[targetLocalTriId][localEdgeId];
        if (!buildEdgeEdgeContact(view, sourceEdgeId, sourceShapeId, targetEdgeId, targetShapeId, contactPoint, nTarget, depth))
            return false;
        contactType = CT_EDGE_EDGE;
        return true;
    }

    if (regionType == MESH_REGION_VERTEX)
    {
        int edge0 = -1;
        int edge1 = -1;
        if (!getLocalIncidentEdges(view.triangleEdges[targetLocalTriId], localVertexId, edge0, edge1))
            return false;

        int bestTargetEdge = -1;
        Real bestDist2 = std::numeric_limits<Real>::max();
        Real bestAlign = Real(-1);

        if (edge0 >= 0)
        {
            const int globalTargetEdgeId = view.shape2EdgeOffsets[targetShapeId] + edge0;
            TSegment3D<Real> targetSegment;
            if (getWorldEdge(view, globalTargetEdgeId, targetShapeId, targetSegment))
            {
                Coord tS = sourceSegment.direction();
                Coord tT = targetSegment.direction();
                if (tS.normSquared() > Real(1e-12) && tT.normSquared() > Real(1e-12))
                {
                    tS.normalize();
                    tT.normalize();

                    auto edgePair = sourceSegment.proximity(targetSegment);
                    Real dist2 = edgePair.lengthSquared();
                    Real align = absValue(tS.dot(tT));
                    if (bestTargetEdge < 0 || dist2 < bestDist2 || (absValue(dist2 - bestDist2) <= Real(1e-9) && align > bestAlign))
                    {
                        bestTargetEdge = globalTargetEdgeId;
                        bestDist2 = dist2;
                        bestAlign = align;
                    }
                }
            }
        }
        if (edge1 >= 0)
        {
            const int globalTargetEdgeId = view.shape2EdgeOffsets[targetShapeId] + edge1;
            TSegment3D<Real> targetSegment;
            if (getWorldEdge(view, globalTargetEdgeId, targetShapeId, targetSegment))
            {
                Coord tS = sourceSegment.direction();
                Coord tT = targetSegment.direction();
                if (tS.normSquared() > Real(1e-12) && tT.normSquared() > Real(1e-12))
                {
                    tS.normalize();
                    tT.normalize();

                    auto edgePair = sourceSegment.proximity(targetSegment);
                    Real dist2 = edgePair.lengthSquared();
                    Real align = absValue(tS.dot(tT));
                    if (bestTargetEdge < 0 || dist2 < bestDist2 || (absValue(dist2 - bestDist2) <= Real(1e-9) && align > bestAlign))
                    {
                        bestTargetEdge = globalTargetEdgeId;
                        bestDist2 = dist2;
                        bestAlign = align;
                    }
                }
            }
        }
        if (bestTargetEdge < 0)
            return false;

        if (!buildEdgeEdgeContact(view, sourceEdgeId, sourceShapeId, bestTargetEdge, targetShapeId, contactPoint, nTarget, depth))
            return false;
        contactType = CT_EDGE_EDGE;
        return true;
    }

    return false;
}

template<typename ContactPair, typename Coord, typename Real>
DYN_FUNC inline void writeContact(
    ContactPair& contact,
    int bodyId1,
    int bodyId2,
    int tri0,
    int tri1,
    const Coord& contactPoint,
    const Coord& nTarget,
    bool targetIsTri1,
    Real depth,
    ContactType type)
{
    contact.bodyId1 = bodyId1;
    contact.bodyId2 = bodyId2;
    contact.localId1 = tri0;
    contact.localId2 = tri1;
    contact.pos1 = contactPoint;
    contact.pos2 = contactPoint;
    if (targetIsTri1)
    {
        contact.normal1 = nTarget;
        contact.normal2 = -nTarget;
    }
    else
    {
        contact.normal1 = -nTarget;
        contact.normal2 = nTarget;
    }
    contact.contactType = type;
    contact.interpenetration = depth < Real(0) ? Real(0) : depth;
}

template<typename View>
DYN_FUNC inline bool buildTriPairContext(
    const View& view,
    int tri0,
    int tri1,
    int pairId,
    TriPairContext<View>& ctx)
{
    if (pairId >= 0 && pairId < view.patchPairs.size())
    {
        const PatchPair pair = view.patchPairs[pairId];
        if (pair.patch_a < 0 || pair.patch_b < 0
            || pair.patch_a >= view.patch2Shape.size() || pair.patch_b >= view.patch2Shape.size())
            return false;

        ctx.tri0Shape = view.patch2Shape[pair.patch_a];
        ctx.tri1Shape = view.patch2Shape[pair.patch_b];
    }
    else if (pairId >= 0 && pairId < view.shapePairs.size())
    {
        const auto pair = view.shapePairs[pairId];
        ctx.tri0Shape = static_cast<int>(pair.first);
        ctx.tri1Shape = static_cast<int>(pair.second);
    }
    else
    {
        return false;
    }

    if (ctx.tri0Shape < 0 || ctx.tri1Shape < 0
        || ctx.tri0Shape >= view.shape2BodyFlat.size() || ctx.tri1Shape >= view.shape2BodyFlat.size())
        return false;

    ctx.bodyId1 = view.shape2BodyFlat[ctx.tri0Shape];
    ctx.bodyId2 = view.shape2BodyFlat[ctx.tri1Shape];

    typename View::Coord p00, p01, p02;
    typename View::Coord p10, p11, p12;
    if (!getWorldTriangle(view, tri0, ctx.tri0Shape, p00, p01, p02)
        || !getWorldTriangle(view, tri1, ctx.tri1Shape, p10, p11, p12))
        return false;

    ctx.tri0 = tri0;
    ctx.tri1 = tri1;
    ctx.triangle0 = TTriangle3D<typename View::Real>(p00, p01, p02);
    ctx.triangle1 = TTriangle3D<typename View::Real>(p10, p11, p12);
    return true;
}

__global__ void CountTriPairsPerShapePairKernel(
    DArray<int> counts,
    DArray<Pair<uint, uint>> shapePairs,
    DArray<int> shape2TriOffsets)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= counts.size() || pairId >= shapePairs.size())
        return;

    const auto pair = shapePairs[pairId];
    const int shape0 = static_cast<int>(pair.first);
    const int shape1 = static_cast<int>(pair.second);
    if (shape0 < 0 || shape1 < 0
        || shape0 + 1 >= shape2TriOffsets.size()
        || shape1 + 1 >= shape2TriOffsets.size())
    {
        counts[pairId] = 0;
        return;
    }

    int count0 = shape2TriOffsets[shape0 + 1] - shape2TriOffsets[shape0];
    int count1 = shape2TriOffsets[shape1 + 1] - shape2TriOffsets[shape1];
    count0 = count0 > 0 ? count0 : 0;
    count1 = count1 > 0 ? count1 : 0;
    counts[pairId] = count0 * count1;
}

__global__ void SetTriPairsFromShapePairsKernel(
    DArray<int> tri0Out,
    DArray<int> tri1Out,
    DArray<int> pairIdOut,
    DArray<int> offsets,
    DArray<int> counts,
    DArray<Pair<uint, uint>> shapePairs,
    DArray<int> shape2TriOffsets)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= shapePairs.size() || pairId >= offsets.size() || pairId >= counts.size())
        return;

    const int count = counts[pairId];
    if (count <= 0)
        return;

    const auto pair = shapePairs[pairId];
    const int shape0 = static_cast<int>(pair.first);
    const int shape1 = static_cast<int>(pair.second);
    if (shape0 < 0 || shape1 < 0
        || shape0 + 1 >= shape2TriOffsets.size()
        || shape1 + 1 >= shape2TriOffsets.size())
        return;

    const int begin0 = shape2TriOffsets[shape0];
    const int end0 = shape2TriOffsets[shape0 + 1];
    const int begin1 = shape2TriOffsets[shape1];
    const int end1 = shape2TriOffsets[shape1 + 1];
    const int count0 = end0 - begin0;
    const int count1 = end1 - begin1;
    if (count0 <= 0 || count1 <= 0)
        return;

    const int base = offsets[pairId];
    for (int i = 0; i < count0; ++i)
    {
        const int tri0 = begin0 + i;
        for (int j = 0; j < count1; ++j)
        {
            const int outIdx = base + i * count1 + j;
            if (outIdx >= tri0Out.size() || outIdx >= tri1Out.size() || outIdx >= pairIdOut.size())
                return;

            tri0Out[outIdx] = tri0;
            tri1Out[outIdx] = begin1 + j;
            pairIdOut[outIdx] = pairId;
        }
    }
}

template<typename View>
DYN_FUNC inline bool getPrimitivePassContext(
    const TriPairContext<View>& ctx,
    int passType,
    int& sourceTriId,
    int& sourceShapeId,
    const TTriangle3D<typename View::Real>*& sourceTriangle,
    int& targetTriId,
    int& targetShapeId,
    const TTriangle3D<typename View::Real>*& targetTriangle,
    bool& targetIsTri1,
    bool& vertexPass)
{
    switch (passType)
    {
    case MESH_PASS_TRI0_VERTEX:
        sourceTriId = ctx.tri0;
        sourceShapeId = ctx.tri0Shape;
        sourceTriangle = &ctx.triangle0;
        targetTriId = ctx.tri1;
        targetShapeId = ctx.tri1Shape;
        targetTriangle = &ctx.triangle1;
        targetIsTri1 = true;
        vertexPass = true;
        return true;
    case MESH_PASS_TRI0_EDGE:
        sourceTriId = ctx.tri0;
        sourceShapeId = ctx.tri0Shape;
        sourceTriangle = &ctx.triangle0;
        targetTriId = ctx.tri1;
        targetShapeId = ctx.tri1Shape;
        targetTriangle = &ctx.triangle1;
        targetIsTri1 = true;
        vertexPass = false;
        return true;
    case MESH_PASS_TRI1_VERTEX:
        sourceTriId = ctx.tri1;
        sourceShapeId = ctx.tri1Shape;
        sourceTriangle = &ctx.triangle1;
        targetTriId = ctx.tri0;
        targetShapeId = ctx.tri0Shape;
        targetTriangle = &ctx.triangle0;
        targetIsTri1 = false;
        vertexPass = true;
        return true;
    case MESH_PASS_TRI1_EDGE:
        sourceTriId = ctx.tri1;
        sourceShapeId = ctx.tri1Shape;
        sourceTriangle = &ctx.triangle1;
        targetTriId = ctx.tri0;
        targetShapeId = ctx.tri0Shape;
        targetTriangle = &ctx.triangle0;
        targetIsTri1 = false;
        vertexPass = false;
        return true;
    default:
        break;
    }

    return false;
}

template<typename View>
DYN_FUNC inline int processPrimitivePass(
    const View& view,
    const TriPairContext<View>& ctx,
    int passType,
    typename View::ContactPair* contacts,
    unsigned long long* primitiveKeys,
    int contactsSize,
    int writeBase,
    bool write)
{
    int sourceTriId = -1;
    int sourceShapeId = -1;
    int targetTriId = -1;
    int targetShapeId = -1;
    const TTriangle3D<typename View::Real>* sourceTriangle = nullptr;
    const TTriangle3D<typename View::Real>* targetTriangle = nullptr;
    bool targetIsTri1 = true;
    bool vertexPass = true;
    if (!getPrimitivePassContext(
            ctx,
            passType,
            sourceTriId,
            sourceShapeId,
            sourceTriangle,
            targetTriId,
            targetShapeId,
            targetTriangle,
            targetIsTri1,
            vertexPass))
        return 0;

    int count = 0;
    const int localSourceTriId = sourceTriId - view.shape2TriOffsets[sourceShapeId];
    if (localSourceTriId < 0 || localSourceTriId >= view.templateTriangles.size())
        return 0;

    if (vertexPass)
    {
        const auto sourceTriIndices = view.templateTriangles[localSourceTriId];
        const int vertexBase = view.shape2VertexOffsets[sourceShapeId];
        for (int localVertexId = 0; localVertexId < 3; ++localVertexId)
        {
            const int globalVertexId = vertexBase + sourceTriIndices[localVertexId];
            typename View::Coord contactPoint;
            typename View::Coord nTarget;
            typename View::Real depth = typename View::Real(0);
            ContactType type = CT_UNKNOWN;
            if (!tryVertexTriangleContact(
                    view,
                    sourceShapeId,
                    globalVertexId,
                    targetTriId,
                    *targetTriangle,
                    contactPoint,
                    nTarget,
                    depth,
                    type))
                continue;

            if (write && contacts != nullptr && primitiveKeys != nullptr)
            {
                const int outIdx = writeBase + count;
                if (outIdx >= 0 && outIdx < contactsSize)
                {
                    typename View::ContactPair cp;
                    writeContact(cp, ctx.bodyId1, ctx.bodyId2, ctx.tri0, ctx.tri1, contactPoint, nTarget, targetIsTri1, depth, type);
                    contacts[outIdx] = cp;
                    primitiveKeys[outIdx] = encodeVertexPrimitiveKey(globalVertexId);
                }
            }

            ++count;
        }
        return count;
    }

    const int edgeBase = view.shape2EdgeOffsets[sourceShapeId];
    const auto sourceTriEdges = view.triangleEdges[localSourceTriId];
    for (int localEdgeId = 0; localEdgeId < 3; ++localEdgeId)
    {
        const int localTemplateEdgeId = sourceTriEdges[localEdgeId];
        if (localTemplateEdgeId < 0)
            continue;

        const int globalEdgeId = edgeBase + localTemplateEdgeId;
        typename View::Coord contactPoint;
        typename View::Coord nTarget;
        typename View::Real depth = typename View::Real(0);
        ContactType type = CT_UNKNOWN;
        if (!tryEdgeTriangleContact(
                view,
                sourceShapeId,
                globalEdgeId,
                targetTriId,
                targetShapeId,
                *targetTriangle,
                contactPoint,
                nTarget,
                depth,
                type))
            continue;

        if (write && contacts != nullptr && primitiveKeys != nullptr)
        {
            const int outIdx = writeBase + count;
            if (outIdx >= 0 && outIdx < contactsSize)
            {
                typename View::ContactPair cp;
                writeContact(cp, ctx.bodyId1, ctx.bodyId2, ctx.tri0, ctx.tri1, contactPoint, nTarget, targetIsTri1, depth, type);
                contacts[outIdx] = cp;
                primitiveKeys[outIdx] = encodeEdgePrimitiveKey(globalEdgeId);
            }
        }

        ++count;
    }

    return count;
}

__global__ void CountTriPairsPerPatchPairKernel(
    DArray<int> counts,
    DArray<PatchPair> patchPairs,
    DArray<int> patch2TriOffsets)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= counts.size() || pairId >= patchPairs.size())
        return;

    const PatchPair pair = patchPairs[pairId];
    if (pair.patch_a < 0 || pair.patch_b < 0
        || pair.patch_a + 1 >= patch2TriOffsets.size()
        || pair.patch_b + 1 >= patch2TriOffsets.size())
    {
        counts[pairId] = 0;
        return;
    }

    int count0 = patch2TriOffsets[pair.patch_a + 1] - patch2TriOffsets[pair.patch_a];
    int count1 = patch2TriOffsets[pair.patch_b + 1] - patch2TriOffsets[pair.patch_b];
    count0 = count0 > 0 ? count0 : 0;
    count1 = count1 > 0 ? count1 : 0;
    counts[pairId] = count0 * count1;
}

__global__ void SetTriPairsKernel(
    DArray<int> tri0Out,
    DArray<int> tri1Out,
    DArray<int> patchPairIdOut,
    DArray<int> offsets,
    DArray<int> counts,
    DArray<PatchPair> patchPairs,
    DArray<int> patch2TriOffsets,
    DArray<int> patch2TriIndices)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= patchPairs.size() || pairId >= offsets.size() || pairId >= counts.size())
        return;

    const int count = counts[pairId];
    if (count <= 0)
        return;

    const PatchPair pair = patchPairs[pairId];
    if (pair.patch_a < 0 || pair.patch_b < 0
        || pair.patch_a + 1 >= patch2TriOffsets.size()
        || pair.patch_b + 1 >= patch2TriOffsets.size())
        return;

    const int begin0 = patch2TriOffsets[pair.patch_a];
    const int end0 = patch2TriOffsets[pair.patch_a + 1];
    const int begin1 = patch2TriOffsets[pair.patch_b];
    const int end1 = patch2TriOffsets[pair.patch_b + 1];
    const int count0 = end0 - begin0;
    const int count1 = end1 - begin1;
    if (count0 <= 0 || count1 <= 0)
        return;

    const int base = offsets[pairId];
    for (int i = 0; i < count0; ++i)
    {
        const int tri0 = patch2TriIndices[begin0 + i];
        for (int j = 0; j < count1; ++j)
        {
            const int outIdx = base + i * count1 + j;
            if (outIdx >= tri0Out.size() || outIdx >= tri1Out.size() || outIdx >= patchPairIdOut.size())
                return;

            tri0Out[outIdx] = tri0;
            tri1Out[outIdx] = patch2TriIndices[begin1 + j];
            patchPairIdOut[outIdx] = pairId;
        }
    }
}

template<typename AABB, typename Real>
__global__ void CountCoarsePassedTriPairsKernel(
    DArray<int> counts,
    DArray<int> tri0,
    DArray<int> tri1,
    DArray<AABB> triangleAabbs,
    Real dHat)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= counts.size() || pairId >= tri0.size() || pairId >= tri1.size())
        return;

    const int t0 = tri0[pairId];
    const int t1 = tri1[pairId];
    if (t0 < 0 || t1 < 0 || t0 >= triangleAabbs.size() || t1 >= triangleAabbs.size())
    {
        counts[pairId] = 0;
        return;
    }

    AABB box0 = triangleAabbs[t0];
    AABB box1 = triangleAabbs[t1];
    const auto expandVec = typename AABB::Coord3D(dHat, dHat, dHat);
    box0.v0 -= expandVec;
    box0.v1 += expandVec;
    box1.v0 -= expandVec;
    box1.v1 += expandVec;
    counts[pairId] = box0.checkOverlap(box1) ? 1 : 0;
}

__global__ void SetCoarsePassedTriPairsKernel(
    DArray<int> filteredTri0,
    DArray<int> filteredTri1,
    DArray<int> filteredPatchPairId,
    DArray<int> tri0,
    DArray<int> tri1,
    DArray<int> patchPairId,
    DArray<int> offsets,
    DArray<int> counts)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= counts.size() || pairId >= offsets.size())
        return;

    if (counts[pairId] <= 0)
        return;

    const int outIdx = offsets[pairId];
    if (outIdx < 0 || outIdx >= filteredTri0.size() || outIdx >= filteredTri1.size() || outIdx >= filteredPatchPairId.size())
        return;

    filteredTri0[outIdx] = tri0[pairId];
    filteredTri1[outIdx] = tri1[pairId];
    filteredPatchPairId[outIdx] = patchPairId[pairId];
}

template<typename View>
__global__ void CountPrimitiveCandidatesPerPassKernel(
    DArray<int> primitivePassCounts,
    DArray<int> filteredTri0,
    DArray<int> filteredTri1,
    DArray<int> filteredPatchPairId,
    View view)
{
    int slotId = threadIdx.x + blockIdx.x * blockDim.x;
    if (slotId >= primitivePassCounts.size())
        return;

    const int pairId = slotId / MESH_PASS_COUNT;
    const int passType = slotId % MESH_PASS_COUNT;
    if (pairId >= filteredTri0.size() || pairId >= filteredTri1.size() || pairId >= filteredPatchPairId.size())
    {
        primitivePassCounts[slotId] = 0;
        return;
    }

    TriPairContext<View> ctx;
    if (!buildTriPairContext(view, filteredTri0[pairId], filteredTri1[pairId], filteredPatchPairId[pairId], ctx))
    {
        primitivePassCounts[slotId] = 0;
        return;
    }

    primitivePassCounts[slotId] = processPrimitivePass(
        view,
        ctx,
        passType,
        static_cast<typename View::ContactPair*>(nullptr),
        nullptr,
        0,
        0,
        false);
}

template<typename View>
__global__ void SetPrimitiveCandidatesPerPassKernel(
    DArray<typename View::ContactPair> primitiveCandidateContacts,
    DArray<unsigned long long> primitiveCandidateKeys,
    DArray<int> primitivePassOffsets,
    DArray<int> primitivePassCounts,
    DArray<int> filteredTri0,
    DArray<int> filteredTri1,
    DArray<int> filteredPatchPairId,
    View view)
{
    int slotId = threadIdx.x + blockIdx.x * blockDim.x;
    if (slotId >= primitivePassOffsets.size() || slotId >= primitivePassCounts.size())
        return;

    const int count = primitivePassCounts[slotId];
    if (count <= 0)
        return;

    const int pairId = slotId / MESH_PASS_COUNT;
    const int passType = slotId % MESH_PASS_COUNT;
    if (pairId >= filteredTri0.size() || pairId >= filteredTri1.size() || pairId >= filteredPatchPairId.size())
        return;

    TriPairContext<View> ctx;
    if (!buildTriPairContext(view, filteredTri0[pairId], filteredTri1[pairId], filteredPatchPairId[pairId], ctx))
        return;

    processPrimitivePass(
        view,
        ctx,
        passType,
        primitiveCandidateContacts.begin(),
        primitiveCandidateKeys.begin(),
        primitiveCandidateContacts.size(),
        primitivePassOffsets[slotId],
        true);
}

__global__ void InitPrimitiveCandidateIndicesKernel(DArray<int> primitiveCandidateSortedIndices)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= primitiveCandidateSortedIndices.size())
        return;
    primitiveCandidateSortedIndices[idx] = idx;
}

template<typename ContactPair, typename Real>
__global__ void MarkMinDepthCandidatesPerPrimitiveKeyKernel(
    DArray<int> primitiveCandidateKeepFlags,
    DArray<unsigned long long> primitiveCandidateKeys,
    DArray<int> primitiveCandidateSortedIndices,
    DArray<ContactPair> primitiveCandidateContacts,
    Real depthTieEps,
    Real sameDirectionDotEps)
{
    int sortedIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (sortedIdx >= primitiveCandidateKeys.size() || sortedIdx >= primitiveCandidateSortedIndices.size())
        return;

    if (sortedIdx > 0 && primitiveCandidateKeys[sortedIdx - 1] == primitiveCandidateKeys[sortedIdx])
        return;

    const unsigned long long key = primitiveCandidateKeys[sortedIdx];
    int groupEnd = sortedIdx;
    const bool edgePrimitive = isEdgePrimitiveKey(key);
    bool preferEdgeFace = false;
    if (edgePrimitive)
    {
        for (int i = sortedIdx; i < primitiveCandidateKeys.size() && primitiveCandidateKeys[i] == key; ++i)
        {
            const int rawIdx = primitiveCandidateSortedIndices[i];
            if (rawIdx >= 0 && rawIdx < primitiveCandidateContacts.size()
                && primitiveCandidateContacts[rawIdx].contactType == CT_EDGE_FACE)
            {
                preferEdgeFace = true;
                break;
            }
        }
    }

    Real minDepth = std::numeric_limits<Real>::max();
    while (groupEnd < primitiveCandidateKeys.size() && primitiveCandidateKeys[groupEnd] == key)
    {
        const int rawIdx = primitiveCandidateSortedIndices[groupEnd];
        if (rawIdx >= 0 && rawIdx < primitiveCandidateContacts.size())
        {
            const ContactType type = primitiveCandidateContacts[rawIdx].contactType;
            if (isPreferredEdgeContactType(edgePrimitive, preferEdgeFace, type))
            {
                const Real depth = primitiveCandidateContacts[rawIdx].interpenetration;
                if (depth < minDepth)
                    minDepth = depth;
            }
        }
        ++groupEnd;
    }

    if (minDepth == std::numeric_limits<Real>::max())
        return;

    for (int i = sortedIdx; i < groupEnd; ++i)
    {
        const int rawIdx = primitiveCandidateSortedIndices[i];
        if (rawIdx < 0 || rawIdx >= primitiveCandidateKeepFlags.size() || rawIdx >= primitiveCandidateContacts.size())
            continue;

        const ContactType type = primitiveCandidateContacts[rawIdx].contactType;
        if (!isPreferredEdgeContactType(edgePrimitive, preferEdgeFace, type))
            continue;

        const Real depth = primitiveCandidateContacts[rawIdx].interpenetration;
        if (depth > minDepth + depthTieEps)
            continue;

        auto direction = primitiveCandidateContacts[rawIdx].normal1;
        const Real dirNorm2 = direction.normSquared();
        if (dirNorm2 > Real(1e-12))
        {
            direction /= sqrt(dirNorm2);
            bool duplicateDirection = false;
            for (int j = sortedIdx; j < i; ++j)
            {
                const int prevRawIdx = primitiveCandidateSortedIndices[j];
                if (prevRawIdx < 0 || prevRawIdx >= primitiveCandidateKeepFlags.size()
                    || prevRawIdx >= primitiveCandidateContacts.size()
                    || primitiveCandidateKeepFlags[prevRawIdx] <= 0)
                    continue;

                const ContactType prevType = primitiveCandidateContacts[prevRawIdx].contactType;
                if (!isPreferredEdgeContactType(edgePrimitive, preferEdgeFace, prevType))
                    continue;

                const Real prevDepth = primitiveCandidateContacts[prevRawIdx].interpenetration;
                if (prevDepth > minDepth + depthTieEps)
                    continue;

                auto prevDirection = primitiveCandidateContacts[prevRawIdx].normal1;
                const Real prevNorm2 = prevDirection.normSquared();
                if (prevNorm2 <= Real(1e-12))
                    continue;

                prevDirection /= sqrt(prevNorm2);
                if (direction.dot(prevDirection) >= Real(1) - sameDirectionDotEps)
                {
                    duplicateDirection = true;
                    break;
                }
            }

            if (duplicateDirection)
                continue;
        }

        primitiveCandidateKeepFlags[rawIdx] = 1;
    }
}

template<typename ContactPair, typename Real>
__global__ void SuppressRedundantEdgeFaceAgainstVertexFaceKernel(
    DArray<int> primitiveCandidateKeepFlags,
    DArray<int> primitivePassCounts,
    DArray<int> primitivePassOffsets,
    DArray<ContactPair> primitiveCandidateContacts,
    Real positionNearEps,
    Real sameDirectionDotEps)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    const int slotBase = getPairPassSlot(pairId, 0);
    if (slotBase < 0
        || slotBase + (MESH_PASS_COUNT - 1) >= primitivePassCounts.size()
        || slotBase >= primitivePassOffsets.size())
        return;

    int rawCount = 0;
    for (int passType = 0; passType < MESH_PASS_COUNT; ++passType)
        rawCount += primitivePassCounts[slotBase + passType];
    if (rawCount <= 0)
        return;

    const int rawBegin = primitivePassOffsets[slotBase];
    const Real positionNearEps2 = positionNearEps * positionNearEps;
    const Real minNormalNorm2 = Real(1e-12);

    for (int rawIdx = rawBegin; rawIdx < rawBegin + rawCount; ++rawIdx)
    {
        if (rawIdx < 0 || rawIdx >= primitiveCandidateKeepFlags.size()
            || rawIdx >= primitiveCandidateContacts.size()
            || primitiveCandidateKeepFlags[rawIdx] <= 0)
            continue;

        const ContactPair edgeFace = primitiveCandidateContacts[rawIdx];
        if (edgeFace.contactType != CT_EDGE_FACE)
            continue;

        auto edgeNormal = edgeFace.normal1;
        const Real edgeNormalNorm2 = edgeNormal.normSquared();
        if (edgeNormalNorm2 <= minNormalNorm2)
            continue;
        edgeNormal /= sqrt(edgeNormalNorm2);

        bool suppressEdgeFace = false;
        for (int otherIdx = rawBegin; otherIdx < rawBegin + rawCount; ++otherIdx)
        {
            if (otherIdx < 0 || otherIdx >= primitiveCandidateKeepFlags.size()
                || otherIdx >= primitiveCandidateContacts.size()
                || primitiveCandidateKeepFlags[otherIdx] <= 0
                || otherIdx == rawIdx)
                continue;

            const ContactPair vertexFace = primitiveCandidateContacts[otherIdx];
            if (vertexFace.contactType != CT_VERTEX_FACE)
                continue;

            auto delta = vertexFace.pos1 - edgeFace.pos1;
            if (delta.normSquared() > positionNearEps2)
                continue;

            auto vertexNormal = vertexFace.normal1;
            const Real vertexNormalNorm2 = vertexNormal.normSquared();
            if (vertexNormalNorm2 <= minNormalNorm2)
                continue;
            vertexNormal /= sqrt(vertexNormalNorm2);

            if (edgeNormal.dot(vertexNormal) >= Real(1) - sameDirectionDotEps)
            {
                suppressEdgeFace = true;
                break;
            }
        }

        if (suppressEdgeFace)
            primitiveCandidateKeepFlags[rawIdx] = 0;
    }
}

__global__ void CountSelectedPrimitiveContactsPerTriPairKernel(
    DArray<int> selectedPrimitiveCounts,
    DArray<int> primitivePassCounts,
    DArray<int> primitivePassOffsets,
    DArray<int> primitiveCandidateKeepFlags)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= selectedPrimitiveCounts.size())
        return;

    const int slotBase = getPairPassSlot(pairId, 0);
    if (slotBase < 0
        || slotBase + (MESH_PASS_COUNT - 1) >= primitivePassCounts.size()
        || slotBase >= primitivePassOffsets.size())
    {
        selectedPrimitiveCounts[pairId] = 0;
        return;
    }

    int rawCount = 0;
    for (int passType = 0; passType < MESH_PASS_COUNT; ++passType)
        rawCount += primitivePassCounts[slotBase + passType];
    if (rawCount <= 0)
    {
        selectedPrimitiveCounts[pairId] = 0;
        return;
    }

    const int rawBegin = primitivePassOffsets[slotBase];
    int selectedCount = 0;
    for (int rawIdx = rawBegin; rawIdx < rawBegin + rawCount && rawIdx < primitiveCandidateKeepFlags.size(); ++rawIdx)
    {
        if (primitiveCandidateKeepFlags[rawIdx] > 0)
            ++selectedCount;
    }
    selectedPrimitiveCounts[pairId] = selectedCount;
}

__global__ void SetFinalContactCountsKernel(
    DArray<int> finalContactCounts,
    DArray<int> selectedPrimitiveCounts)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= finalContactCounts.size() || pairId >= selectedPrimitiveCounts.size())
        return;
    finalContactCounts[pairId] = selectedPrimitiveCounts[pairId];
}

template<typename ContactPair>
__global__ void SetFinalContactsPerTriPairKernel(
    DArray<ContactPair> contacts,
    DArray<int> offsets,
    DArray<int> primitivePassCounts,
    DArray<int> primitivePassOffsets,
    DArray<int> primitiveCandidateKeepFlags,
    DArray<ContactPair> primitiveCandidateContacts,
    DArray<int> selectedPrimitiveCounts)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= offsets.size() || pairId >= selectedPrimitiveCounts.size())
        return;

    const int writeBase = offsets[pairId];
    const int selectedCount = selectedPrimitiveCounts[pairId];
    if (selectedCount <= 0)
        return;

    const int slotBase = getPairPassSlot(pairId, 0);
    if (slotBase < 0
        || slotBase + (MESH_PASS_COUNT - 1) >= primitivePassCounts.size()
        || slotBase >= primitivePassOffsets.size())
        return;

    int rawCount = 0;
    for (int passType = 0; passType < MESH_PASS_COUNT; ++passType)
        rawCount += primitivePassCounts[slotBase + passType];
    const int rawBegin = primitivePassOffsets[slotBase];
    int written = 0;
    for (int rawIdx = rawBegin; rawIdx < rawBegin + rawCount && rawIdx < primitiveCandidateKeepFlags.size(); ++rawIdx)
    {
        if (primitiveCandidateKeepFlags[rawIdx] <= 0 || rawIdx >= primitiveCandidateContacts.size())
            continue;

        const int outIdx = writeBase + written;
        if (outIdx >= 0 && outIdx < contacts.size())
            contacts[outIdx] = primitiveCandidateContacts[rawIdx];
        ++written;
    }
}

} // namespace cd_internal
} // namespace dyno
