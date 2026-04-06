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
    using TemplateView = MeshTemplateKernelView<TDataType>;

    DArray<TemplateView> meshTemplates;
    DArray<BodyContactId> bodyPairs;
    DArray<int> batchBodies;
    DArray2D<int> shapeTypes;
    DArray2D<int> shapeIndices;
    DArray2D<Coord> batchPositions;
    DArray2D<Matrix> batchRotations;
    DArray2D<BoxInfo> boxes;
    int maxBodies = 0;
    DevArr2D<int> bodyToMeshTemplate;

    DevArr2D<int> body2PatchOffsets;
    DevArr2D<int> body2TriOffsets;
    DevArr2D<int> body2EdgeOffsets;
    DevArr2D<int> body2VertexOffsets;
    DArray<MeshBodyId> patch2Body;
    DArray<MeshBodyId> tri2Body;
    DArray<MeshBodyId> edge2Body;
    DArray<int> patch2TriOffsets;
    DArray<int> patch2TriIndices;

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
    MeshBodyId tri0Body;
    MeshBodyId tri1Body;
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

    // Coord longestEdge = p1 - p0;
    // Real longestEdgeLen = longestEdge.normSquared();

    // Coord edge1 = p2 - p1;
    // Real edge1Len = edge1.normSquared();
    // if (edge1Len > longestEdgeLen)
    // {
    //     longestEdge = edge1;
    //     longestEdgeLen = edge1Len;
    // }

    // Coord edge2 = p0 - p2;
    // if (edge2.normSquared() > longestEdgeLen)
    //     longestEdge = edge2;

    // return stablePerpendicular(longestEdge);
}

template<typename Real>
DYN_FUNC inline Real absValue(Real v)
{
    return v < Real(0) ? -v : v;
}

template<typename View>
DYN_FUNC inline bool isValidBodyId(
    const View& view,
    int envId,
    int bodyId)
{
    if (envId < 0 || envId >= view.batchBodies.size())
        return false;
    if (bodyId < 0 || bodyId >= view.batchBodies[envId])
        return false;
    return true;
}

template<typename View>
DYN_FUNC inline bool hasBodyLayoutEntry(
    const View& view,
    const DevArr2D<int>& offsets,
    int envId,
    int bodyId)
{
    if (!isValidBodyId(view, envId, bodyId))
        return false;
    if (envId < 0 || envId >= offsets.NumBlocks())
        return false;
    if (bodyId < 0 || bodyId >= offsets.BlockSize(envId))
        return false;
    return true;
}

template<typename View>
DYN_FUNC inline int getBodyLayoutBase(
    const View& view,
    const DevArr2D<int>& offsets,
    int envId,
    int bodyId)
{
    if (!hasBodyLayoutEntry(view, offsets, envId, bodyId))
        return -1;

    const int* block = offsets.BlockPtr(envId);
    return block != nullptr ? block[bodyId] : -1;
}

template<typename View>
DYN_FUNC inline int getBodyTemplateId(
    const View& view,
    int envId,
    int bodyId)
{
    if (!isValidBodyId(view, envId, bodyId))
        return -1;
    if (envId < 0 || envId >= view.bodyToMeshTemplate.NumBlocks())
        return -1;
    if (bodyId < 0 || bodyId >= view.bodyToMeshTemplate.BlockSize(envId))
        return -1;
    const int templateId = view.bodyToMeshTemplate(envId, bodyId);
    return templateId >= 0 && templateId < view.meshTemplates.size() ? templateId : -1;
}

template<typename View>
DYN_FUNC inline const typename View::TemplateView* getBodyTemplateView(
    const View& view,
    int envId,
    int bodyId)
{
    const int templateId = getBodyTemplateId(view, envId, bodyId);
    return templateId >= 0 ? &view.meshTemplates[templateId] : nullptr;
}

template<typename View>
DYN_FUNC inline int getBodyTemplatePatchCount(
    const View& view,
    int envId,
    int bodyId)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    return tpl != nullptr ? (tpl->numPatches > 0 ? tpl->numPatches : 1) : 0;
}

template<typename View>
DYN_FUNC inline int getBodyTemplateTriangleCount(
    const View& view,
    int envId,
    int bodyId)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    return tpl != nullptr ? tpl->numTriangles : 0;
}

template<typename View>
DYN_FUNC inline int getBodyTemplateEdgeCount(
    const View& view,
    int envId,
    int bodyId)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    return tpl != nullptr ? tpl->numEdges : 0;
}

template<typename View>
DYN_FUNC inline int getBodyTemplateVertexCount(
    const View& view,
    int envId,
    int bodyId)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    return tpl != nullptr ? tpl->numVertices : 0;
}

template<typename View>
DYN_FUNC inline bool getBodyBoxTransform(
    const View& view,
    int envId,
    int bodyId,
    typename View::Coord& shapeCenter,
    typename View::Matrix& shapeRotation,
    typename View::Coord& halfLength,
    typename View::Coord* invHalfLength = nullptr)
{
    using Real = typename View::Real;

    if (!isValidBodyId(view, envId, bodyId))
        return false;

    if (view.shapeTypes(envId, bodyId) != 1)
        return false;

    const int shapeIdx = view.shapeIndices(envId, bodyId);
    if (shapeIdx < 0)
        return false;

    const BoxInfo box = view.boxes(envId, shapeIdx);
    const auto bodyPos = view.batchPositions(envId, bodyId);
    const auto bodyRot = view.batchRotations(envId, bodyId);

    shapeCenter = bodyPos + bodyRot * box.center;
    shapeRotation = bodyRot * box.rot.toMatrix3x3();
    halfLength = box.halfLength;

    if (invHalfLength != nullptr)
    {
        *invHalfLength = typename View::Coord(
            halfLength[0] != Real(0) ? Real(1) / halfLength[0] : Real(0),
            halfLength[1] != Real(0) ? Real(1) / halfLength[1] : Real(0),
            halfLength[2] != Real(0) ? Real(1) / halfLength[2] : Real(0));
    }

    return true;
}

template<typename View>
DYN_FUNC inline bool getWorldVertex(
    const View& view,
    int globalVertexId,
    int envId,
    int bodyId,
    typename View::Coord& p)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    if (tpl == nullptr)
        return false;

    const int begin = getBodyLayoutBase(view, view.body2VertexOffsets, envId, bodyId);
    if (begin < 0)
        return false;

    const int end = begin + tpl->numVertices;
    if (globalVertexId < begin || globalVertexId >= end)
        return false;

    const int localVertexId = globalVertexId - begin;
    if (localVertexId < 0 || localVertexId >= tpl->numVertices)
        return false;

    typename View::Coord shapeCenter;
    typename View::Matrix shapeRotation;
    typename View::Coord halfLength;
    if (!getBodyBoxTransform(view, envId, bodyId, shapeCenter, shapeRotation, halfLength))
        return false;

    const auto& local = tpl->vertices[localVertexId];
    const auto scaled = scalePoint<typename View::Real, typename View::Coord>(local, halfLength);
    p = shapeCenter + shapeRotation * scaled;
    return true;
}

template<typename View>
DYN_FUNC inline bool getWorldTriangle(
    const View& view,
    int globalTriId,
    int envId,
    int bodyId,
    typename View::Coord& p0,
    typename View::Coord& p1,
    typename View::Coord& p2)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    if (tpl == nullptr)
        return false;

    const int begin = getBodyLayoutBase(view, view.body2TriOffsets, envId, bodyId);
    if (begin < 0)
        return false;

    const int end = begin + tpl->numTriangles;
    if (globalTriId < begin || globalTriId >= end)
        return false;

    const int localTriId = globalTriId - begin;
    if (localTriId < 0 || localTriId >= tpl->numTriangles)
        return false;

    const auto tri = tpl->triangles[localTriId];
    const int vertexBase = getBodyLayoutBase(view, view.body2VertexOffsets, envId, bodyId);
    if (vertexBase < 0)
        return false;

    return getWorldVertex(view, vertexBase + tri[0], envId, bodyId, p0)
        && getWorldVertex(view, vertexBase + tri[1], envId, bodyId, p1)
        && getWorldVertex(view, vertexBase + tri[2], envId, bodyId, p2);
}

template<typename View>
DYN_FUNC inline bool getWorldEdge(
    const View& view,
    int globalEdgeId,
    int envId,
    int bodyId,
    TSegment3D<typename View::Real>& segment)
{
    const auto* tpl = getBodyTemplateView(view, envId, bodyId);
    if (tpl == nullptr)
        return false;

    const int begin = getBodyLayoutBase(view, view.body2EdgeOffsets, envId, bodyId);
    if (begin < 0)
        return false;

    const int end = begin + tpl->numEdges;
    if (globalEdgeId < begin || globalEdgeId >= end)
        return false;

    const int localEdgeId = globalEdgeId - begin;
    if (localEdgeId < 0 || localEdgeId >= tpl->numEdges)
        return false;

    const auto edge = tpl->edges[localEdgeId];
    typename View::Coord p0;
    typename View::Coord p1;
    const int vertexBase = getBodyLayoutBase(view, view.body2VertexOffsets, envId, bodyId);
    if (vertexBase < 0)
        return false;

    if (!getWorldVertex(view, vertexBase + edge[0], envId, bodyId, p0)
        || !getWorldVertex(view, vertexBase + edge[1], envId, bodyId, p1))
        return false;

    segment = TSegment3D<typename View::Real>(p0, p1);
    return true;
}

template<typename View>
DYN_FUNC inline typename View::Coord transformWorldPointToTargetRest(
    const View& view,
    const typename View::Coord& pWorld,
    int targetEnvId,
    int targetBodyId)
{
    typename View::Coord shapeCenter;
    typename View::Matrix shapeRotation;
    typename View::Coord halfLength;
    typename View::Coord invHalfLength;
    if (!getBodyBoxTransform(view, targetEnvId, targetBodyId, shapeCenter, shapeRotation, halfLength, &invHalfLength))
        return typename View::Coord(0);

    const auto local = shapeRotation.transpose() * (pWorld - shapeCenter);
    return scalePoint<typename View::Real, typename View::Coord>(local, invHalfLength);
}

template<typename View>
DYN_FUNC inline bool buildSourcePatchAabbInTargetRest(
    const View& view,
    int sourcePatchId,
    int targetEnvId,
    int targetBodyId,
    typename View::AABB& outAabb)
{
    using Coord = typename View::Coord;
    using Real = typename View::Real;

    if (sourcePatchId < 0 || sourcePatchId + 1 >= view.patch2TriOffsets.size())
        return false;

    if (sourcePatchId < 0 || sourcePatchId >= view.patch2Body.size())
        return false;

    const MeshBodyId sourceBody = view.patch2Body[sourcePatchId];
    if (!isValidBodyId(view, sourceBody.env_id, sourceBody.body_id)
        || !isValidBodyId(view, targetEnvId, targetBodyId))
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
        if (!getWorldTriangle(view, globalTriId, sourceBody.env_id, sourceBody.body_id, p0, p1, p2))
            continue;

        const Coord r0 = transformWorldPointToTargetRest(view, p0, targetEnvId, targetBodyId);
        const Coord r1 = transformWorldPointToTargetRest(view, p1, targetEnvId, targetBodyId);
        const Coord r2 = transformWorldPointToTargetRest(view, p2, targetEnvId, targetBodyId);
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
    DArray<MeshBodyId> sourceTargetBodies,
    LinearBVH<TDataType> templatePatchBvh,
    View view)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= counts.size() || sourceId >= sourcePatchIds.size() || sourceId >= sourceTargetBodies.size())
        return;

    const int sourcePatchId = sourcePatchIds[sourceId];
    const MeshBodyId targetBody = sourceTargetBodies[sourceId];
    if (!hasBodyLayoutEntry(view, view.body2PatchOffsets, targetBody.env_id, targetBody.body_id))
    {
        counts[sourceId] = 0;
        return;
    }

    const auto* targetTemplate = getBodyTemplateView(view, targetBody.env_id, targetBody.body_id);
    const int targetPatchCount = getBodyTemplatePatchCount(view, targetBody.env_id, targetBody.body_id);
    if (targetPatchCount <= 0)
    {
        counts[sourceId] = 0;
        return;
    }

    typename View::AABB sourceAabb;
    if (!buildSourcePatchAabbInTargetRest(view, sourcePatchId, targetBody.env_id, targetBody.body_id, sourceAabb))
    {
        counts[sourceId] = 0;
        return;
    }

    if (targetPatchCount == 1)
    {
        counts[sourceId] = (targetTemplate != nullptr
            && targetTemplate->patchAABBs.size() > 0
            && sourceAabb.checkOverlap(targetTemplate->patchAABBs[0])) ? 1 : 0;
        return;
    }

    int hitCount = 0;
    for (int localPatchId = 0; localPatchId < targetPatchCount; ++localPatchId)
    {
        if (targetTemplate != nullptr
            && localPatchId < targetTemplate->patchAABBs.size()
            && sourceAabb.checkOverlap(targetTemplate->patchAABBs[localPatchId]))
            ++hitCount;
    }
    counts[sourceId] = hitCount;
}

template<typename TDataType, typename View>
__global__ void RequestPatchPairHitsKernel(
    DArrayList<int> hitLists,
    DArray<int> sourcePatchIds,
    DArray<MeshBodyId> sourceTargetBodies,
    LinearBVH<TDataType> templatePatchBvh,
    View view)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= hitLists.size() || sourceId >= sourcePatchIds.size() || sourceId >= sourceTargetBodies.size())
        return;

    auto& list = hitLists[sourceId];
    list.clear();

    const int sourcePatchId = sourcePatchIds[sourceId];
    const MeshBodyId targetBody = sourceTargetBodies[sourceId];
    if (!hasBodyLayoutEntry(view, view.body2PatchOffsets, targetBody.env_id, targetBody.body_id))
        return;

    const auto* targetTemplate = getBodyTemplateView(view, targetBody.env_id, targetBody.body_id);
    const int targetPatchCount = getBodyTemplatePatchCount(view, targetBody.env_id, targetBody.body_id);
    if (targetPatchCount <= 0)
        return;

    typename View::AABB sourceAabb;
    if (!buildSourcePatchAabbInTargetRest(view, sourcePatchId, targetBody.env_id, targetBody.body_id, sourceAabb))
        return;

    if (targetPatchCount == 1)
    {
        if (targetTemplate != nullptr
            && targetTemplate->patchAABBs.size() > 0
            && sourceAabb.checkOverlap(targetTemplate->patchAABBs[0]))
            list.insert(0);
        return;
    }

    for (int localPatchId = 0; localPatchId < targetPatchCount; ++localPatchId)
    {
        if (targetTemplate != nullptr
            && localPatchId < targetTemplate->patchAABBs.size()
            && sourceAabb.checkOverlap(targetTemplate->patchAABBs[localPatchId]))
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
    DArray<MeshBodyId> sourceTargetBodies,
    DArray<MeshBodyId> patch2Body,
    DevArr2D<int> body2PatchOffsets)
{
    int sourceId = threadIdx.x + blockIdx.x * blockDim.x;
    if (sourceId >= hitLists.size() || sourceId >= offsets.size() || sourceId >= counts.size())
        return;

    const int count = counts[sourceId];
    if (count <= 0)
        return;

    const int sourcePatchId = sourcePatchIds[sourceId];
    const MeshBodyId targetBody = sourceTargetBodies[sourceId];
    if (sourcePatchId < 0 || sourcePatchId >= patch2Body.size())
        return;

    const MeshBodyId sourceBody = patch2Body[sourcePatchId];
    if (sourceBody.env_id != targetBody.env_id
        || sourceBody.env_id < 0
        || sourceBody.env_id >= body2PatchOffsets.NumBlocks()
        || sourceBody.body_id < 0
        || sourceBody.body_id >= body2PatchOffsets.BlockSize(sourceBody.env_id)
        || targetBody.body_id < 0
        || targetBody.body_id >= body2PatchOffsets.BlockSize(sourceBody.env_id))
        return;

    const int targetPatchBase = body2PatchOffsets(targetBody.env_id, targetBody.body_id);

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
        pair.env_id = sourceBody.env_id;
        pair.body_a = sourceBody.body_id;
        pair.body_b = targetBody.body_id;
        pair.patch_a = sourcePatchId;
        pair.patch_b = targetPatchBase + localPatchId;
        pair.type = MESH_MESH;
        patchPairs[outIdx] = pair;
        ++written;
    }
}

template<typename View>
__device__ inline void PrepareTriangleWorldDataAtId(
    const View& view,
    int triId)
{
    using Coord = typename View::Coord;

    if (triId >= view.tri2Body.size())
        return;

    const MeshBodyId owner = view.tri2Body[triId];
    Coord p0, p1, p2;
    if (!getWorldTriangle(view, triId, owner.env_id, owner.body_id, p0, p1, p2))
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
__global__ void PrepareTriangleWorldDataKernel(View view)
{
    int triId = threadIdx.x + blockIdx.x * blockDim.x;
    if (triId >= view.triangleAabbsWorld.size() || triId >= view.faceNormalsWorld.size())
        return;

    PrepareTriangleWorldDataAtId(view, triId);
}

template<typename View>
__global__ void PrepareTriangleWorldDataWorklistKernel(
    DArray<int> triIds,
    View view)
{
    int workId = threadIdx.x + blockIdx.x * blockDim.x;
    if (workId >= triIds.size())
        return;

    const int triId = triIds[workId];
    if (triId < 0 || triId >= view.triangleAabbsWorld.size() || triId >= view.faceNormalsWorld.size())
        return;

    PrepareTriangleWorldDataAtId(view, triId);
}

template<typename View>
__device__ inline void PrepareEdgeNormalsWorldAtId(
    const View& view,
    int edgeId)
{
    using Coord = typename View::Coord;
    using Real = typename View::Real;

    if (edgeId >= view.edge2Body.size())
        return;

    const MeshBodyId owner = view.edge2Body[edgeId];
    const auto* ownerTemplate = getBodyTemplateView(view, owner.env_id, owner.body_id);
    const int edgeBase = getBodyLayoutBase(view, view.body2EdgeOffsets, owner.env_id, owner.body_id);
    const int triBase = getBodyLayoutBase(view, view.body2TriOffsets, owner.env_id, owner.body_id);
    if (ownerTemplate == nullptr || edgeBase < 0 || triBase < 0)
    {
        view.edgeNormalsWorld[edgeId] = Coord(1, 0, 0);
        return;
    }

    const int localEdgeId = edgeId - edgeBase;
    const Real epsSqr = Real(1e-12);

    Coord edgeNormal(0);
    if (localEdgeId >= 0 && localEdgeId < ownerTemplate->numEdges)
    {
        const auto adjacentFaces = ownerTemplate->edgeAdjacentFaces[localEdgeId];
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
        if (getWorldEdge(view, edgeId, owner.env_id, owner.body_id, edgeSegment))
            edgeNormal = stablePerpendicular(edgeSegment.direction());
        else
            edgeNormal = Coord(1, 0, 0);
    }

    view.edgeNormalsWorld[edgeId] = normalizeOrFallback(edgeNormal, Coord(1, 0, 0));
}

template<typename View>
__global__ void PrepareEdgeNormalsWorldKernel(View view)
{
    int edgeId = threadIdx.x + blockIdx.x * blockDim.x;
    if (edgeId >= view.edgeNormalsWorld.size())
        return;

    PrepareEdgeNormalsWorldAtId(view, edgeId);
}

template<typename View>
__global__ void PrepareEdgeNormalsWorldWorklistKernel(
    DArray<int> edgeIds,
    View view)
{
    int workId = threadIdx.x + blockIdx.x * blockDim.x;
    if (workId >= edgeIds.size())
        return;

    const int edgeId = edgeIds[workId];
    if (edgeId < 0 || edgeId >= view.edgeNormalsWorld.size())
        return;

    PrepareEdgeNormalsWorldAtId(view, edgeId);
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

template<typename View>
DYN_FUNC inline bool isCoplanarInternalTriangleEdge(
    const View& view,
    const typename View::TemplateView& targetTemplate,
    int targetTriBase,
    int targetTriId,
    int localEdgeId)
{
    using Real = typename View::Real;

    if (localEdgeId < 0 || targetTriBase < 0)
        return false;

    const int localTriId = targetTriId - targetTriBase;
    if (localTriId < 0 || localTriId >= targetTemplate.numTriangles)
        return false;

    const int localTemplateEdgeId = targetTemplate.triangleEdges[localTriId][localEdgeId];
    if (localTemplateEdgeId < 0 || localTemplateEdgeId >= targetTemplate.numEdges)
        return false;

    const auto adjacentFaces = targetTemplate.edgeAdjacentFaces[localTemplateEdgeId];
    if (adjacentFaces[0] < 0 || adjacentFaces[1] < 0)
        return false;

    const int globalFace0 = targetTriBase + adjacentFaces[0];
    const int globalFace1 = targetTriBase + adjacentFaces[1];
    if (globalFace0 < 0 || globalFace0 >= view.faceNormalsWorld.size()
        || globalFace1 < 0 || globalFace1 >= view.faceNormalsWorld.size())
        return false;

    auto n0 = view.faceNormalsWorld[globalFace0];
    auto n1 = view.faceNormalsWorld[globalFace1];
    const Real n0Norm2 = n0.normSquared();
    const Real n1Norm2 = n1.normSquared();
    if (n0Norm2 <= Real(1e-12) || n1Norm2 <= Real(1e-12))
        return false;

    n0 /= sqrt(n0Norm2);
    n1 /= sqrt(n1Norm2);
    return n0.dot(n1) >= Real(1) - Real(1e-4);
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
    const MeshBodyId& sourceBody,
    int targetEdgeId,
    const MeshBodyId& targetBody,
    typename View::Coord& contactPoint,
    typename View::Coord& nTarget,
    typename View::Real& depth)
{
    using Real = typename View::Real;
    using Coord = typename View::Coord;

    const Real epsSqr = Real(1e-12);
    TSegment3D<Real> sourceSegment;
    TSegment3D<Real> targetSegment;
    if (!getWorldEdge(view, sourceEdgeId, sourceBody.env_id, sourceBody.body_id, sourceSegment)
        || !getWorldEdge(view, targetEdgeId, targetBody.env_id, targetBody.body_id, targetSegment))
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
    const MeshBodyId& sourceBody,
    int sourceVertexId,
    int targetTriId,
    const MeshBodyId& targetBody,
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
    if (!getWorldVertex(view, sourceVertexId, sourceBody.env_id, sourceBody.body_id, p))
        return false;

    Coord r = TPoint3D<Real>(p).project(targetTriangle).origin;
    int regionType = MESH_REGION_INVALID;
    int localEdgeId = -1;
    int localVertexId = -1;
    Real bary[3] = { Real(0), Real(0), Real(0) };
    if (!classifyTriangleRegion(targetTriangle, r, epsBary, regionType, localEdgeId, localVertexId, bary))
        return false;

    const auto* targetTemplate = getBodyTemplateView(view, targetBody.env_id, targetBody.body_id);
    const int targetTriBase = getBodyLayoutBase(view, view.body2TriOffsets, targetBody.env_id, targetBody.body_id);
    if (targetTemplate == nullptr || targetTriBase < 0)
        return false;

    if (regionType == MESH_REGION_EDGE
        && isCoplanarInternalTriangleEdge(view, *targetTemplate, targetTriBase, targetTriId, localEdgeId))
    {
        regionType = MESH_REGION_FACE;
    }

    if (regionType != MESH_REGION_FACE)
        return false;

    Coord faceNormal = targetTriId >= 0 && targetTriId < view.faceNormalsWorld.size()
        ? view.faceNormalsWorld[targetTriId]
        : buildRobustFaceNormal(targetTriangle.v[0], targetTriangle.v[1], targetTriangle.v[2]);
    // nTarget = normalizeOrFallback(faceNormal, stablePerpendicular(targetTriangle.v[1] - targetTriangle.v[0]));
    nTarget = faceNormal;
    
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
    const MeshBodyId& sourceBody,
    int sourceEdgeId,
    int targetTriId,
    const MeshBodyId& targetBody,
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
    if (!getWorldEdge(view, sourceEdgeId, sourceBody.env_id, sourceBody.body_id, sourceSegment))
        return false;

    auto pq = sourceSegment.proximity(targetTriangle);
    Coord cSource = pq.startPoint();
    Coord cTarget = pq.endPoint();
    int regionType = MESH_REGION_INVALID;
    int localEdgeId = -1;
    int localVertexId = -1;
    Real bary[3] = { Real(0), Real(0), Real(0) };
    if (!classifyTriangleRegion(targetTriangle, cTarget, epsBary, regionType, localEdgeId, localVertexId, bary))
        return false;

    const int targetTriBase = getBodyLayoutBase(view, view.body2TriOffsets, targetBody.env_id, targetBody.body_id);
    const int targetEdgeBase = getBodyLayoutBase(view, view.body2EdgeOffsets, targetBody.env_id, targetBody.body_id);
    const auto* targetTemplate = getBodyTemplateView(view, targetBody.env_id, targetBody.body_id);
    if (targetTemplate == nullptr || targetTriBase < 0 || targetEdgeBase < 0)
        return false;

    const int targetLocalTriId = targetTriId - targetTriBase;
    if (targetLocalTriId < 0 || targetLocalTriId >= targetTemplate->numTriangles)
        return false;

    if (regionType == MESH_REGION_EDGE
        && isCoplanarInternalTriangleEdge(view, *targetTemplate, targetTriBase, targetTriId, localEdgeId))
    {
        regionType = MESH_REGION_FACE;
    }

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

    if (minSignedDistance <= edgeActivation)
    {
        bool useFaceContact = (regionType == MESH_REGION_FACE);
        if (!useFaceContact)
        {
            Coord sourceDir = sourceSegment.direction();
            const Real sourceDirNorm2 = sourceDir.normSquared();
            if (sourceDirNorm2 > Real(1e-12))
            {
                sourceDir /= sqrt(sourceDirNorm2);
                const Real absDirFaceDot = absValue(sourceDir.dot(nTarget));
                // If the edge is nearly parallel to the face plane, keep the
                // face normal even when the closest point lies on a triangle
                // boundary. This preserves the expected supporting contact for
                // offset box-on-box stacking.
                useFaceContact = absDirFaceDot <= Real(0.25);
            }
        }

        if (useFaceContact)
        {
            contactPoint = cTarget;
            depth = (cTarget - cSource).dot(nTarget);
            if (depth < Real(0))
                depth = Real(0);
            contactType = CT_EDGE_FACE;
            return true;
        }
    }

    if (regionType == MESH_REGION_EDGE)
    {
        const int targetEdgeId = targetEdgeBase + targetTemplate->triangleEdges[targetLocalTriId][localEdgeId];
        if (!buildEdgeEdgeContact(view, sourceEdgeId, sourceBody, targetEdgeId, targetBody, contactPoint, nTarget, depth))
            return false;
        contactType = CT_EDGE_EDGE;
        return true;
    }

    if (regionType == MESH_REGION_VERTEX)
    {
        int edge0 = -1;
        int edge1 = -1;
        if (!getLocalIncidentEdges(targetTemplate->triangleEdges[targetLocalTriId], localVertexId, edge0, edge1))
            return false;

        int bestTargetEdge = -1;
        Real bestDist2 = std::numeric_limits<Real>::max();
        Real bestAlign = Real(-1);

        if (edge0 >= 0)
        {
            const int globalTargetEdgeId = targetEdgeBase + edge0;
            TSegment3D<Real> targetSegment;
            if (getWorldEdge(view, globalTargetEdgeId, targetBody.env_id, targetBody.body_id, targetSegment))
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
            const int globalTargetEdgeId = targetEdgeBase + edge1;
            TSegment3D<Real> targetSegment;
            if (getWorldEdge(view, globalTargetEdgeId, targetBody.env_id, targetBody.body_id, targetSegment))
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

        if (!buildEdgeEdgeContact(view, sourceEdgeId, sourceBody, bestTargetEdge, targetBody, contactPoint, nTarget, depth))
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
    // Keep a single convention for solver assembly:
    // normal1 points from bodyId2 toward bodyId1, and normal2 is its opposite.
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
DYN_FUNC inline bool getPairBodiesForContext(
    const View& view,
    int pairId,
    int& bodyId1,
    int& bodyId2,
    MeshBodyId& body0,
    MeshBodyId& body1)
{
    if (pairId >= 0 && pairId < view.patchPairs.size())
    {
        const PatchPair pair = view.patchPairs[pairId];
        body0.env_id = pair.env_id;
        body0.body_id = pair.body_a;
        body1.env_id = pair.env_id;
        body1.body_id = pair.body_b;
        if (!isValidBodyId(view, body0.env_id, body0.body_id)
            || !isValidBodyId(view, body1.env_id, body1.body_id)
            || view.maxBodies <= 0)
            return false;

        bodyId1 = body0.env_id * view.maxBodies + body0.body_id;
        bodyId2 = body1.env_id * view.maxBodies + body1.body_id;
        return true;
    }

    if (pairId >= 0 && pairId < view.bodyPairs.size())
    {
        const BodyContactId pair = view.bodyPairs[pairId];
        body0.env_id = pair.env_id;
        body0.body_id = pair.body_id_1;
        body1.env_id = pair.env_id;
        body1.body_id = pair.body_id_2;
        if (!isValidBodyId(view, body0.env_id, body0.body_id)
            || !isValidBodyId(view, body1.env_id, body1.body_id)
            || view.maxBodies <= 0)
            return false;

        bodyId1 = body0.env_id * view.maxBodies + body0.body_id;
        bodyId2 = body1.env_id * view.maxBodies + body1.body_id;
        return true;
    }

    return false;
}

template<typename View>
DYN_FUNC inline bool buildTriPairContext(
    const View& view,
    int tri0,
    int tri1,
    int pairId,
    TriPairContext<View>& ctx)
{
    if (!getPairBodiesForContext(view, pairId, ctx.bodyId1, ctx.bodyId2, ctx.tri0Body, ctx.tri1Body))
        return false;

    typename View::Coord p00, p01, p02;
    typename View::Coord p10, p11, p12;
    if (!getWorldTriangle(view, tri0, ctx.tri0Body.env_id, ctx.tri0Body.body_id, p00, p01, p02)
        || !getWorldTriangle(view, tri1, ctx.tri1Body.env_id, ctx.tri1Body.body_id, p10, p11, p12))
        return false;

    ctx.tri0 = tri0;
    ctx.tri1 = tri1;
    ctx.triangle0 = TTriangle3D<typename View::Real>(p00, p01, p02);
    ctx.triangle1 = TTriangle3D<typename View::Real>(p10, p11, p12);
    return true;
}

template<typename View>
__global__ void CountTriPairsPerBodyPairKernel(
    DArray<int> counts,
    View view)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= counts.size() || pairId >= view.bodyPairs.size())
        return;

    const BodyContactId pair = view.bodyPairs[pairId];
    if (pair.env_id < 0 || pair.env_id >= view.batchBodies.size()
        || pair.body_id_1 < 0 || pair.body_id_2 < 0
        || pair.body_id_1 >= view.batchBodies[pair.env_id]
        || pair.body_id_2 >= view.batchBodies[pair.env_id])
    {
        counts[pairId] = 0;
        return;
    }

    if (view.shapeTypes(pair.env_id, pair.body_id_1) != 1
        || view.shapeTypes(pair.env_id, pair.body_id_2) != 1)
    {
        counts[pairId] = 0;
        return;
    }

    const int begin0 = getBodyLayoutBase(view, view.body2TriOffsets, pair.env_id, pair.body_id_1);
    const int begin1 = getBodyLayoutBase(view, view.body2TriOffsets, pair.env_id, pair.body_id_2);
    const int count0 = getBodyTemplateTriangleCount(view, pair.env_id, pair.body_id_1);
    const int count1 = getBodyTemplateTriangleCount(view, pair.env_id, pair.body_id_2);
    if (begin0 < 0 || begin1 < 0 || count0 <= 0 || count1 <= 0)
    {
        counts[pairId] = 0;
        return;
    }

    counts[pairId] = count0 * count1;
}

template<typename View>
__global__ void SetTriPairsFromBodyPairsKernel(
    DArray<int> tri0Out,
    DArray<int> tri1Out,
    DArray<int> pairIdOut,
    DArray<int> offsets,
    DArray<int> counts,
    View view)
{
    int pairId = threadIdx.x + blockIdx.x * blockDim.x;
    if (pairId >= view.bodyPairs.size() || pairId >= offsets.size() || pairId >= counts.size())
        return;

    const int count = counts[pairId];
    if (count <= 0)
        return;

    const BodyContactId pair = view.bodyPairs[pairId];
    if (pair.env_id < 0 || pair.env_id >= view.batchBodies.size()
        || pair.body_id_1 < 0 || pair.body_id_2 < 0
        || pair.body_id_1 >= view.batchBodies[pair.env_id]
        || pair.body_id_2 >= view.batchBodies[pair.env_id])
        return;

    if (view.shapeTypes(pair.env_id, pair.body_id_1) != 1
        || view.shapeTypes(pair.env_id, pair.body_id_2) != 1)
        return;

    const int begin0 = getBodyLayoutBase(view, view.body2TriOffsets, pair.env_id, pair.body_id_1);
    const int begin1 = getBodyLayoutBase(view, view.body2TriOffsets, pair.env_id, pair.body_id_2);
    const int count0 = getBodyTemplateTriangleCount(view, pair.env_id, pair.body_id_1);
    const int count1 = getBodyTemplateTriangleCount(view, pair.env_id, pair.body_id_2);
    if (begin0 < 0 || begin1 < 0 || count0 <= 0 || count1 <= 0)
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
    MeshBodyId& sourceBody,
    const TTriangle3D<typename View::Real>*& sourceTriangle,
    int& targetTriId,
    MeshBodyId& targetBody,
    const TTriangle3D<typename View::Real>*& targetTriangle,
    bool& targetIsTri1,
    bool& vertexPass)
{
    switch (passType)
    {
    case MESH_PASS_TRI0_VERTEX:
        sourceTriId = ctx.tri0;
        sourceBody = ctx.tri0Body;
        sourceTriangle = &ctx.triangle0;
        targetTriId = ctx.tri1;
        targetBody = ctx.tri1Body;
        targetTriangle = &ctx.triangle1;
        targetIsTri1 = true;
        vertexPass = true;
        return true;
    case MESH_PASS_TRI0_EDGE:
        sourceTriId = ctx.tri0;
        sourceBody = ctx.tri0Body;
        sourceTriangle = &ctx.triangle0;
        targetTriId = ctx.tri1;
        targetBody = ctx.tri1Body;
        targetTriangle = &ctx.triangle1;
        targetIsTri1 = true;
        vertexPass = false;
        return true;
    case MESH_PASS_TRI1_VERTEX:
        sourceTriId = ctx.tri1;
        sourceBody = ctx.tri1Body;
        sourceTriangle = &ctx.triangle1;
        targetTriId = ctx.tri0;
        targetBody = ctx.tri0Body;
        targetTriangle = &ctx.triangle0;
        targetIsTri1 = false;
        vertexPass = true;
        return true;
    case MESH_PASS_TRI1_EDGE:
        sourceTriId = ctx.tri1;
        sourceBody = ctx.tri1Body;
        sourceTriangle = &ctx.triangle1;
        targetTriId = ctx.tri0;
        targetBody = ctx.tri0Body;
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
    MeshBodyId sourceBody;
    int targetTriId = -1;
    MeshBodyId targetBody;
    const TTriangle3D<typename View::Real>* sourceTriangle = nullptr;
    const TTriangle3D<typename View::Real>* targetTriangle = nullptr;
    bool targetIsTri1 = true;
    bool vertexPass = true;
    if (!getPrimitivePassContext(
            ctx,
            passType,
            sourceTriId,
            sourceBody,
            sourceTriangle,
            targetTriId,
            targetBody,
            targetTriangle,
            targetIsTri1,
            vertexPass))
        return 0;

    int count = 0;
    const int sourceTriBase = getBodyLayoutBase(view, view.body2TriOffsets, sourceBody.env_id, sourceBody.body_id);
    const auto* sourceTemplate = getBodyTemplateView(view, sourceBody.env_id, sourceBody.body_id);
    if (sourceTemplate == nullptr || sourceTriBase < 0)
        return 0;

    const int localSourceTriId = sourceTriId - sourceTriBase;
    if (localSourceTriId < 0 || localSourceTriId >= sourceTemplate->numTriangles)
        return 0;

    if (vertexPass)
    {
        const auto sourceTriIndices = sourceTemplate->triangles[localSourceTriId];
        const int vertexBase = getBodyLayoutBase(view, view.body2VertexOffsets, sourceBody.env_id, sourceBody.body_id);
        if (vertexBase < 0)
            return 0;
        for (int localVertexId = 0; localVertexId < 3; ++localVertexId)
        {
            const int globalVertexId = vertexBase + sourceTriIndices[localVertexId];
            typename View::Coord sourcePoint;
            if (!getWorldVertex(view, globalVertexId, sourceBody.env_id, sourceBody.body_id, sourcePoint))
                continue;

            typename View::Coord targetPoint;
            typename View::Coord nTarget;
            typename View::Real depth = typename View::Real(0);
            ContactType type = CT_UNKNOWN;
            if (!tryVertexTriangleContact(
                    view,
                    sourceBody,
                    globalVertexId,
                    targetTriId,
                    targetBody,
                    *targetTriangle,
                    targetPoint,
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
                    writeContact(cp, ctx.bodyId1, ctx.bodyId2, ctx.tri0, ctx.tri1, targetPoint, nTarget, targetIsTri1, depth, type);
                    contacts[outIdx] = cp;
                    primitiveKeys[outIdx] = encodeVertexPrimitiveKey(globalVertexId);
                }
            }

            ++count;
        }
        return count;
    }

    const int edgeBase = getBodyLayoutBase(view, view.body2EdgeOffsets, sourceBody.env_id, sourceBody.body_id);
    if (edgeBase < 0)
        return 0;
    const auto sourceTriEdges = sourceTemplate->triangleEdges[localSourceTriId];
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
                sourceBody,
                globalEdgeId,
                targetTriId,
                targetBody,
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

    // SimNode point contacts should represent the nearest current contact on a
    // primitive. Keep the shallowest compatible candidate so a remote side
    // face cannot suppress a closer supporting face contact.
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
