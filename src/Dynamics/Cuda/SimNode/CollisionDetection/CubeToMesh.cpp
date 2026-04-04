#include "CubeToMesh.h"
#include "MeshPatching/MeshTopologyBuilder.h"
#include "MeshPatching/KMeansPatcher.h"
#include "MeshPatching/PatchingTypes.h"

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <cassert>

namespace dyno {

namespace {

// Pack two vertex indices into a canonical edge key (smaller index in high bits)
static inline uint64_t packEdgeKey(int a, int b)
{
    uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

} // anonymous namespace

template<typename TDataType>
void GenerateUnitCubeMesh(MeshTemplateData<TDataType>& out)
{
    using Real = typename TDataType::Real;
    using Coord = typename TDataType::Coord;
    using AABB = TAlignedBox3D<Real>;
    using Triangle = typename TopologyModule::Triangle;
    using Edge = typename TopologyModule::Edge;
    using Tri2Edg = typename TopologyModule::Tri2Edg;
    using Edg2Tri = typename TopologyModule::Edg2Tri;

    // ========== 1. Generate unit cube geometry ==========
    // 8 vertices at (+/-1, +/-1, +/-1)
    std::vector<Coord> vertices_host(8);
    for (int i = 0; i < 8; i++) {
        Real x = (i & 1) ? Real(1) : Real(-1);
        Real y = (i & 2) ? Real(1) : Real(-1);
        Real z = (i & 4) ? Real(1) : Real(-1);
        vertices_host[i] = Coord(x, y, z);
    }

    // Vertex ordering (bit pattern):
    //   0: (-1,-1,-1)  1: (+1,-1,-1)  2: (-1,+1,-1)  3: (+1,+1,-1)
    //   4: (-1,-1,+1)  5: (+1,-1,+1)  6: (-1,+1,+1)  7: (+1,+1,+1)
    //
    // 12 triangles (2 per face), outward-facing normals (CCW winding from outside)
    std::vector<Triangle> triangles_host = {
        // -Z face (z=-1): vertices 0,1,3,2
        Triangle(0, 3, 1), Triangle(0, 2, 3),
        // +Z face (z=+1): vertices 4,5,7,6
        Triangle(4, 5, 7), Triangle(4, 7, 6),
        // -Y face (y=-1): vertices 0,1,5,4
        Triangle(0, 1, 5), Triangle(0, 5, 4),
        // +Y face (y=+1): vertices 2,3,7,6
        Triangle(2, 7, 3), Triangle(2, 6, 7),
        // -X face (x=-1): vertices 0,2,6,4
        Triangle(0, 4, 6), Triangle(0, 6, 2),
        // +X face (x=+1): vertices 1,3,7,5
        Triangle(1, 3, 7), Triangle(1, 7, 5),
    };

    const int numVerts = 8;
    const int numTris = 12;

    // ========== 2. Build mesh topology (face adjacency) ==========
    MeshTopologyHost topo = MeshTopologyBuilder::BuildFromTriangles(numVerts, triangles_host);

    // ========== 3. Run KMeans patching ==========
    PatchingParams params;
    params.targetFacesPerPatch = 256; // 12 faces << 256, so expect ~1 patch
    params.maxIters = 50;

    PatchingResultHost patchResult;
    KMeansPatcher patcher;
    patcher.BuildPatches(topo, params, patchResult);

    const int numPatches = patchResult.numPatches;

    // ========== 4. Compute rest-space AABB per patch ==========
    std::vector<AABB> patchAABBs_host(numPatches);
    for (int p = 0; p < numPatches; p++) {
        Coord vmin(Real(1e30), Real(1e30), Real(1e30));
        Coord vmax(Real(-1e30), Real(-1e30), Real(-1e30));

        int faceBegin = patchResult.patchOffsets[p];
        int faceEnd = patchResult.patchOffsets[p + 1];
        for (int fi = faceBegin; fi < faceEnd; fi++) {
            int fid = patchResult.patchFaces[fi];
            const Triangle& tri = triangles_host[fid];
            for (int k = 0; k < 3; k++) {
                const Coord& v = vertices_host[tri[k]];
                vmin = Coord(std::min(vmin[0], v[0]), std::min(vmin[1], v[1]), std::min(vmin[2], v[2]));
                vmax = Coord(std::max(vmax[0], v[0]), std::max(vmax[1], v[1]), std::max(vmax[2], v[2]));
            }
        }
        patchAABBs_host[p].v0 = vmin;
        patchAABBs_host[p].v1 = vmax;
    }

    // ========== 5. Build edge topology ==========
    // Build edge list, triangle-to-edge mapping, edge-to-face adjacency
    std::unordered_map<uint64_t, int> edgeMap;
    edgeMap.reserve(numTris * 3);

    std::vector<Edge> edges_host;
    std::vector<Tri2Edg> triangleEdges_host(numTris);

    // Two-face adjacency per edge
    struct EdgeFaces { int f0 = -1; int f1 = -1; };
    std::vector<EdgeFaces> edgeFaces_tmp;

    for (int f = 0; f < numTris; f++) {
        const Triangle& tri = triangles_host[f];
        int verts[3] = { static_cast<int>(tri[0]), static_cast<int>(tri[1]), static_cast<int>(tri[2]) };

        for (int e = 0; e < 3; e++) {
            int va = verts[e];
            int vb = verts[(e + 1) % 3];
            uint64_t key = packEdgeKey(va, vb);

            auto it = edgeMap.find(key);
            int edgeId;
            if (it == edgeMap.end()) {
                edgeId = static_cast<int>(edges_host.size());
                edgeMap[key] = edgeId;
                edges_host.push_back(Edge(std::min(va, vb), std::max(va, vb)));
                edgeFaces_tmp.push_back(EdgeFaces{f, -1});
            } else {
                edgeId = it->second;
                if (edgeFaces_tmp[edgeId].f1 == -1) {
                    edgeFaces_tmp[edgeId].f1 = f;
                }
            }
            triangleEdges_host[f][e] = edgeId;
        }
    }

    const int numEdges = static_cast<int>(edges_host.size());

    std::vector<Edg2Tri> edgeAdjacentFaces_host(numEdges);
    for (int e = 0; e < numEdges; e++) {
        edgeAdjacentFaces_host[e][0] = edgeFaces_tmp[e].f0;
        edgeAdjacentFaces_host[e][1] = edgeFaces_tmp[e].f1;
    }

    // ========== 6. Build vertex-to-face adjacency (CSR) ==========
    std::vector<int> vertexFaceCounts(numVerts, 0);
    for (int f = 0; f < numTris; f++) {
        const Triangle& tri = triangles_host[f];
        for (int k = 0; k < 3; k++) {
            vertexFaceCounts[tri[k]]++;
        }
    }

    std::vector<int> vertexFaceOffsets_host(numVerts + 1, 0);
    for (int v = 0; v < numVerts; v++) {
        vertexFaceOffsets_host[v + 1] = vertexFaceOffsets_host[v] + vertexFaceCounts[v];
    }

    int totalVF = vertexFaceOffsets_host[numVerts];
    std::vector<int> vertexFaceIndices_host(totalVF);
    std::vector<int> cursor(numVerts, 0);
    for (int f = 0; f < numTris; f++) {
        const Triangle& tri = triangles_host[f];
        for (int k = 0; k < 3; k++) {
            int v = tri[k];
            int offset = vertexFaceOffsets_host[v] + cursor[v];
            vertexFaceIndices_host[offset] = f;
            cursor[v]++;
        }
    }

    // ========== 7. Upload to GPU ==========
    out.numVertices = numVerts;
    out.numTriangles = numTris;
    out.numPatches = numPatches;
    out.numEdges = numEdges;

    // Geometry
    CArray<Coord> vertices_ca(numVerts);
    for (int i = 0; i < numVerts; i++) vertices_ca[i] = vertices_host[i];
    out.vertices.assign(vertices_ca);

    CArray<Triangle> triangles_ca(numTris);
    for (int i = 0; i < numTris; i++) triangles_ca[i] = triangles_host[i];
    out.triangles.assign(triangles_ca);

    // Patches
    CArray<int> patchOffsets_ca(numPatches + 1);
    for (int i = 0; i <= numPatches; i++) patchOffsets_ca[i] = patchResult.patchOffsets[i];
    out.patchOffsets.assign(patchOffsets_ca);

    CArray<int> patchFaces_ca(patchResult.numFaces);
    for (int i = 0; i < patchResult.numFaces; i++) patchFaces_ca[i] = patchResult.patchFaces[i];
    out.patchFaces.assign(patchFaces_ca);

    CArray<AABB> patchAABBs_ca(numPatches);
    for (int i = 0; i < numPatches; i++) patchAABBs_ca[i] = patchAABBs_host[i];
    out.patchAABBs.assign(patchAABBs_ca);

    // Edge topology
    CArray<Edge> edges_ca(numEdges);
    for (int i = 0; i < numEdges; i++) edges_ca[i] = edges_host[i];
    out.edges.assign(edges_ca);

    CArray<Tri2Edg> triEdges_ca(numTris);
    for (int i = 0; i < numTris; i++) triEdges_ca[i] = triangleEdges_host[i];
    out.triangleEdges.assign(triEdges_ca);

    CArray<Edg2Tri> edgeFaces_ca(numEdges);
    for (int i = 0; i < numEdges; i++) edgeFaces_ca[i] = edgeAdjacentFaces_host[i];
    out.edgeAdjacentFaces.assign(edgeFaces_ca);

    // Vertex-face adjacency
    CArray<int> vfOffsets_ca(numVerts + 1);
    for (int i = 0; i <= numVerts; i++) vfOffsets_ca[i] = vertexFaceOffsets_host[i];
    out.vertexFaceOffsets.assign(vfOffsets_ca);

    CArray<int> vfIndices_ca(totalVF);
    for (int i = 0; i < totalVF; i++) vfIndices_ca[i] = vertexFaceIndices_host[i];
    out.vertexFaceIndices.assign(vfIndices_ca);

    // ========== 8. Build rest-space patch BVH ==========
    out.patchBVH.construct(out.patchAABBs);
}

// Explicit instantiation
template void GenerateUnitCubeMesh<DataType3f>(MeshTemplateData<DataType3f>& outTemplate);

} // namespace dyno
