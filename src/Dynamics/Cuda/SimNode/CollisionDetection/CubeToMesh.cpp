#include "CubeToMesh.h"
#include "MeshPatching/MeshTopologyBuilder.h"
#include "MeshPatching/KMeansPatcher.h"
#include "MeshPatching/PatchingTypes.h"
#include "Topology/TriangleSet.h"

#include <vector>
#include <algorithm>

namespace dyno {

template<typename TDataType>
void GenerateUnitCubeMesh(MeshTemplateData<TDataType>& out)
{
    using Real = typename TDataType::Real;
    using Coord = typename TDataType::Coord;
    using AABB = TAlignedBox3D<Real>;
    using Triangle = typename TopologyModule::Triangle;

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

    // ========== 2. Build TriangleSet template geometry/topology ==========
    TriangleSet<TDataType> cubeTriSet;
    cubeTriSet.setPoints(vertices_host);
    cubeTriSet.setTriangles(triangles_host);
    cubeTriSet.update();

    // ========== 3. Build mesh topology (face adjacency) ==========
    MeshTopologyHost topo = MeshTopologyBuilder::BuildFromTriangles(numVerts, triangles_host);

    // ========== 4. Run KMeans patching ==========
    PatchingParams params;
    params.targetFacesPerPatch = 256; // 12 faces << 256, so expect ~1 patch
    params.maxIters = 50;

    PatchingResultHost patchResult;
    KMeansPatcher patcher;
    patcher.BuildPatches(topo, params, patchResult);

    const int numPatches = patchResult.numPatches;

    // ========== 5. Compute rest-space AABB per patch ==========
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

    // ========== 6. Extract topology from TriangleSet ==========
    const int numEdges = static_cast<int>(cubeTriSet.edgeIndices().size());
    CArrayList<int> vertex2TriangleHost;
    vertex2TriangleHost.assign(cubeTriSet.vertex2Triangle());

    std::vector<int> vertexFaceOffsets_host(numVerts + 1, 0);
    for (int v = 0; v < numVerts; ++v)
        vertexFaceOffsets_host[v] = static_cast<int>(vertex2TriangleHost.index()[v]);
    vertexFaceOffsets_host[numVerts] = static_cast<int>(vertex2TriangleHost.elements().size());

    const int totalVF = static_cast<int>(vertex2TriangleHost.elements().size());
    std::vector<int> vertexFaceIndices_host(totalVF, 0);
    for (int i = 0; i < totalVF; ++i)
        vertexFaceIndices_host[i] = vertex2TriangleHost.elements()[i];

    // ========== 7. Upload to GPU ==========
    out.numVertices = numVerts;
    out.numTriangles = numTris;
    out.numPatches = numPatches;
    out.numEdges = numEdges;

    // Geometry/topology come from TriangleSet
    out.vertices.assign(cubeTriSet.getPoints());
    out.triangles.assign(cubeTriSet.triangleIndices());

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

    out.edges.assign(cubeTriSet.edgeIndices());
    out.triangleEdges.assign(cubeTriSet.triangle2Edge());
    out.edgeAdjacentFaces.assign(cubeTriSet.edge2Triangle());

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
