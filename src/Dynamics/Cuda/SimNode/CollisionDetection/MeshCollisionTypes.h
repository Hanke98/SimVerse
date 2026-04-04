#pragma once

#include <Array/Array.h>
#include <Array/Array2D.h>
#include <Module/TopologyModule.h>
#include "Primitive/Primitive3D.h"
#include "Topology/LinearBVH.h"
#include "Collision/CollisionData.h"

namespace dyno {

    // Stores one mesh template's geometry, patches, and topology on GPU
    template<typename TDataType>
    struct MeshTemplateData {
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using AABB = TAlignedBox3D<Real>;
        using Triangle = typename TopologyModule::Triangle;
        using Edge = typename TopologyModule::Edge;
        using Tri2Edg = typename TopologyModule::Tri2Edg;
        using Edg2Tri = typename TopologyModule::Edg2Tri;

        int numVertices = 0;
        int numTriangles = 0;
        int numPatches = 0;
        int numEdges = 0;

        // Rest-space geometry
        DArray<Coord>    vertices;
        DArray<Triangle> triangles;

        // Patching (CSR format)
        DArray<int>   patchOffsets;    // [numPatches + 1]
        DArray<int>   patchFaces;     // face indices per patch
        DArray<AABB>  patchAABBs;     // [numPatches] rest-space AABB per patch

        // Topology for narrow phase dedup
        DArray<Edge>    edges;
        DArray<Tri2Edg> triangleEdges;     // [numTriangles] 3 edge indices per triangle
        DArray<Edg2Tri> edgeAdjacentFaces; // [numEdges] 2 face indices per edge (-1 if boundary)

        // Vertex-to-face adjacency (CSR)
        DArray<int> vertexFaceOffsets;  // [numVertices + 1]
        DArray<int> vertexFaceIndices;  // face indices

        // Pre-built patch BVH (rest-space)
        LinearBVH<TDataType> patchBVH;
    };

    struct BodyPair {
        int env_id;
        int body_a;   // canonical: body_a < body_b
        int body_b;
    };

    enum PairType : int {
        PRIM_PRIM = 0,
        PRIM_MESH = 1,
        MESH_PRIM = 2,
        MESH_MESH = 3
    };

    struct PatchPair {
        int env_id;
        int body_a;
        int body_b;
        int patch_a;   // -1 for primitive side
        int patch_b;   // -1 for primitive side
        PairType type;
    };

} // namespace dyno
