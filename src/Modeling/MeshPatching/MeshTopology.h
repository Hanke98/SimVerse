#pragma once
#include <cstdint>
#include <vector>

#include <Module/TopologyModule.h>

namespace dyno {

// // 每个 face 三条边的邻居 face id；边界用 -1
// struct FaceAdjacencyHost
// {
//     std::vector<int> adj; // size = 3*numFaces, adj[3*f + e] = neighborFaceId or -1
// };

struct MeshTopologyHost
{
    int numVertices = 0;
    int numFaces    = 0;

    // triangles：按 face 顺序存，每个 face 三个顶点索引
    std::vector<TopologyModule::Triangle> faceVerts; // size = numFaces

    // The list of adjacent faces for each face
    // Store according to the order of faces, storing all adjacent face IDs.
    std::vector<int> faceAdj;

    // Connected component label (by face)
    std::vector<int> componentId; // size = numFaces, in [0..numComponents-1]
    int numComponents = 0;

    // Isolated face label
    std::vector<uint8_t> isIsolatedFace; // size = numFaces, 0/1

    // The number of faces of each component
    std::vector<int> componentSizes; // size = numComponents
};

} // namespace dyno