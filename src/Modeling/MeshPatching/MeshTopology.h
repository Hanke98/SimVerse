#pragma once
#include <cstdint>
#include <vector>

#include <Module/TopologyModule.h>

namespace dyno {

// 每个 face 三条边的邻居 face id；边界用 -1
struct FaceAdjacencyHost
{
    std::vector<int> adj; // size = 3*numFaces, adj[3*f + e] = neighborFaceId or -1
};

struct MeshTopologyHost
{
    int numVertices = 0;
    int numFaces    = 0;

    // triangles：按 face 顺序存，每个 face 三个顶点索引
    std::vector<TopologyModule::Triangle> faceVerts; // size = numFaces

    // FaceAdjacencyHost faceAdj;
    std::vector<int> faceAdj;

    // 连通分量标签（按 face）
    std::vector<int> componentId; // size = numFaces, in [0..numComponents-1]
    int numComponents = 0;

    // 孤立 face（可按“度为0”或“单面分量”定义）
    std::vector<uint8_t> isIsolatedFace; // size = numFaces, 0/1

    // 可选：每个 component 的 face 数（Stage0 分 seed 用）
    std::vector<int> componentSizes; // size = numComponents
};

} // namespace dyno