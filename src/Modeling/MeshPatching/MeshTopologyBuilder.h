#pragma once
#include "MeshTopology.h"
#include <vector>

namespace dyno {

class MeshTopologyBuilder
{
public:
    static MeshTopologyHost BuildFromTriangles(
        int numVertices,
        const std::vector<TopologyModule::Triangle>& vertexIndex);
};

} // namespace dyno