// MeshTopologyBuilder.cpp
#include "MeshTopologyBuilder.h"

#include <unordered_map>
#include <queue>
#include <cstdint>
#include <cassert>
#include <algorithm>

namespace dyno {

namespace {

static inline uint64_t packEdgeKey(int a, int b)
{
    const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

struct EdgeRef
{
    int faceId = -1; // first face that saw this edge
    int edgeId = -1; // local edge: 0,1,2
};

static void ValidateTopology(const MeshTopologyHost& topo)
{
#ifndef NDEBUG
    const int F = topo.numFaces;
    assert(static_cast<int>(topo.faceTris.size()) == F);
    assert(static_cast<int>(topo.faceAdj.size()) == 3 * F);
    assert(static_cast<int>(topo.componentId.size()) == F);
    assert(static_cast<int>(topo.isIsolatedFace.size()) == F);

    // 1) adjacency symmetry
    for (int f = 0; f < F; ++f)
    {
        for (int e = 0; e < 3; ++e)
        {
            const int g = topo.faceAdj[3 * f + e];
            if (g == -1) continue;

            assert(g >= 0 && g < F);

            bool foundBack = false;
            for (int ge = 0; ge < 3; ++ge)
            {
                if (topo.faceAdj[3 * g + ge] == f)
                {
                    foundBack = true;
                    break;
                }
            }
            assert(foundBack);
        }
    }

    // 2) component range
    assert(topo.numComponents >= 0);
    for (int f = 0; f < F; ++f)
    {
        const int c = topo.componentId[f];
        assert(c >= 0 && c < topo.numComponents);
    }

    // 3) component size sum
    int sum = 0;
    assert(static_cast<int>(topo.componentSizes.size()) == topo.numComponents);
    for (int c = 0; c < topo.numComponents; ++c)
    {
        assert(topo.componentSizes[c] > 0);
        sum += topo.componentSizes[c];
    }
    assert(sum == F);

    // 4) isolated consistency: isolated => component size == 1
    for (int f = 0; f < F; ++f)
    {
        if (topo.isIsolatedFace[f])
        {
            const int c = topo.componentId[f];
            assert(topo.componentSizes[c] == 1);
        }
    }
#endif
}

} // anonymous namespace

MeshTopologyHost MeshTopologyBuilder::BuildFromTriangles(
    int numVertices,
    const std::vector<TopologyModule::Triangle>& vertexIndex)
{
    MeshTopologyHost topo;
    topo.numVertices = std::max(0, numVertices);
    topo.numFaces    = static_cast<int>(vertexIndex.size());
    topo.faceVerts    = vertexIndex;

    const int F = topo.numFaces;

    topo.faceAdj.assign(3 * F, -1);

    // -------- A) Build adjacency via undirected edge hash --------
    std::unordered_map<uint64_t, EdgeRef> edgeMap;
    edgeMap.reserve(static_cast<size_t>(F) * 3);

    for (int f = 0; f < F; ++f)
    {
        const auto& tri = topo.faceVerts[f];
        const int v0 = static_cast<int>(tri[0]);
        const int v1 = static_cast<int>(tri[1]);
        const int v2 = static_cast<int>(tri[2]);

        const int ev[3][2] = { {v0, v1}, {v1, v2}, {v2, v0} };

        for (int e = 0; e < 3; ++e)
        {
            const int a = ev[e][0];
            const int b = ev[e][1];

            if (a == b) continue; // degenerate edge

            const uint64_t key = packEdgeKey(a, b);

            auto it = edgeMap.find(key);
            if (it == edgeMap.end())
            {
                edgeMap.emplace(key, EdgeRef{ f, e });
            }
            else
            {
                const int otherF = it->second.faceId;
                const int otherE = it->second.edgeId;

                // Non-manifold handling: connect only first pair
                if (otherF >= 0 &&
                    topo.faceAdj[3 * otherF + otherE] == -1 &&
                    topo.faceAdj[3 * f + e] == -1)
                {
                    topo.faceAdj[3 * otherF + otherE] = f;
                    topo.faceAdj[3 * f + e]          = otherF;
                }
            }
        }
    }

    // -------- B) Connected components via BFS on face adjacency --------
    topo.componentId.assign(F, -1);
    topo.componentSizes.clear();
    topo.componentSizes.reserve(std::max(1, F / 64));

    int compCount = 0;
    std::queue<int> q;

    for (int f = 0; f < F; ++f)
    {
        if (topo.componentId[f] != -1) continue;

        const int cid = compCount++;
        topo.componentId[f] = cid;

        int compSize = 0;
        while (!q.empty()) q.pop();
        q.push(f);

        while (!q.empty())
        {
            const int cur = q.front();
            q.pop();
            ++compSize;

            for (int e = 0; e < 3; ++e)
            {
                const int nb = topo.faceAdj[3 * cur + e];
                if (nb == -1) continue;
                if (topo.componentId[nb] != -1) continue;

                topo.componentId[nb] = cid;
                q.push(nb);
            }
        }

        topo.componentSizes.push_back(compSize);
    }

    topo.numComponents = compCount;

    // -------- C) Mark isolated faces --------
    topo.isIsolatedFace.assign(F, 0);
    for (int f = 0; f < F; ++f)
    {
        const int cid = topo.componentId[f];
        topo.isIsolatedFace[f] = (topo.componentSizes[cid] == 1) ? 1 : 0;
    }

    ValidateTopology(topo);
    return topo;
}

} // dyno