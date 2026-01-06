#include "MortonChunkPatcher.h"

#include <algorithm> // std::max
#include <cassert>

namespace dyno {

void MortonChunkPatcher::BuildPatches(
    const MeshTopologyHost& topo,
    const PatchingParams& params,
    PatchingResultHost& out)
{
    out = PatchingResultHost{}; // reset

    out.numFaces = std::max(0, topo.numFaces);

    const int sp = std::max(1, params.targetFacesPerPatch);
    const int n  = out.numFaces;

    // ceil(n / sp)
    out.numPatches = (n == 0) ? 0 : ((n + sp - 1) / sp);

    out.facePatchId.resize(n);
    out.patchFaces.resize(n);
    out.patchOffsets.resize(out.numPatches + 1, 0);

    // offsets: p*sp, last = n
    for (int p = 0; p < out.numPatches; ++p)
    {
        out.patchOffsets[p] = p * sp;
    }
    out.patchOffsets[out.numPatches] = n;

    // patchFaces = 0..n-1, facePatchId = f/sp
    for (int f = 0; f < n; ++f)
    {
        out.patchFaces[f]  = f;
        out.facePatchId[f] = f / sp;
        // 防御性断言：facePatchId 不应越界
        assert(out.facePatchId[f] >= 0 && out.facePatchId[f] < out.numPatches);
    }
}

} // namespace dyno