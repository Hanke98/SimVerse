#pragma once
#include <vector>
#include <cstdint>

namespace dyno {

struct PatchingParams
{
    int targetFacesPerPatch = 256;   // S_p
    int maxIters            = 50;
    int seedAddPeriod       = 5;     // Period of adding seed faces
    uint32_t rngSeed        = 1234;

    int maxPropIters        = 1<<30;
};

struct PatchingResultHost
{
    int numFaces   = 0;
    int numPatches = 0;

    // face -> patchId
    std::vector<int> facePatchId;

    // CSR: patchOffsets size = numPatches+1; patchFaces size = numFaces
    std::vector<int> patchOffsets;
    std::vector<int> patchFaces;

    // int maxPatchSize = 0;
    // int itersUsed    = 0;
};

} // namespace dyno