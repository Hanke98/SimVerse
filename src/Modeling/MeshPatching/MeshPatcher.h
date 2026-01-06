#pragma once
#include "PatchingTypes.h"
#include "MeshTopology.h"

namespace dyno {

class MeshPatcher
{
public:
    virtual ~MeshPatcher() = default;

    virtual void BuildPatches(
        const MeshTopologyHost& topo,
        const PatchingParams& params,
        PatchingResultHost& out
    ) = 0;

    // virtual void BuildPatches(
    //     int numFaces,
    //     const PatchingParams& params,
    //     PatchingResultHost& out) = 0;
};

} // namespace dyno