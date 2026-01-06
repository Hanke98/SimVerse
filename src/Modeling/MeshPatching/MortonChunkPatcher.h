#pragma once

#include "MeshPatcher.h"

namespace dyno {

class MortonChunkPatcher final : public MeshPatcher
{
public:
    MortonChunkPatcher() = default;
    ~MortonChunkPatcher() override = default;

    void BuildPatches(
        const MeshTopologyHost& topo,
        const PatchingParams& params,
        PatchingResultHost& out) override;
};

} // namespace dyno