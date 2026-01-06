#pragma once

#include "MeshPatcher.h"

namespace dyno {

class KMeansPatcher final : public MeshPatcher
{
public:
    KMeansPatcher() = default;
    ~KMeansPatcher() override = default;
    void BuildPatches(
        const MeshTopologyHost& topo,
        const PatchingParams& params,
        PatchingResultHost& out) override;

private:
    // Stage0: Host seeds init (最小正确：每个 component 至少一个 seed)
    static void InitSeedsHost(
        const MeshTopologyHost& topo,
        const PatchingParams& params,
        std::vector<int>& outSeedFaces);
};

} // namespace dyno