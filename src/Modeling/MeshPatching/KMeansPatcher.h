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

    // Stage1-2: GPU assign + CSR 构建（Milestone 3.1 已通过的部分抽成函数）
    static void RunAssignAndCSR_GPU(
        const MeshTopologyHost& topo,
        const std::vector<int>& seedFaces,
        const PatchingParams& params,
        PatchingResultHost& out,
        int& outMaxPatchSize);

    // Stage3: CPU 更新 seeds（从 boundary 向内 BFS，取最后一层）
    static void UpdateSeedsHost(
        const MeshTopologyHost& topo,
        const PatchingResultHost& res,
        uint32_t rngSeed,
        int iter,
        std::vector<int>& outSeedFaces);

    // Stage4: CPU 插 seed（对大 patch 从 boundary 挑 k 个新 seed）
    static void AddSeedsHost(
        const MeshTopologyHost& topo,
        const PatchingResultHost& res,
        const PatchingParams& params,
        int iter,
        std::vector<int>& inoutSeedFaces);
};

} // namespace dyno