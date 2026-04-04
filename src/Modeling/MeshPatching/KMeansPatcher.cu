#include "KMeansPatcher.h"
#include "KMeansKernels.cuh"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <vector>

#define DYNO_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d (%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        } \
    } while (0)

namespace dyno {

// 一个很轻量的可复现“伪随机”hash（避免 <random> 影响可移植性/编译）
static inline uint32_t Hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// void KMeansPatcher::InitSeedsHost(
//     const MeshTopologyHost& topo,
//     const PatchingParams& params,
//     std::vector<int>& outSeedFaces)
// {
//     outSeedFaces.clear();

//     const int F  = topo.numFaces;
//     if (F <= 0) return;

//     const int Sp = std::max(1, params.targetFacesPerPatch);
//     const int targetSeeds = std::max(1, (F + Sp - 1) / Sp);

//     const int C = std::max(0, topo.numComponents);

//     // mark used faces
//     std::vector<uint8_t> used(static_cast<size_t>(F), 0);

//     // every component has at least 1 seed: select the first face of the component
//     if (C > 0 && static_cast<int>(topo.componentId.size()) == F)
//     {
//         std::vector<int> firstFaceOfComp(static_cast<size_t>(C), -1);
//         for (int f = 0; f < F; ++f)
//         {
//             int cid = topo.componentId[f];
//             if (cid >= 0 && cid < C && firstFaceOfComp[cid] == -1)
//                 firstFaceOfComp[cid] = f;
//         }

//         for (int cid = 0; cid < C; ++cid)
//         {
//             int f = firstFaceOfComp[cid];
//             if (f >= 0 && f < F && !used[f])
//             {
//                 used[f] = 1;
//                 outSeedFaces.push_back(f);
//             }
//         }
//     }

//     // 2) 补足到 targetSeeds：用 hash 方式从全局 faces 里挑（去重）
//     const uint32_t rngSeed = (params.rngSeed == 0 ? 1234U : params.rngSeed);

//     int attempt = 0;
//     while (static_cast<int>(outSeedFaces.size()) < targetSeeds && attempt < F * 8)
//     {
//         uint32_t h = Hash32(static_cast<uint32_t>(attempt) ^ rngSeed);
//         int f = static_cast<int>(h % static_cast<uint32_t>(F));
//         if (!used[f])
//         {
//             used[f] = 1;
//             outSeedFaces.push_back(f);
//         }
//         ++attempt;
//     }

//     // 兜底：极端情况下仍不足（比如 F 很小），按顺序补
//     for (int f = 0; static_cast<int>(outSeedFaces.size()) < targetSeeds && f < F; ++f)
//     {
//         if (!used[f])
//         {
//             used[f] = 1;
//             outSeedFaces.push_back(f);
//         }
//     }
// }

void KMeansPatcher::InitSeedsHost(
    const MeshTopologyHost& topo,
    const PatchingParams& params,
    std::vector<int>& outSeedFaces)
{
    outSeedFaces.clear();

    const int F = topo.numFaces;
    if (F <= 0) return;

    const uint32_t rngSeed = (params.rngSeed == 0 ? 1234U : params.rngSeed);

    // 没有 component 信息则退化：仅按 isolated + 全局 hash
    const int C = std::max(0, topo.numComponents);
    const bool hasComp = (C > 0 &&
                          (int)topo.componentId.size() == F &&
                          (int)topo.componentSizes.size() == C &&
                          (int)topo.isIsolatedFace.size() == F);

    // mark used faces (avoid duplicate seeds)
    std::vector<uint8_t> used((size_t)F, 0);

    // ------------------------------------------------------------
    // Step 0) All isolated faces become seeds (may exceed targetSeeds)
    // ------------------------------------------------------------
    if ((int)topo.isIsolatedFace.size() == F)
    {
        for (int f = 0; f < F; ++f)
        {
            if (topo.isIsolatedFace[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
            }
        }
    }

    // targetSeeds = ceil(F / S_p)
    const int Sp = std::max(1, params.targetFacesPerPatch);
    const int targetSeeds = std::max(1, (F + Sp - 1) / Sp);

    // 若 isolated 已经 >= targetSeeds，直接返回（你允许超过 targetSeeds）
    if ((int)outSeedFaces.size() >= targetSeeds)
        return;

    // ------------------------------------------------------------
    // If no component info: fill remaining globally (excluding used)
    // ------------------------------------------------------------
    if (!hasComp)
    {
        int remaining = targetSeeds - (int)outSeedFaces.size();

        int attempt = 0;
        while (remaining > 0 && attempt < F * 16)
        {
            uint32_t h = Hash32((uint32_t)attempt ^ rngSeed);
            int f = (int)(h % (uint32_t)F);
            if (!used[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
                --remaining;
            }
            ++attempt;
        }

        for (int f = 0; remaining > 0 && f < F; ++f)
        {
            if (!used[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
                --remaining;
            }
        }
        return;
    }

    // ------------------------------------------------------------
    // Build isolatedCount per component and effective size
    // ------------------------------------------------------------
    std::vector<int> isolatedCount((size_t)C, 0);
    for (int f = 0; f < F; ++f)
    {
        int cid = topo.componentId[f];
        if (cid < 0 || cid >= C) continue;
        if (topo.isIsolatedFace[f]) isolatedCount[cid]++;
    }

    std::vector<int> effectiveSize((size_t)C, 0);
    for (int cid = 0; cid < C; ++cid)
    {
        int eff = topo.componentSizes[cid] - isolatedCount[cid];
        effectiveSize[cid] = std::max(0, eff);
    }

    // 有效 components（还有可增长的非-isolated faces）
    std::vector<int> effComps;
    effComps.reserve((size_t)C);
    for (int cid = 0; cid < C; ++cid)
        if (effectiveSize[cid] > 0)
            effComps.push_back(cid);

    if (effComps.empty())
    {
        // 所有 faces 要么 isolated，要么 component 信息异常；兜底补 global
        int remaining = targetSeeds - (int)outSeedFaces.size();
        for (int f = 0; remaining > 0 && f < F; ++f)
        {
            if (!used[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
                --remaining;
            }
        }
        return;
    }

    // 还需要多少 seeds（不包含 isolated，isolated 已经加入）
    int remaining = targetSeeds - (int)outSeedFaces.size();
    if (remaining <= 0) return;

    // ------------------------------------------------------------
    // Step 1) One base seed per effective component (if budget allows)
    // Base seed = minimum faceId within component that is NOT isolated
    // ------------------------------------------------------------
    // 若 remaining < effComps.size()，按 effectiveSize 从大到小选
    std::vector<int> compOrder = effComps;
    std::sort(compOrder.begin(), compOrder.end(),
              [&](int a, int b) { return effectiveSize[a] > effectiveSize[b]; });

    int baseCount = std::min((int)compOrder.size(), remaining);

    // 记录 base seed 已分配的 component
    std::vector<uint8_t> hasBase((size_t)C, 0);

    // 预先求每个 component 的最小非-isolated face（一次扫描）
    std::vector<int> minNonIsoFace((size_t)C, -1);
    for (int f = 0; f < F; ++f)
    {
        int cid = topo.componentId[f];
        if (cid < 0 || cid >= C) continue;
        if (topo.isIsolatedFace[f]) continue;
        if (minNonIsoFace[cid] == -1 || f < minNonIsoFace[cid])
            minNonIsoFace[cid] = f;
    }

    for (int i = 0; i < baseCount; ++i)
    {
        int cid = compOrder[i];
        int f0 = minNonIsoFace[cid];

        // 理论上 effectiveSize>0 则 f0 必存在；仍做兜底
        if (f0 >= 0 && f0 < F && !used[f0])
        {
            used[f0] = 1;
            outSeedFaces.push_back(f0);
            hasBase[cid] = 1;
            --remaining;
        }
        else
        {
            // 兜底：顺扫找一个该 component 内未用且非-isolated 的 face
            for (int f = 0; f < F; ++f)
            {
                if (topo.componentId[f] != cid) continue;
                if (topo.isIsolatedFace[f]) continue;
                if (used[f]) continue;
                used[f] = 1;
                outSeedFaces.push_back(f);
                hasBase[cid] = 1;
                --remaining;
                break;
            }
        }
        if (remaining <= 0) return;
    }

    // ------------------------------------------------------------
    // Step 2) Distribute remaining seeds proportionally to effectiveSize
    // Only among effective components (effectiveSize>0)
    // ------------------------------------------------------------
    // 计算 totalEffective
    long long totalEff = 0;
    for (int cid : effComps) totalEff += (long long)effectiveSize[cid];

    if (totalEff <= 0)
    {
        // 兜底：全局补
        for (int f = 0; remaining > 0 && f < F; ++f)
        {
            if (!used[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
                --remaining;
            }
        }
        return;
    }

    std::vector<int> extra((size_t)C, 0);

    struct FracItem { int cid; double frac; };
    std::vector<FracItem> fracs;
    fracs.reserve(effComps.size());

    int sumFloor = 0;
    for (int cid : effComps)
    {
        double exact = (double)remaining * (double)effectiveSize[cid] / (double)totalEff;
        int e = (int)std::floor(exact);
        extra[cid] = e;
        sumFloor += e;
        fracs.push_back({cid, exact - (double)e});
    }

    int leftover = remaining - sumFloor;
    std::sort(fracs.begin(), fracs.end(),
              [](const FracItem& a, const FracItem& b) { return a.frac > b.frac; });

    for (int i = 0; i < leftover && i < (int)fracs.size(); ++i)
        extra[fracs[i].cid]++;

    // ------------------------------------------------------------
    // Step 3) Pick extra seeds inside each component (hash retry + fallback scan)
    // Excluding isolated faces and used faces
    // ------------------------------------------------------------
    for (int cid : effComps)
    {
        int need = extra[cid];
        if (need <= 0) continue;

        // hash 重试：在 [0..F) 上采样，过滤 componentId & non-iso & unused
        int tries = 0;
        const int maxTries = std::max(64, effectiveSize[cid] * 8);

        while (need > 0 && tries < maxTries)
        {
            uint32_t h = Hash32(rngSeed ^ ((uint32_t)cid * 0x9e3779b9U) ^ (uint32_t)tries);
            int f = (int)(h % (uint32_t)F);

            if (topo.componentId[f] != cid) { ++tries; continue; }
            if (topo.isIsolatedFace[f])     { ++tries; continue; }
            if (used[f])                    { ++tries; continue; }

            used[f] = 1;
            outSeedFaces.push_back(f);
            --need;
            ++tries;
        }

        // 兜底：顺扫该 component 内的非-isolated face
        for (int f = 0; need > 0 && f < F; ++f)
        {
            if (topo.componentId[f] != cid) continue;
            if (topo.isIsolatedFace[f]) continue;
            if (used[f]) continue;

            used[f] = 1;
            outSeedFaces.push_back(f);
            --need;
        }
    }

    // ------------------------------------------------------------
    // Final safety: if still short, fill globally with any unused faces
    // ------------------------------------------------------------
    for (int f = 0; (int)outSeedFaces.size() < targetSeeds && f < F; ++f)
    {
        if (!used[f])
        {
            used[f] = 1;
            outSeedFaces.push_back(f);
        }
    }
}

void KMeansPatcher::RunAssignAndCSR_GPU(
    const MeshTopologyHost& topo,
    const std::vector<int>& seedFaces,
    const PatchingParams& params,
    PatchingResultHost& out,
    int& outMaxPatchSize)
{
    out = PatchingResultHost{};
    outMaxPatchSize = 0;

    const int F = topo.numFaces;
    out.numFaces = std::max(0, F);
    if (F <= 0)
    {
        out.numPatches = 0;
        return;
    }

    int P = static_cast<int>(seedFaces.size());
    if (P <= 0)
    {
        // At least 1 patch
        P = 1;
    }
    out.numPatches = P;

    // Copy inputs to device
    thrust::device_vector<int> d_faceAdj(topo.faceAdj.begin(), topo.faceAdj.end());
    thrust::device_vector<int> d_facePatchId(static_cast<size_t>(F));
    thrust::device_vector<int> d_seedFaces(seedFaces.begin(), seedFaces.end());

    // Scalars on device
    thrust::device_vector<int> d_changed(1);
    thrust::device_vector<int> d_assigned(1);

    const int threads = 256;
    const int blocksF = (F + threads - 1) / threads;
    const int blocksP = (P + threads - 1) / threads;

    // facePatchId = -1
    KernelFillInt<<<blocksF, threads>>>(
        thrust::raw_pointer_cast(d_facePatchId.data()), F, -1);
    DYNO_CUDA_CHECK(cudaGetLastError());

    // set seeds: facePatchId[seedFace] = seedId
    KernelSetSeeds<<<(P + threads - 1) / threads, threads>>>(
        thrust::raw_pointer_cast(d_facePatchId.data()),
        thrust::raw_pointer_cast(d_seedFaces.data()),
        P);
    DYNO_CUDA_CHECK(cudaGetLastError());

    // assignedCount = numSeeds
    int initAssigned = P;
    DYNO_CUDA_CHECK(cudaMemcpy(
        thrust::raw_pointer_cast(d_assigned.data()),
        &initAssigned,
        sizeof(int),
        cudaMemcpyHostToDevice));
    
    // -------------------------
    // Stage1: propagation (multi-source flood fill)
    // -------------------------
    const int maxPropIters = (params.maxPropIters > 0) ? params.maxPropIters : (1 << 30);

    int h_assigned = 0;
    for (int iter = 0; iter < maxPropIters; ++iter)
    {
        DYNO_CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_changed.data()), 0, sizeof(int)));

        KernelPropagate<<<blocksF, threads>>>(
            thrust::raw_pointer_cast(d_faceAdj.data()),
            thrust::raw_pointer_cast(d_facePatchId.data()),
            F,
            thrust::raw_pointer_cast(d_changed.data()),
            thrust::raw_pointer_cast(d_assigned.data()));
        DYNO_CUDA_CHECK(cudaGetLastError());

        int h_changed = 0;
        DYNO_CUDA_CHECK(cudaMemcpy(&h_changed, thrust::raw_pointer_cast(d_changed.data()),
                                   sizeof(int), cudaMemcpyDeviceToHost));
        DYNO_CUDA_CHECK(cudaMemcpy(&h_assigned, thrust::raw_pointer_cast(d_assigned.data()),
                                   sizeof(int), cudaMemcpyDeviceToHost));

        if (h_assigned >= F) break;
        if (h_changed == 0) break;
    }

#ifndef NDEBUG
    if (h_assigned < F)
    {
        printf("[KMeansPatcher] Warning: only %d/%d faces assigned.\n", h_assigned, F);
    }
#endif

    // Check for unassigned faces (facePatchId == -1)
    thrust::host_vector<int> h_facePatchId = d_facePatchId;
    std::vector<int> unassignedFaces;
    for (int i = 0; i < F; ++i)
    {
        if (h_facePatchId[i] == -1) // Unassigned face
        {
            unassignedFaces.push_back(i);
        }
    }

    // If there are unassigned faces, treat them as new seeds
    if (!unassignedFaces.empty())
    {
        printf("[KMeansPatcher] Warning: Found %zu unassigned faces after propagation.\n",
            unassignedFaces.size());

        std::vector<int> newSeeds(seedFaces.begin(), seedFaces.end());

        // Add ONE seed per uncovered component (preferred)
        if ((int)topo.componentId.size() == F && topo.numComponents > 0)
        {
            std::vector<int> picked((size_t)topo.numComponents, -1);

            // pick the first unassigned face per component
            for (int f : unassignedFaces)
            {
                int cid = topo.componentId[f];
                if (cid < 0 || cid >= topo.numComponents) continue;
                if (picked[cid] == -1) picked[cid] = f;
            }

            int added = 0;
            for (int cid = 0; cid < topo.numComponents; ++cid)
            {
                if (picked[cid] != -1)
                {
                    newSeeds.push_back(picked[cid]);
                    ++added;
                }
            }

            printf("[KMeansPatcher] Adding %d new seeds (one per uncovered component).\n", added);
        }
        else
        {
            // Fallback: no component labels; add all unassigned faces as seeds
            // (should be rare; still safe)
            newSeeds.insert(newSeeds.end(), unassignedFaces.begin(), unassignedFaces.end());
            printf("[KMeansPatcher] topo.componentId unavailable, adding all unassigned faces as seeds.\n");
        }

        // Re-run assignment with the new seeds
        RunAssignAndCSR_GPU(topo, newSeeds, params, out, outMaxPatchSize);

        return;
    }

    // -------------------------
    // Stage2: build CSR (counts -> scan -> scatter), and maxPatchSize
    // -------------------------
    thrust::device_vector<int> d_patchCounts(static_cast<size_t>(P));
    KernelFillZeroInt<<<blocksP, threads>>>(
        thrust::raw_pointer_cast(d_patchCounts.data()), P);
    DYNO_CUDA_CHECK(cudaGetLastError());

    KernelCountPatches<<<blocksF, threads>>>(
        thrust::raw_pointer_cast(d_facePatchId.data()),
        F,
        thrust::raw_pointer_cast(d_patchCounts.data()));
    DYNO_CUDA_CHECK(cudaGetLastError());

    // patchOffsets device (P+1)
    thrust::device_vector<int> d_patchOffsets(static_cast<size_t>(P + 1));

    // Exclusive scan counts -> offsets[0..P-1]
    size_t temp_bytes_scan = 0;
    DYNO_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes_scan,
        thrust::raw_pointer_cast(d_patchCounts.data()),
        thrust::raw_pointer_cast(d_patchOffsets.data()),
        P));

    thrust::device_vector<uint8_t> d_temp_scan(temp_bytes_scan);
    DYNO_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        thrust::raw_pointer_cast(d_temp_scan.data()), temp_bytes_scan,
        thrust::raw_pointer_cast(d_patchCounts.data()),
        thrust::raw_pointer_cast(d_patchOffsets.data()),
        P));

    KernelSetLastOffset<<<1, 1>>>(
        thrust::raw_pointer_cast(d_patchOffsets.data()),
        P,
        F);
    DYNO_CUDA_CHECK(cudaGetLastError());

    // maxPatchSize = max(patchCounts)
    thrust::device_vector<int> d_maxPatchSize(1);
    size_t temp_bytes_reduce = 0;

    DYNO_CUDA_CHECK(cub::DeviceReduce::Max(
        nullptr, temp_bytes_reduce,
        thrust::raw_pointer_cast(d_patchCounts.data()),
        thrust::raw_pointer_cast(d_maxPatchSize.data()),
        P));

    thrust::device_vector<uint8_t> d_temp_reduce(temp_bytes_reduce);
    DYNO_CUDA_CHECK(cub::DeviceReduce::Max(
        thrust::raw_pointer_cast(d_temp_reduce.data()), temp_bytes_reduce,
        thrust::raw_pointer_cast(d_patchCounts.data()),
        thrust::raw_pointer_cast(d_maxPatchSize.data()),
        P));

    // Scatter faces into patchFaces using atomic offsets
    thrust::device_vector<int> d_patchWrite(static_cast<size_t>(P));
    DYNO_CUDA_CHECK(cudaMemcpy(
        thrust::raw_pointer_cast(d_patchWrite.data()),
        thrust::raw_pointer_cast(d_patchOffsets.data()),
        sizeof(int) * P,
        cudaMemcpyDeviceToDevice));

    thrust::device_vector<int> d_patchFaces(static_cast<size_t>(F));

    KernelScatterFaces<<<blocksF, threads>>>(
        thrust::raw_pointer_cast(d_facePatchId.data()),
        F,
        thrust::raw_pointer_cast(d_patchWrite.data()),
        thrust::raw_pointer_cast(d_patchFaces.data()));
    DYNO_CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    out.facePatchId.resize(static_cast<size_t>(F));
    out.patchOffsets.resize(static_cast<size_t>(P + 1));
    out.patchFaces.resize(static_cast<size_t>(F));

    thrust::copy(d_facePatchId.begin(), d_facePatchId.end(), out.facePatchId.begin());
    thrust::copy(d_patchOffsets.begin(), d_patchOffsets.end(), out.patchOffsets.begin());
    thrust::copy(d_patchFaces.begin(), d_patchFaces.end(), out.patchFaces.begin());

    DYNO_CUDA_CHECK(cudaMemcpy(&outMaxPatchSize,
                               thrust::raw_pointer_cast(d_maxPatchSize.data()),
                               sizeof(int),
                               cudaMemcpyDeviceToHost));
}

void KMeansPatcher::UpdateSeedsHost(
    const MeshTopologyHost& topo,
    const PatchingResultHost& res,
    uint32_t rngSeed,
    int iter,
    std::vector<int>& outSeedFaces)
{
    const int F = topo.numFaces;
    const int P = res.numPatches;

    outSeedFaces.assign(static_cast<size_t>(P), -1);

    if (F <= 0 || P <= 0) return;

    // dist[f] = distance to the boundary within the patch; -1 = unvisited
    std::vector<int> dist(static_cast<size_t>(F), -1);
    std::vector<int> q;
    q.reserve(1024);

    std::vector<int> lastLayer;
    lastLayer.reserve(256);

    for (int p = 0; p < P; ++p)
    {
        const int begin = res.patchOffsets[p];
        const int end   = res.patchOffsets[p + 1];

        if (begin >= end)
        {
            outSeedFaces[p] = 0; // it should not happen
            continue;
        }

        q.clear();
        lastLayer.clear();

        // Find boundary faces
        // Neighbors are out of bounds (-1) or neighbors are not in this patch
        for (int i = begin; i < end; ++i)
        {
            const int f = res.patchFaces[i];

            bool isBoundary = false;
            for (int e = 0; e < 3; ++e)
            {
                const int nb = topo.faceAdj[3 * f + e];
                if (nb < 0 || res.facePatchId[nb] != p)
                {
                    isBoundary = true;
                    break;
                }
            }

            if (isBoundary)
            {
                dist[f] = 0;
                q.push_back(f);
            }
        }

        // If the boundary is empty: select the first face within the patch
        if (q.empty())
        {
            const int f0 = res.patchFaces[begin];
            outSeedFaces[p] = f0;
            dist[f0] = -1;
            continue;
        }

        // Multi-source BFS moves inward
        // Only traversing adjacents within the patch
        int head = 0;
        int maxD = 0;

        while (head < static_cast<int>(q.size()))
        {
            const int cur = q[head++];
            const int d   = dist[cur];

            if (d > maxD)
            {
                maxD = d;
                lastLayer.clear();
            }
            if (d == maxD)
            {
                lastLayer.push_back(cur);
            }

            for (int e = 0; e < 3; ++e)
            {
                const int nb = topo.faceAdj[3 * cur + e];
                if (nb < 0) continue; // not a neighbor
                if (res.facePatchId[nb] != p) continue; // not in the same patch
                if (dist[nb] != -1) continue; // already visited

                // Otherwise, visit it
                dist[nb] = d + 1;
                q.push_back(nb);
            }
        }

        // Select one from the last layer as the seed
        // Adding iter is to avoid the seed selection from being too "frozen" in multiple rounds of Lloyd
        const uint32_t h = Hash32(rngSeed ^ (static_cast<uint32_t>(p) * 0x9e3779b9U) ^ static_cast<uint32_t>(iter));
        outSeedFaces[p]  = lastLayer[static_cast<size_t>(h % static_cast<uint32_t>(lastLayer.size()))];
    }
}

void KMeansPatcher::AddSeedsHost(
    const MeshTopologyHost& topo,
    const PatchingResultHost& res,
    const PatchingParams& params,
    int iter,
    std::vector<int>& inoutSeedFaces)
{
    const int F  = topo.numFaces;
    const int P  = res.numPatches;
    const int Sp = std::max(1, params.targetFacesPerPatch);
    if (F <= 0 || P <= 0) return;

    // Mark the existing seed to avoid inserting the same face repeatedly
    std::vector<uint8_t> used(static_cast<size_t>(F), 0);
    for (int f : inoutSeedFaces)
    {
        if (f >= 0 && f < F) used[f] = 1;
    }

    std::vector<int> boundary;
    boundary.reserve(512);

    const uint32_t rngSeed = (params.rngSeed == 0 ? 1234U : params.rngSeed);

    for (int p = 0; p < P; ++p)
    {
        const int begin = res.patchOffsets[p];
        const int end   = res.patchOffsets[p + 1];
        const int sz    = end - begin;

        if (sz <= Sp) continue;

        // k = ceil(sz/Sp) - 1：k is the number of boundary faces to add as new seeds
        const int k = std::max(1, ((sz + Sp - 1) / Sp) - 1);

        boundary.clear();

        // 1) 收集 boundary faces
        for (int i = begin; i < end; ++i)
        {
            const int f = res.patchFaces[i];

            bool isBoundary = false;
            for (int e = 0; e < 3; ++e)
            {
                const int nb = topo.faceAdj[3 * f + e];
                if (nb < 0 || res.facePatchId[nb] != p)
                {
                    isBoundary = true;
                    break;
                }
            }
            if (isBoundary) boundary.push_back(f);
        }

        if (boundary.empty())
        {
            boundary.reserve((size_t)sz);
            for (int i = begin; i < end; ++i) boundary.push_back(res.patchFaces[i]);
        }

        // Select k new seeds from the boundary
        // int added = 0;
        const int maxTry = std::max(16, (int)boundary.size() * 4);

        for (int j = 0; j < k && (int)inoutSeedFaces.size() < F; ++j)
        {
            int chosen = -1;

            for (int t = 0; t < maxTry; ++t)
            {
                const uint32_t h = Hash32(rngSeed ^ (uint32_t)p * 0x9e3779b9U ^ (uint32_t)iter ^ (uint32_t)j ^ (uint32_t)t);
                const int idx = (int)(h % (uint32_t)boundary.size());
                const int f = boundary[idx];

                if (f < 0 || f >= F) continue; // invalid
                if (used[f]) continue; // already used as a seed

                chosen = f;
                break;
            }

            // if still cannot find a seed, select one from the whole boundary list
            if (chosen == -1)
            {
                for (int f : boundary)
                {
                    if (f >= 0 && f < F && !used[f])
                    {
                        chosen = f;
                        break;
                    }
                }
            }

            if (chosen != -1)
            {
                used[chosen] = 1;
                inoutSeedFaces.push_back(chosen);
                // ++added;
            }
        }

        // (void)added;
    }
}

void KMeansPatcher::BuildPatches(
    const MeshTopologyHost& topo,
    const PatchingParams& params,
    PatchingResultHost& out)
{
    out = PatchingResultHost{};

    const int F = topo.numFaces;
    out.numFaces = std::max(0, F);

    if (F <= 0)
    {
        out.numPatches = 0;
        return;
    }

    const int Sp = std::max(1, params.targetFacesPerPatch);
    const int seedAddPeriod = params.seedAddPeriod;

    // -------------------------
    // Stage0: init seeds
    // -------------------------
    std::vector<int> seedFaces;
    InitSeedsHost(topo, params, seedFaces);

    if (seedFaces.empty())
        seedFaces.push_back(0);

    // Lloyd loop (Milestone 3.2)
    const int maxIters = std::max(1, params.maxIters);
    const uint32_t rngSeed = (params.rngSeed == 0 ? 1234U : params.rngSeed);

    PatchingResultHost curRes;
    int maxPatchSize = 0;

    for (int iter = 0; iter < maxIters; ++iter)
    {
        // Stage1-2 (GPU): assign + CSR
        RunAssignAndCSR_GPU(topo, seedFaces, params, curRes, maxPatchSize);

        if (maxPatchSize <= Sp)
        {
            std::cout << "KMeansPatcher: iter " << iter << ": " << curRes.numPatches << " patches, " 
                  << maxPatchSize << " max patch size" << std::endl;
            std::cout << "KMeansPatcher: converged with target " << Sp << std::endl;
            break;
        }

        // Stage3 (CPU): update seeds
        std::vector<int> newSeeds;
        UpdateSeedsHost(topo, curRes, rngSeed, iter, newSeeds);

        // // Coverage：seed unchanged
        // if (newSeeds.size() == seedFaces.size() &&
        //     std::equal(newSeeds.begin(), newSeeds.end(), seedFaces.begin()))
        // {
            
        //     break;
        // }

        // Stage4 (CPU): every seedAddPeriod iters, add seeds for big patches
        if (seedAddPeriod > 0 && ((iter + 1) % seedAddPeriod == 0))
        {
            AddSeedsHost(topo, curRes, params, iter, newSeeds);
        }

        seedFaces.swap(newSeeds);

        
    }

    out = std::move(curRes);
}


} // namespace dyno