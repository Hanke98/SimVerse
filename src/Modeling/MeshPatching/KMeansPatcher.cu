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

void KMeansPatcher::InitSeedsHost(
    const MeshTopologyHost& topo,
    const PatchingParams& params,
    std::vector<int>& outSeedFaces)
{
    outSeedFaces.clear();

    const int F  = topo.numFaces;
    if (F <= 0) return;

    const int Sp = std::max(1, params.targetFacesPerPatch);
    const int targetSeeds = std::max(1, (F + Sp - 1) / Sp);

    const int C = std::max(0, topo.numComponents);

    // mark used faces
    std::vector<uint8_t> used(static_cast<size_t>(F), 0);

    // 1) 每个 component 至少 1 个 seed：取该 component 的第一个 face
    if (C > 0 && static_cast<int>(topo.componentId.size()) == F)
    {
        std::vector<int> firstFaceOfComp(static_cast<size_t>(C), -1);
        for (int f = 0; f < F; ++f)
        {
            int cid = topo.componentId[f];
            if (cid >= 0 && cid < C && firstFaceOfComp[cid] == -1)
                firstFaceOfComp[cid] = f;
        }

        for (int cid = 0; cid < C; ++cid)
        {
            int f = firstFaceOfComp[cid];
            if (f >= 0 && f < F && !used[f])
            {
                used[f] = 1;
                outSeedFaces.push_back(f);
            }
        }
    }

    // 2) 补足到 targetSeeds：用 hash 方式从全局 faces 里挑（去重）
    // 说明：Milestone 3.1 先保证正确覆盖，不追求最优 seeds 分布
    const uint32_t rngSeed =
        (params.rngSeed == 0 ? 1234U : params.rngSeed);

    int attempt = 0;
    while (static_cast<int>(outSeedFaces.size()) < targetSeeds && attempt < F * 8)
    {
        uint32_t h = Hash32(static_cast<uint32_t>(attempt) ^ rngSeed);
        int f = static_cast<int>(h % static_cast<uint32_t>(F));
        if (!used[f])
        {
            used[f] = 1;
            outSeedFaces.push_back(f);
        }
        ++attempt;
    }

    // 兜底：极端情况下仍不足（比如 F 很小），按顺序补
    for (int f = 0; static_cast<int>(outSeedFaces.size()) < targetSeeds && f < F; ++f)
    {
        if (!used[f])
        {
            used[f] = 1;
            outSeedFaces.push_back(f);
        }
    }
}

// void KMeansPatcher::BuildPatches(
//     const MeshTopologyHost& topo,
//     const PatchingParams& params,
//     PatchingResultHost& out)
// {
//     out = PatchingResultHost{};

//     const int F = topo.numFaces;
//     out.numFaces = std::max(0, F);

//     if (F <= 0)
//     {
//         out.numPatches = 0;
//         return;
//     }

//     // -------------------------
//     // Stage0: host init seeds
//     // -------------------------
//     std::vector<int> seedFaces;
//     InitSeedsHost(topo, params, seedFaces); // seedFaces: host vector of seed face ids

//     const int P = static_cast<int>(seedFaces.size());
//     out.numPatches = P;

//     // 若 seeds 为空（理论不应发生），兜底给 1 个 seed
//     if (P <= 0)
//     {
//         seedFaces = {0};
//         out.numPatches = 1;
//     }

//     // -------------------------
//     // Copy inputs to device
//     // -------------------------
//     thrust::device_vector<int> d_faceAdj(topo.faceAdj.begin(), topo.faceAdj.end());
//     thrust::device_vector<int> d_facePatchId(static_cast<size_t>(F));
//     thrust::device_vector<int> d_seedFaces(seedFaces.begin(), seedFaces.end());

//     // Scalars on device
//     thrust::device_vector<int> d_changed(1);
//     thrust::device_vector<int> d_assigned(1);

//     const int threads = 256;
//     const int blocksF = (F + threads - 1) / threads;
//     const int blocksP = (out.numPatches + threads - 1) / threads;

//     // facePatchId = -1
//     KernelFillInt<<<blocksF, threads>>>(
//         thrust::raw_pointer_cast(d_facePatchId.data()), F, -1);
//     DYNO_CUDA_CHECK(cudaGetLastError());

//     // set seeds: facePatchId[seedFace] = seedId
//     KernelSetSeeds<<<(out.numPatches + threads - 1) / threads, threads>>>(
//         thrust::raw_pointer_cast(d_facePatchId.data()),
//         thrust::raw_pointer_cast(d_seedFaces.data()),
//         out.numPatches);
//     DYNO_CUDA_CHECK(cudaGetLastError());

//     // assignedCount = numSeeds（注意：若 seeds 有重复 face，会高估；我们已做去重）
//     {
//         int initAssigned = out.numPatches;
//         DYNO_CUDA_CHECK(cudaMemcpy(
//             thrust::raw_pointer_cast(d_assigned.data()),
//             &initAssigned,
//             sizeof(int),
//             cudaMemcpyHostToDevice));
//     }

//     // -------------------------
//     // Stage1: propagation (multi-source flood fill)
//     // -------------------------
//     const int maxPropIters = (params.maxPropIters > 0) ? params.maxPropIters : (1 << 30);

//     int h_assigned = 0;
//     for (int iter = 0; iter < maxPropIters; ++iter)
//     {
//         // changed = 0
//         DYNO_CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_changed.data()), 0, sizeof(int)));

//         KernelPropagate<<<blocksF, threads>>>(
//             thrust::raw_pointer_cast(d_faceAdj.data()),
//             thrust::raw_pointer_cast(d_facePatchId.data()),
//             F,
//             thrust::raw_pointer_cast(d_changed.data()),
//             thrust::raw_pointer_cast(d_assigned.data()));
//         DYNO_CUDA_CHECK(cudaGetLastError());

//         int h_changed = 0;
//         DYNO_CUDA_CHECK(cudaMemcpy(&h_changed, thrust::raw_pointer_cast(d_changed.data()),
//                                    sizeof(int), cudaMemcpyDeviceToHost));
//         DYNO_CUDA_CHECK(cudaMemcpy(&h_assigned, thrust::raw_pointer_cast(d_assigned.data()),
//                                    sizeof(int), cudaMemcpyDeviceToHost));

//         if (h_assigned >= F) break;
//         if (h_changed == 0) break;
//     }

//     // Milestone 3.1 默认要求所有 face 都被覆盖
//     // 若你担心坏数据，可以在这里做兜底：把未分配 face 变成独立 patch（Milestone 3.2 再加）
// // #ifndef NDEBUG
//     if (h_assigned < F)
//     {
//         printf("[KMeansPatcher] Warning: only %d/%d faces assigned.\n", h_assigned, F);
//     }
// // #endif

//     // -------------------------
//     // Stage2: build CSR (counts -> scan -> scatter), and maxPatchSize
//     // -------------------------
//     thrust::device_vector<int> d_patchCounts(static_cast<size_t>(out.numPatches));
//     KernelFillZeroInt<<<blocksP, threads>>>(
//         thrust::raw_pointer_cast(d_patchCounts.data()), out.numPatches);
//     DYNO_CUDA_CHECK(cudaGetLastError());
    
//     // count faces per patch
//     KernelCountPatches<<<blocksF, threads>>>(
//         thrust::raw_pointer_cast(d_facePatchId.data()),
//         F,
//         thrust::raw_pointer_cast(d_patchCounts.data()));
//     DYNO_CUDA_CHECK(cudaGetLastError());

//     // patchOffsets device (P+1)
//     thrust::device_vector<int> d_patchOffsets(static_cast<size_t>(out.numPatches + 1));

//     // Exclusive scan counts -> offsets[0..P-1]
//     size_t temp_bytes_scan = 0;
//     DYNO_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
//         nullptr, temp_bytes_scan,
//         thrust::raw_pointer_cast(d_patchCounts.data()),
//         thrust::raw_pointer_cast(d_patchOffsets.data()),
//         out.numPatches));

//     thrust::device_vector<uint8_t> d_temp_scan(temp_bytes_scan);
//     DYNO_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
//         thrust::raw_pointer_cast(d_temp_scan.data()), temp_bytes_scan,
//         thrust::raw_pointer_cast(d_patchCounts.data()),
//         thrust::raw_pointer_cast(d_patchOffsets.data()),
//         out.numPatches));

//     // set offsets[P] = F
//     KernelSetLastOffset<<<1,1>>>(
//         thrust::raw_pointer_cast(d_patchOffsets.data()),
//         out.numPatches,
//         F);
//     DYNO_CUDA_CHECK(cudaGetLastError());

//     // maxPatchSize = max(patchCounts)
//     thrust::device_vector<int> d_maxPatchSize(1);
//     size_t temp_bytes_reduce = 0;

//     DYNO_CUDA_CHECK(cub::DeviceReduce::Max(
//         nullptr, temp_bytes_reduce,
//         thrust::raw_pointer_cast(d_patchCounts.data()),
//         thrust::raw_pointer_cast(d_maxPatchSize.data()),
//         out.numPatches));

//     thrust::device_vector<uint8_t> d_temp_reduce(temp_bytes_reduce);
//     DYNO_CUDA_CHECK(cub::DeviceReduce::Max(
//         thrust::raw_pointer_cast(d_temp_reduce.data()), temp_bytes_reduce,
//         thrust::raw_pointer_cast(d_patchCounts.data()),
//         thrust::raw_pointer_cast(d_maxPatchSize.data()),
//         out.numPatches));

//     // Scatter faces into patchFaces using atomic offsets
//     thrust::device_vector<int> d_patchWrite(static_cast<size_t>(out.numPatches));
//     // copy offsets[0..P-1] -> patchWrite
//     DYNO_CUDA_CHECK(cudaMemcpy(
//         thrust::raw_pointer_cast(d_patchWrite.data()),
//         thrust::raw_pointer_cast(d_patchOffsets.data()),
//         sizeof(int) * out.numPatches,
//         cudaMemcpyDeviceToDevice));

//     thrust::device_vector<int> d_patchFaces(static_cast<size_t>(F)); // reorder the indices of faces by patches

//     KernelScatterFaces<<<blocksF, threads>>>(
//         thrust::raw_pointer_cast(d_facePatchId.data()),
//         F,
//         thrust::raw_pointer_cast(d_patchWrite.data()),
//         thrust::raw_pointer_cast(d_patchFaces.data()));
//     DYNO_CUDA_CHECK(cudaGetLastError());

//     // -------------------------
//     // Copy results back to host
//     // -------------------------
//     out.facePatchId.resize(static_cast<size_t>(F));
//     out.patchOffsets.resize(static_cast<size_t>(out.numPatches + 1));
//     out.patchFaces.resize(static_cast<size_t>(F));

//     thrust::copy(d_facePatchId.begin(), d_facePatchId.end(), out.facePatchId.begin());
//     thrust::copy(d_patchOffsets.begin(), d_patchOffsets.end(), out.patchOffsets.begin());
//     thrust::copy(d_patchFaces.begin(), d_patchFaces.end(), out.patchFaces.begin());

// #ifndef NDEBUG
//     int h_maxPatch = 0;
//     DYNO_CUDA_CHECK(cudaMemcpy(&h_maxPatch,
//                                thrust::raw_pointer_cast(d_maxPatchSize.data()),
//                                sizeof(int),
//                                cudaMemcpyDeviceToHost));
//     // 你可以把 maxPatchSize 临时打印出来，或后续把它加到 PatchingResultHost 里
//     // printf("[RxMeshKMeansPatcher] maxPatchSize=%d, P=%d, F=%d\n", h_maxPatch, out.numPatches, F);
// #endif
// }

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
        // 兜底：至少 1 个 patch
        P = 1;
    }
    out.numPatches = P;

    // -------------------------
    // Copy inputs to device
    // -------------------------
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

    // assignedCount = numSeeds（假设 seeds 去重）
    {
        int initAssigned = P;
        DYNO_CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_assigned.data()),
            &initAssigned,
            sizeof(int),
            cudaMemcpyHostToDevice));
    }

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

    // dist 复用：-1 表示未访问
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

        // 1) 找 boundary faces：邻居越界(-1) 或 neighbor 不在本 patch
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

        // boundary 为空：选 patch 内第一个 face
        if (q.empty())
        {
            const int f0 = res.patchFaces[begin];
            outSeedFaces[p] = f0;
            // 清理 dist（仅 f0 被改动）
            dist[f0] = -1;
            continue;
        }

        // 2) 多源 BFS 向内推进，只走 patch 内的邻接
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
                if (nb < 0) continue;
                if (res.facePatchId[nb] != p) continue;
                if (dist[nb] != -1) continue;

                dist[nb] = d + 1;
                q.push_back(nb);
            }
        }

        // 3) 从最后一层中选一个作为 seed（可复现）
        //    加 iter 是为了在多轮 Lloyd 中避免 seed 选择过于“冻结”
        const uint32_t h = Hash32(rngSeed ^ (static_cast<uint32_t>(p) * 0x9e3779b9U) ^ static_cast<uint32_t>(iter));
        outSeedFaces[p]  = lastLayer[static_cast<size_t>(h % static_cast<uint32_t>(lastLayer.size()))];

        // 4) 清理 dist：只清理访问过的点（q 包含了本 patch BFS 访问的所有 face）
        for (int f : q) dist[f] = -1;
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

        // Stage3 (CPU): update seeds
        std::vector<int> newSeeds;
        UpdateSeedsHost(topo, curRes, rngSeed, iter, newSeeds);

        // 收敛判定：seed 不变
        if (newSeeds.size() == seedFaces.size() &&
            std::equal(newSeeds.begin(), newSeeds.end(), seedFaces.begin()))
        {
            
            break;
        }

        seedFaces.swap(newSeeds);

        std::cout << "KMeansPatcher: iter " << iter << ": " << curRes.numPatches << " patches, " << maxPatchSize << " max patch size" << std::endl;
    }

    out = std::move(curRes);
}


} // namespace dyno