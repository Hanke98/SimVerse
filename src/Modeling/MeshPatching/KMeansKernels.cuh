#pragma once

#include <cuda_runtime.h>

namespace dyno {

// #ifndef DYNO_CUDA_CHECK
// #define DYNO_CUDA_CHECK(call)                                                   \
//     do {                                                                        \
//         cudaError_t err__ = (call);                                             \
//         if (err__ != cudaSuccess) {                                             \
//             printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                \
//                    cudaGetErrorString(err__));                                  \
//             asm("trap;");                                                       \
//         }                                                                       \
//     } while (0)
// #endif

__global__ void KernelFillInt(int* a, int n, int v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__ void KernelFillZeroInt(int* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}

// facePatchId[seedFace] = seedId
__global__ void KernelSetSeeds(int* facePatchId, const int* seedFaces, int numSeeds)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < numSeeds)
    {
        int f = seedFaces[s];
        // 假设 seedFaces 已合法（0..F-1）
        facePatchId[f] = s;
    }
}

// Full-volume scanning propagation: For each face, if there is an existing patchId, try to propagate the patchId to unassigned neighbors.
// faceAdj: size=3*F, faceAdj[3*f+e] = neighbor face id or -1
__global__ void KernelPropagate(
    const int* faceAdj,
    int*       facePatchId,
    int        F,
    int*       changed,
    int*       assignedCount)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    int pid = facePatchId[f];
    if (pid == -1) return;

    // 尝试给 3 个邻居赋值
    #pragma unroll
    for (int e = 0; e < 3; ++e)
    {
        int nb = faceAdj[3 * f + e];
        if (nb < 0) continue;

        // 只在邻居未赋值时写入
        if (atomicCAS(&facePatchId[nb], -1, pid) == -1)
        {
            atomicExch(changed, 1);
            atomicAdd(assignedCount, 1);
        }
    }
}

__global__ void KernelCountPatches(const int* facePatchId, int F, int* patchCounts)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    int pid = facePatchId[f];
    if (pid >= 0)
    {
        atomicAdd(&patchCounts[pid], 1);
    }
}

// patchWrite 初始为 patchOffsets[0..P-1]，对每个 face 原子领取一个写入位置
__global__ void KernelScatterFaces(
    const int* facePatchId,
    int        F,
    int*       patchWrite,
    int*       patchFaces)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    int pid = facePatchId[f];
    // Milestone3.1 默认要求所有 face 都已分配
    int idx = atomicAdd(&patchWrite[pid], 1);
    patchFaces[idx] = f;
}

__global__ void KernelSetLastOffset(int* patchOffsets, int P, int totalFaces)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        patchOffsets[P] = totalFaces;
    }
}

} // namespace dyno