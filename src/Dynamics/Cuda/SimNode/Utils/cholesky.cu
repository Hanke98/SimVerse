#include "utils.h"
#include "cholesky.h"
#include "SimBlockArray.h"
#include <cstddef>
#include <cstdio>
#include <vector>


namespace dyno
{
    // warp reduce
    template<typename T>
    __device__ T warp_reduce_sum(T val)
    {
        for (int offset = 16; offset > 0; offset >>= 1) {
            // 0xffffffff刚好32个1,作为掩码表示warp内所有线程参与计算，这里的sync相当于一个reduce前的barrier
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }
    // 更新A_kk这样的下三角块
    // 每个tile是32 * 32小矩阵，128个线程分成4个warp，每个warp负责8行，每行32列，lane对应列
    // 每次调用计算A_kk = A_kk - L_kj * L_kj^T
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void SyrkSubLower(T* sA, const T* sB, int tile_stride)
    {
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid % 32;

        // 128 threads = 4 warps
        // 每个 warp 负责 8 行，lane 对应列
        constexpr int ROWS_PER_WARP = 8;

        const int row_begin = warp_id * ROWS_PER_WARP;
        const int row_end = row_begin + ROWS_PER_WARP;

        for (int row = row_begin; row < row_end && row < NTILES; ++row)
        {
            const int col = lane;
            // 只更新下三角
            if (col < NTILES && row >= col)
            {
                T sum = T(0);
                for (int k = 0; k < NTILES; ++k)
                {
                    sum += sB[row * tile_stride + k] * sB[col * tile_stride + k];
                }
                sA[row * tile_stride + col] -= sum;
            }
        }
        __syncthreads();
    }

    // cholesky分解一个tile，结果存回sA
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void PotrfTileLowerInplace(T* sA, int tile_stride)
    {
        const int thread_id = threadIdx.x;
        for (int j = 0; j < NTILES; ++j)
        {
            // Step 1: compute diagonal element L(j,j)
            if (thread_id == 0)
            {
                T sum = sA[j * tile_stride + j];
                for (int i = 0; i < j; ++i)
                {
                    T v = sA[j * tile_stride + i];
                    sum -= v * v;
                }
                sA[j * tile_stride + j] = sqrt(sum);
            }
            __syncthreads();

            // Step 2: compute elements below the diagonal in column j
            T L_jj = sA[j * tile_stride + j];

            for (int i = j + 1 + thread_id; i < NTILES; i += NTHREADS)
            {
                T sum = sA[i * tile_stride + j];
                for (int k = 0; k < j; ++k)
                {
                    sum -= sA[i * tile_stride + k] * sA[j * tile_stride + k];
                }
                sA[i * tile_stride + j] = sum / L_jj;
            }
            __syncthreads();
        }
    }

    // 更新A_ik这样的非对角块
    // 每次调用计算A_ik(sB) = A_ik(sB) - L_ij(sC) * L_kj^T(sD)
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void GemmNTSub(T* sB, const T* sC, const T* sD, int tile_stride)
    {
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid % 32;

        constexpr int ROWS_PER_WARP = 8;

        const int row_begin = warp_id * ROWS_PER_WARP;
        const int row_end = row_begin + ROWS_PER_WARP;

        for (int row = row_begin; row < row_end && row < NTILES; ++row)
        {
            const int col = lane;
            if (col < NTILES)
            {
                T sum = T(0);
                for (int k = 0; k < NTILES; ++k)
                {
                    sum += sC[row * tile_stride + k] * sD[col * tile_stride + k];
                }
                sB[row * tile_stride + col] -= sum;
            }
        }
        __syncthreads();
    }

    // 更新A_ij
    // 每次调用计算A_ik(sB) = A_ik(sB) - L_ij(sC) * L_kj^T(sD) 其中的4行(4个warp分别负责1行)
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void PanelGemmNTSub(T* sB, const T* sC, const T* sD, int tile_stride)
    {

    }

    // solve L_ik * L_kk^T = A_ik for L_ik
    // sA存L_kk，sB存A_ik，结果写回sB
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void TrsmRightLowerTranspose(const T* sA, T* sB, int tile_stride)
    {
        const int tid = threadIdx.x;

        // 一个线程负责一整行
        // row 之间并行，row 内部的列 c 串行推进
        // L_ik(row, c) = (A_ik(row, c) - sum_{p < c} L_ik(row, p) * L_kk(c, p)) / L_kk(c, c)
        for (int row = tid; row < NTILES; row += NTHREADS)
        {
            for (int c = 0; c < NTILES; ++c)
            {
                T sum = sB[row * tile_stride + c];

                // 减去前面已经求出来的这一行的贡献
                // TODO: 这里可以考虑改成使用4个线程来做reduce负责计算每行的sum，参考之前的warp的reduce
                for (int p = 0; p < c; ++p)
                {
                    sum -= sB[row * tile_stride + p] * sA[c * tile_stride + p];
                }

                // 除以对角元
                sum /= sA[c * tile_stride + c];

                // 原地写回 X(row, c)
                sB[row * tile_stride + c] = sum;
            }
        }

        __syncthreads();
    }

    // 计算b_i = b_i - sum_{j=0}^{i-1} L_ij * y_j
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void GemvSubLowerNTile(T* sB, const T* sC, const T* sD, int tile_stride)
    {
        const int tid = threadIdx.x;

        // sB -= sC * sD
        // sC: NTILES x NTILES dense tile
        // sD: NTILES vector
        // sB: NTILES vector

        for (int row = tid; row < NTILES; row += NTHREADS)
        {
            T sum = T(0);
            // TODO: Four path reduction for sum can be implemented here as well
            for (int k = 0; k < NTILES; ++k)
            {
                sum += sC[row * tile_stride + k] * sD[k];
            }
            sB[row] -= sum;
        }

        __syncthreads();
    }

    // 计算b_i = b_i - sum_{j=i+1}^{n-1} L_ji^T * x_j
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void GemvSubLowerTTile(T* sB, const T* sC, const T* sD, int tile_stride)
    {
        const int tid = threadIdx.x;

        // sB -= sC^T * sD
        // sC: NTILES x NTILES dense tile
        // sD: NTILES vector
        // sB: NTILES vector

        for (int row = tid; row < NTILES; row += NTHREADS)
        {
            T sum = T(0);
            // TODO: Four path reduction for sum can be implemented here as well
            for (int k = 0; k < NTILES; ++k)
            {
                // 按照transpose的方式访问sC
                sum += sC[k * tile_stride + row] * sD[k];
            }
            sB[row] -= sum;
        }

        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void TrsvLowerTileInplace(const T* sA, T* sB, int tile_stride)
    {
        const int tid = threadIdx.x;

        // Solve: L * y = rhs
        // sA: lower triangular NTILES x NTILES
        // sB: rhs vector, overwritten with solution y

        // 用一个warp刚好可以负责一整行的reduction
        if (tid < 32)
        {
            const int lane = tid;

            for (int i = 0; i < NTILES; ++i)
            {
                T partial = T(0);

                for (int j = lane; j < i; j += 32)
                {
                    partial += sA[i * tile_stride + j] * sB[j];
                }

                T dot = warp_reduce_sum(partial);

                if (lane == 0)
                {
                    T sum = sB[i] - dot;
                    sum /= sA[i * tile_stride + i];
                    sB[i] = sum;
                }

                __syncwarp();
            }
        }

        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void TrsvUpperTileInplace(const T* sA, T* sB, int tile_stride)
    {
        const int tid = threadIdx.x;

        // Solve: L^T * x = rhs
        // sA: upper triangular NTILES x NTILES
        // sB: rhs vector, overwritten with solution x

        // 用一个warp刚好可以负责一整行的reduction
        if (tid < 32)
        {
            const int lane = tid;

            for (int i = NTILES - 1; i >= 0; --i)
            {
                T partial = T(0);

                for (int j = i + 1 + lane; j < NTILES; j += 32)
                {
                    partial += sA[j * tile_stride + i] * sB[j];
                }

                T dot = warp_reduce_sum(partial);

                if (lane == 0)
                {
                    T sum = sB[i] - dot;
                    sum /= sA[i * tile_stride + i];
                    sB[i] = sum;
                }

                __syncwarp();
            }
        }

        __syncthreads();
    }

    // tile_stride表示tile在shared memory中每行的存储stride
    // A_stride表示A在global memory中每行的存储步长，不考虑padding的情况就是A的列数
    // 统一采用row-major存储
    // 模板参数，NTILES表示tile的行数，NTHREADS表示线程块大小，T表示数据类型
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void LoadTile(const T* A_ij, T* sA, int A_stride, int tile_stride, bool is_diagonal = true)
    {
        // 线程块大小为 NTHREADS，每个线程加载一个元素
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES * NTILES; i += NTHREADS)
        {
            int row = i / NTILES;
            int col = i % NTILES;
            // 如果是对角块，且我们只需要下三角部分，那么只加载下三角部分的元素，或者加载整个块但上三角部分置零
            if(is_diagonal)
            {
                // 只加载下三角部分，或者加载整个块但上三角部分置零
                bool is_lower_triangular = (row >= col);
                if(is_lower_triangular)
                    sA[row * tile_stride + col] = A_ij[row * A_stride + col];
                else
                    sA[row * tile_stride + col] = T(0);
            }
            else 
            {
                sA[row * tile_stride + col] = A_ij[row * A_stride + col];
            }
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void StoreTile(const T* sA, T* L_kk, int L_stride, int tile_stride, bool is_diagonal = true)
    {
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES * NTILES; i += NTHREADS)
        {
            int row = i / NTILES;
            int col = i % NTILES;
            bool is_lower_triangular = (row >= col);
            if(is_diagonal)
            {
                if (is_lower_triangular)
                    L_kk[row * L_stride + col] = sA[row * tile_stride + col];
                else
                    L_kk[row * L_stride + col] = T(0);
            }
            else 
            {
                L_kk[row * L_stride + col] = sA[row * tile_stride + col];
            }
            
        }
        __syncthreads();
    }

    // Load a tile from a block with size block_size and row-major stride block_size.
    // Out-of-bound entries are treated as padded values:
    // - diagonal tile: padded diagonal is identity, other padded entries are zero
    // - non-diagonal tile: padded entries are zero
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void LoadTilePadded(
        const T* A_block,
        T* sA,
        int block_size,
        int tile_row,
        int tile_col,
        int tile_stride,
        bool is_diagonal = true)
    {
        const int tid = threadIdx.x;
        const int row0 = tile_row * NTILES;
        const int col0 = tile_col * NTILES;

        for (int idx = tid; idx < NTILES * NTILES; idx += NTHREADS)
        {
            int r = idx / NTILES;
            int c = idx % NTILES;
            int gr = row0 + r;
            int gc = col0 + c;

            T val = T(0);
            bool in_bound = (gr < block_size) && (gc < block_size);
            // in_bound的元素直接从A_block加载，out-of-bound元素根据是否在对角块以及位置决定是0还是1
            if (in_bound)
            {
                val = A_block[gr * block_size + gc];
            }
            else if (is_diagonal)
            {
                // For padded diagonal area use identity so padded Cholesky stays well-defined.
                if (gr == gc)
                    val = T(1);
            }

            if (is_diagonal && r < c)
                val = T(0);

            sA[r * tile_stride + c] = val;
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void StoreTileBounded(
        const T* sA,
        T* L_block,
        int block_size,
        int tile_row,
        int tile_col,
        int tile_stride,
        bool is_diagonal = true)
    {
        const int tid = threadIdx.x;
        const int row0 = tile_row * NTILES;
        const int col0 = tile_col * NTILES;

        for (int idx = tid; idx < NTILES * NTILES; idx += NTHREADS)
        {
            // shared_memory中的tile元素索引
            int r = idx / NTILES;
            int c = idx % NTILES;
            // global_memory中对应的元素索引
            int gr = row0 + r;
            int gc = col0 + c;
            if (gr >= block_size || gc >= block_size)
                continue;

            bool is_lower_triangular = (gr >= gc);
            if (is_diagonal)
            {
                if (is_lower_triangular)
                    L_block[gr * block_size + gc] = sA[r * tile_stride + c];
                else
                    L_block[gr * block_size + gc] = T(0);
            }
            else
            {
                L_block[gr * block_size + gc] = sA[r * tile_stride + c];
            }
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void LoadVecTile(const T* x, T* sx)
    {
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES; i += NTHREADS)
        {
            sx[i] = x[i];
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void StoreVecTile(const T* sx, T* x)
    {
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES; i += NTHREADS)
        {
            x[i] = sx[i];
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void LoadVecTilePadded(
        const T* x_block,
        T* sx,
        int vec_size,
        int tile_i)
    {
        const int tid = threadIdx.x;
        const int base = tile_i * NTILES;
        for (int i = tid; i < NTILES; i += NTHREADS)
        {
            int gi = base + i;
            sx[i] = (gi < vec_size) ? x_block[gi] : T(0);
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void StoreVecTileBounded(
        const T* sx,
        T* x_block,
        int vec_size,
        int tile_i)
    {
        const int tid = threadIdx.x;
        const int base = tile_i * NTILES;
        for (int i = tid; i < NTILES; i += NTHREADS)
        {
            int gi = base + i;
            if (gi < vec_size)
                x_block[gi] = sx[i];
        }
        __syncthreads();
    }

    // 默认row-major
    // 这个函数是为了得到A中对应tile在global_memory中的起始地址
    template<unsigned NTILES, class T>
    __device__ T* tile(T* A, unsigned A_stride, unsigned i, unsigned j) 
    {
        return A + i * NTILES * A_stride + j * NTILES;
    }

    // __launch_bounds__指定了每个block的最大线程数为NTHREADS，便于编译器优化
    // 这是一个每个block对应的size是uniform的版本，也就是A矩阵的每个diagonal的block是一个block_size * block_size的矩阵
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void CholeskyFactorizeUniformBlockTile(T* A, int block_size, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        T* A_block = A + env_id * block_size * block_size;

        extern __shared__ unsigned char smem[]; // 大小为4 * NTILES * NTILES * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存A_kk Tile
        T* sB = sA + NTILES * NTILES; // 存A_ik Tile
        T* sC = sB + NTILES * NTILES; // 存A_kj Tile
        T* sD = sC + NTILES * NTILES; // 存A_ij Tile

        int tile_dim = (block_size + NTILES - 1) / NTILES;

        for(int k = 0; k < tile_dim; k++)
        {
            // Step 1: Load A_kk tile into shared memory sA (with padding on boundary tiles)
            LoadTilePadded<NTILES, NTHREADS, T>(A_block, sA, block_size, k, k, tile_stride, true);

            // Step 2: Subtract previous contributions from sA
            // A_kk = A_kk - sum_{j=0}^{k-1} L_kj * L_kj^T
            for(int j = 0; j < k; j++)
            {
                // 加载非对角块L_kj到sB
                LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, k, j, tile_stride, false);
                // 计算A_kk - L_kj * L_kj^T
                SyrkSubLower<NTILES, NTHREADS>(sA, sB, tile_stride);
            }

            // Step 3: Cholesky factorization on the tile sA, result stored back in sA
            PotrfTileLowerInplace<NTILES, NTHREADS>(sA, tile_stride);

            // Step 4: Store the result from sA back to A_kk (bounded by real block_size)
            StoreTileBounded<NTILES, NTHREADS, T>(sA, A_block, block_size, k, k, tile_stride, true);

            // Step 5: Update the tiles below the diagonal in column k: A_ik = A_ik - L_ij * L_kj^T for i > k
            for(int i = k + 1; i < tile_dim; i++)
            {
                // 加载A_ik到sB（边界tile自动补零）
                LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, i, k, tile_stride, false);
                for(int j = 0; j < k; j++)
                {
                    // 加载L_ij到sC，加载L_kj到sD
                    LoadTilePadded<NTILES, NTHREADS, T>(A_block, sC, block_size, i, j, tile_stride, false);
                    LoadTilePadded<NTILES, NTHREADS, T>(A_block, sD, block_size, k, j, tile_stride, false);
                    // 计算A_ik = A_ik - L_ij * L_kj^T
                    GemmNTSub<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
                }

                // Step 6: Solve L_ik * L_kk^T = A_ik for L_ik, where L_kk is the Cholesky factor we just computed in sA
                TrsmRightLowerTranspose<NTILES, NTHREADS, T>(sA, sB, tile_stride);

                // Step 7: Store the result back to A_ik (bounded by real block_size)
                StoreTileBounded<NTILES, NTHREADS, T>(sB, A_block, block_size, i, k, tile_stride, false); 
            }
        }
    }

    // inplace cholesky
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void CholeskyFactorizeVariableBlockTile(
        T* A,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks)
            return;

        const int block_size = block_sizes[env_id];
        if (block_size <= 0)
            return;

        const int block_offset = block_offsets[env_id];
        T* A_block = A + block_offset;
        const int tile_stride = NTILES;
        const int tile_dim = (block_size + NTILES - 1) / NTILES;

        extern __shared__ unsigned char smem[];
        T* sA = reinterpret_cast<T*>(smem);
        T* sB = sA + NTILES * NTILES;
        T* sC = sB + NTILES * NTILES;
        T* sD = sC + NTILES * NTILES;

        for (int k = 0; k < tile_dim; ++k)
        {
            // A_kk (with padding on the last tile if needed)
            LoadTilePadded<NTILES, NTHREADS, T>(A_block, sA, block_size, k, k, tile_stride, true);

            // A_kk -= sum_j L_kj * L_kj^T
            for (int j = 0; j < k; ++j)
            {
                LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, k, j, tile_stride, false);
                SyrkSubLower<NTILES, NTHREADS, T>(sA, sB, tile_stride);
            }

            // potrf(A_kk)
            PotrfTileLowerInplace<NTILES, NTHREADS, T>(sA, tile_stride);

            // store L_kk
            StoreTileBounded<NTILES, NTHREADS, T>(sA, A_block, block_size, k, k, tile_stride, true);

            // A_ik update + trsm for i > k
            for (int i = k + 1; i < tile_dim; ++i)
            {
                LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, i, k, tile_stride, false);

                for (int j = 0; j < k; ++j)
                {
                    LoadTilePadded<NTILES, NTHREADS, T>(A_block, sC, block_size, i, j, tile_stride, false);
                    LoadTilePadded<NTILES, NTHREADS, T>(A_block, sD, block_size, k, j, tile_stride, false);
                    GemmNTSub<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
                }

                TrsmRightLowerTranspose<NTILES, NTHREADS, T>(sA, sB, tile_stride);
                StoreTileBounded<NTILES, NTHREADS, T>(sB, A_block, block_size, i, k, tile_stride, false);
            }
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void DiagPotrf(T* A, int block_size, int num_blocks, int k_tile)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks)
            return;

        T* A_block = A + env_id * block_size * block_size;
        const int tile_stride = NTILES;

        extern __shared__ unsigned char smem[]; // 大小为NTILES * NTILES * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存A_kk Tile
        // T* sB = sA + NTILES * NTILES;   // 存L_kj Tile

        // Load A_kk
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sA, block_size, k_tile, k_tile, tile_stride, true);
        
        // // left looking type
        // // Update A_kk(SYRK)
        // // when k > 0, substract history contribution
        // // A_kk -= sum_j L_kj * L_kj^T
        // for(int j = 0; j < k_tile; j++)
        // {
        //     LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, k_tile, j, tile_stride, false);
        //     SyrkSubLower<NTILES, NTHREADS, T>(sA, sB, tile_stride);
        // }
        
        // Potrf A_kk
        PotrfTileLowerInplace<NTILES, NTHREADS, T>(sA, tile_stride);
        // Store L_kk
        StoreTileBounded<NTILES, NTHREADS, T>(sA, A_block, block_size, k_tile, k_tile, tile_stride, true);
    }


    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void PanelTrsm(T* A, int block_size, int num_blocks, int k_tile)
    {
        int env_id = blockIdx.x;
        int i_tile = k_tile + 1 + blockIdx.y;
        int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || i_tile >= num_tiles)
            return;

        T* A_block = A + env_id * block_size * block_size;
        const int tile_stride = NTILES;

        extern __shared__ unsigned char smem[]; // 大小为2 * NTILES * NTILES * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_kk Tile (DiagPotrf Kernel 结果)
        T* sB = sA + NTILES * NTILES;   // 存 A_ik Tile (右手项)

        // Load L_kk(A_kk)
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sA, block_size, k_tile, k_tile, tile_stride, true);
        // Panel trail Result(A_ik)
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, i_tile, k_tile, tile_stride, false);
        // Trsm Solve
        TrsmRightLowerTranspose<NTILES, NTHREADS, T>(sA, sB, tile_stride);
        StoreTileBounded<NTILES, NTHREADS, T>(sB, A_block, block_size, i_tile, k_tile, tile_stride, false);
    }

    
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void PanelTrailGemm(T* A, int block_size, int num_blocks, int k_tile)
    {
        int env_id   = blockIdx.x;
        int i_tile   = k_tile + 1 + blockIdx.y;
        int j_tile   = k_tile + 1 + blockIdx.z;
        int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || i_tile >= num_tiles || j_tile >= num_tiles) 
            return;
        if (i_tile < j_tile) 
            return; // 只算下三角
        
        extern __shared__ unsigned char smem[]; // 大小为3 * NTILES * NTILES * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存A_ij Tile
        T* sB = sA + NTILES * NTILES;   // 存L_ik Tile
        T* sC = sB + NTILES * NTILES;   // 存 L_jk Tile

        bool diag_tile = i_tile == j_tile;

        T* A_block = A + env_id * block_size * block_size;
        const int tile_stride = NTILES;

        // Load A_ij
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sA, block_size, i_tile, j_tile, tile_stride, diag_tile);
        // Load L_ik
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sB, block_size, i_tile, k_tile, tile_stride, false);
        // Load L_jk
        LoadTilePadded<NTILES, NTHREADS, T>(A_block, sC, block_size, j_tile, k_tile, tile_stride, false);

        if (diag_tile)
        {
            SyrkSubLower<NTILES, NTHREADS, T>(sA, sB, tile_stride);
        }
        else 
        {
            GemmNTSub<NTILES, NTHREADS, T>(sA, sB, sC, tile_stride);
        }
        StoreTileBounded<NTILES, NTHREADS, T>(sA, A_block, block_size, i_tile, j_tile, tile_stride, diag_tile);
    }

    // 与上面的cholesky分解配套的Lower和Upper solve函数
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void LowerSolveUniformBlockTile(const T* L, T* x, int block_size, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        
        extern __shared__ unsigned char smem[]; // 大小为2 * (NTILES * NTILES  + NTILES) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_ii Tile
        T* sB = sA + NTILES * NTILES; // 存y_i Tile
        T* sC = sB + NTILES; // 存L_ij Tile
        T* sD = sC + NTILES * NTILES; // 存y_j Tile

        int tile_dim = (block_size + NTILES - 1) / NTILES;
        for(int i = 0; i < tile_dim; i++)
        {
            // Step 1: Load L_ii tile into shared memory sA
            LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, i, i, tile_stride, true);

            // Step 2: Load y_i into sB
            LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i);

            // Step 3: Subtract previous contributions from sB
            // b_i -= sum_{j=0}^{i-1} L_ij * y_j
            for(int j = 0; j < i; j++)
            {
                // Load L_ij into sC, Load y_j into sD
                LoadTilePadded<NTILES, NTHREADS, T>(L_block, sC, block_size, i, j, tile_stride, false);
                LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sD, block_size, j);
                // 计算b_i -= L_ij * y_j
                GemvSubLowerNTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            // Step 4: Solve L_ii * y_i = b for y_i
            TrsvLowerTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);

            // Step 5: Store the result back to y_i (bounded by real block_size)
            StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i);
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void UpperSolveUniformBlockTile(const T* L, T* x, int block_size, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ unsigned char smem[]; // 大小为2 * (NTILES * NTILES  + NTILES) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_ii Tile, 按transpose使用
        T* sB = sA + NTILES * NTILES; // 存x_i Tile
        T* sC = sB + NTILES; // 存L_ji Tile, 按transpose使用
        T* sD = sC + NTILES * NTILES; // 存x_j Tile

        int tile_dim = (block_size + NTILES - 1) / NTILES;
        for(int i = tile_dim - 1; i >= 0; i--)
        {
            // Step 1: Load L_ii tile into shared memory sA（后续按转置方法使用）
            LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, i, i, tile_stride, true);

            // Step 2: Load x_i into sB
            LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i);

            // Step 3: Subtract previous contributions from sB
            // b_i -= sum_{j=i+1}^{n-1} L_ji^T * x_j
            for(int j = i + 1; j < tile_dim; j++)
            {
                // Load L_ji into sC（后续按转置方法使用）, Load y_j into sD
                LoadTilePadded<NTILES, NTHREADS, T>(L_block, sC, block_size, j, i, tile_stride, false);
                LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sD, block_size, j);
                // 计算b_i -= L_ji^T * x_j
                GemvSubLowerTTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            // Step 4: Solve L_ii^T * x_i = b for x_i
            TrsvUpperTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);

            // Step 5: Store the result back to x_i (bounded by real block_size)
            StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i);
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void LowerSolveVariableBlockTile(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks)
            return;

        const int block_size = block_sizes[env_id];
        if (block_size <= 0)
            return;

        const int block_offset = block_offsets[env_id];
        const int x_offset = x_offsets[env_id];
        const T* L_block = L + block_offset;
        T* x_block = x + x_offset;

        const int tile_stride = NTILES;
        const int tile_dim = (block_size + NTILES - 1) / NTILES;

        extern __shared__ unsigned char smem_var_lower[];
        T* sA = reinterpret_cast<T*>(smem_var_lower); // L_ii
        T* sB = sA + NTILES * NTILES;                 // y_i
        T* sC = sB + NTILES;                          // L_ij
        T* sD = sC + NTILES * NTILES;                 // y_j

        for (int i = 0; i < tile_dim; ++i)
        {
            LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, i, i, tile_stride, true);
            LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i);

            for (int j = 0; j < i; ++j)
            {
                LoadTilePadded<NTILES, NTHREADS, T>(L_block, sC, block_size, i, j, tile_stride, false);
                LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sD, block_size, j);
                GemvSubLowerNTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            TrsvLowerTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);
            StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i);
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void UpperSolveVariableBlockTile(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks)
            return;

        const int block_size = block_sizes[env_id];
        if (block_size <= 0)
            return;

        const int block_offset = block_offsets[env_id];
        const int x_offset = x_offsets[env_id];
        const T* L_block = L + block_offset;
        T* x_block = x + x_offset;

        const int tile_stride = NTILES;
        const int tile_dim = (block_size + NTILES - 1) / NTILES;

        extern __shared__ unsigned char smem_var_upper[];
        T* sA = reinterpret_cast<T*>(smem_var_upper); // L_ii (as transpose)
        T* sB = sA + NTILES * NTILES;                 // x_i
        T* sC = sB + NTILES;                          // L_ji
        T* sD = sC + NTILES * NTILES;                 // x_j

        for (int i = tile_dim - 1; i >= 0; --i)
        {
            LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, i, i, tile_stride, true);
            LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i);

            for (int j = i + 1; j < tile_dim; ++j)
            {
                LoadTilePadded<NTILES, NTHREADS, T>(L_block, sC, block_size, j, i, tile_stride, false);
                LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sD, block_size, j);
                GemvSubLowerTTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            TrsvUpperTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);
            StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i);
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void LowerDiagSolve(
        const T* L,
        T* x,
        int block_size,
        int num_blocks,
        int k_tile)
    {
        int env_id = blockIdx.x;

        const int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || k_tile >= num_tiles || block_size <= 0)
            return;

        const int tile_stride = NTILES;

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ unsigned char smem[]; // 大小为NTILES * (NTILES + 1) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_kk Tile
        T* sB = sA + NTILES * NTILES;   // 存rhs右手项

        LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, k_tile, k_tile, tile_stride, true);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, k_tile);
        TrsvLowerTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);
        StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, k_tile);
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void LowerUpdatePanel(
        const T* L,
        T* x,
        int block_size,
        int num_blocks,
        int k_tile)
    {
        int env_id = blockIdx.x;
        int i_tile = blockIdx.y + k_tile + 1;

        const int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || i_tile >= num_tiles || block_size <= 0)
            return;
        const int tile_stride = NTILES;

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ unsigned char smem[]; // 大小为NTILES * (NTILES + 2) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_ik Tile
        T* sB = sA + NTILES * NTILES;   // 存b_i
        T* sC = sB + NTILES; // 存y_k
        LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, i_tile, k_tile, tile_stride, false);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i_tile);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sC, block_size, k_tile);
        
        GemvSubLowerNTile<NTILES, NTHREADS, T>(sB, sA, sC, tile_stride);
        StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i_tile);
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void UpperDiagSolve(
        const T* L,
        T* x,
        int block_size,
        int num_blocks,
        int k_tile)
    {
        int env_id = blockIdx.x;

        const int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || k_tile >= num_tiles || block_size <= 0)
            return;

        const int tile_stride = NTILES;

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ unsigned char smem[]; // 大小为NTILES * (NTILES + 1) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_kk Tile
        T* sB = sA + NTILES * NTILES;   // 存rhs右手项

        LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, k_tile, k_tile, tile_stride, true);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, k_tile);
        TrsvUpperTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);
        StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, k_tile);
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__ __launch_bounds__(NTHREADS) void UpperUpdatePanel(
        const T* L,
        T* x,
        int block_size,
        int num_blocks,
        int k_tile)
    {
        int env_id = blockIdx.x;
        int i_tile = k_tile - blockIdx.y - 1;

        const int num_tiles = (block_size + NTILES - 1) / NTILES;
        if (env_id >= num_blocks || i_tile < 0 || block_size <= 0)
            return;
        const int tile_stride = NTILES;

        const T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ unsigned char smem[]; // 大小为NTILES * (NTILES + 2) * sizeof(T)
        T* sA = reinterpret_cast<T*>(smem); // 存L_ki Tile, 按transpose使用
        T* sB = sA + NTILES * NTILES;   // 存b_i
        T* sC = sB + NTILES; // 存x_k
        LoadTilePadded<NTILES, NTHREADS, T>(L_block, sA, block_size, k_tile, i_tile, tile_stride, false);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sB, block_size, i_tile);
        LoadVecTilePadded<NTILES, NTHREADS, T>(x_block, sC, block_size, k_tile);
        
        GemvSubLowerTTile<NTILES, NTHREADS, T>(sB, sA, sC, tile_stride);
        StoreVecTileBounded<NTILES, NTHREADS, T>(sB, x_block, block_size, i_tile);
    }



    // 在矩阵A的每个env对应block的size比较小的时候，不必再分成很多个tile来处理
    // 直接调用单矩阵Cholesky分解的kernel来处理每个block
    template<unsigned MAX_N, int NTHREADS, class T>
    __global__ void CholeskyFactorizeSingleTile(const T* A, T* L, const int* block_offsets, const int* block_sizes, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int block_size = block_sizes[env_id];
        if (block_size > MAX_N) 
        {
            return;
        }
        int thread_id = threadIdx.x;
        int block_offset = block_offsets[env_id];
        
        // sA的大小为MAX_N * MAX_N
        // block_size <= MAX_N，保证了每个block的矩阵都能放入shared memory中
        extern __shared__ unsigned char smem_single_factor_raw[];
        T* smem_single_factor = reinterpret_cast<T*>(smem_single_factor_raw);
        // Load block diagonal environment matrix into shared memory
        for (int idx = threadIdx.x; idx < block_size * block_size; idx += NTHREADS) 
        {
            int r = idx / block_size;
            int c = idx % block_size;
            smem_single_factor[r * block_size + c] = A[block_offset + r * block_size + c];
        }
        __syncthreads();
        for(int j = 0; j < block_size; j++)
        {
            // thread 0 computes the diagonal element L_ii
            if(thread_id == 0)
            {
                // L_jj = sqrt(A_jj - sum_j_{j<i}(L_ji^2))
                T sum = smem_single_factor[j + j * block_size];
                for (int i = 0; i < j; ++i) 
                {
                    T v = smem_single_factor[j * block_size + i];
                    sum -= v * v;
                }
                smem_single_factor[j + j * block_size] = sqrt(sum);
            }
            __syncthreads();
            T L_ii = smem_single_factor[j * block_size + j];
            // threads 1...i compute the off-diagonal elements L_ij for j < i
            // L_ij = (A_ij - sum_k_{k < j}(L_ik * L_jk)) / L_jj
            for (int i = j + 1 + thread_id; i < block_size; i += NTHREADS)
            {
                T sum = smem_single_factor[i * block_size + j];
                for (int k = 0; k < j; ++k) 
                {
                    sum -= smem_single_factor[i * block_size + k] * smem_single_factor[j * block_size + k];
                }
                smem_single_factor[i * block_size + j] = sum / L_ii;
            }
            __syncthreads();
        }
        // Store the result back to global memory
         for (int idx = thread_id; idx < block_size * block_size; idx += NTHREADS) 
         {
            int r = idx / block_size;
            int c = idx % block_size;

            if (r >= c) 
            {
                L[block_offset + r * block_size + c] = smem_single_factor[r * block_size + c];
            } 
            else 
            {
                L[block_offset + r * block_size + c] = T(0);
            }
        }
    }


    template<unsigned MAX_N, int NTHREADS, class T>
    __global__ void LowerSolveInplaceSingleTile(const T* L, T* x, const int* block_offsets, const int* block_sizes, const int* x_offsets,int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int block_size = block_sizes[env_id];
        if (block_size > MAX_N) 
        {
            return;
        }
        int thread_id = threadIdx.x;
        int block_offset = block_offsets[env_id];
        int x_offset = x_offsets[env_id];
        
        extern __shared__ unsigned char smem[];
        T* sL = reinterpret_cast<T*>(smem);            // size: MAX_N * MAX_N
        T* sx = sL + MAX_N * MAX_N;          // size: MAX_N
        // Load L and x into shared memory
        for (int idx = thread_id; idx < block_size * block_size; idx += NTHREADS) 
        {
            int r = idx / block_size;
            int c = idx % block_size;
            sL[r * block_size + c] = L[block_offset + r * block_size + c];
        }
        for (int idx = thread_id; idx < block_size; idx += NTHREADS) 
        {
            sx[idx] = x[x_offset + idx];
        }
        __syncthreads();
        // Lower solve: L y = b
        // warp reduce sum for each row with only first warp
        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            for (int i = 0; i < block_size; ++i) {
                T partial = 0;
                for (int k = lane; k < i; k += 32) {
                    partial += sL[i * block_size + k] * sx[k];
                }
                T dot = warp_reduce_sum(partial);
                if (lane == 0) {
                    sx[i] = (sx[i] - dot) / sL[i * block_size + i];
                }
                __syncwarp();
            }
        }

        __syncthreads();
        
        for (int i = thread_id; i < block_size; i += NTHREADS) {
            x[x_offset + i] = sx[i];
        }
    }


    template<unsigned MAX_N, int NTHREADS, class T>
    __global__ void UpperSolveInplaceSingleTile(const T* L, T* x, const int* block_offsets, const int* block_sizes, const int* x_offsets,int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int block_size = block_sizes[env_id];
        if (block_size > MAX_N) 
        {
            return;
        }
        int thread_id = threadIdx.x;
        int block_offset = block_offsets[env_id];
        int x_offset = x_offsets[env_id];
        
        extern __shared__ unsigned char smem[];
        T* sL = reinterpret_cast<T*>(smem);            // size: MAX_N * MAX_N
        T* sx = sL + MAX_N * MAX_N;          // size: MAX_N
        // Load L and x into shared memory
        for (int idx = thread_id; idx < block_size * block_size; idx += NTHREADS) 
        {
            int r = idx / block_size;
            int c = idx % block_size;
            sL[r * block_size + c] = L[block_offset + r * block_size + c];
        }
        for (int idx = thread_id; idx < block_size; idx += NTHREADS) 
        {
            sx[idx] = x[x_offset + idx];
        }
        __syncthreads();
        // Lower solve: L^T x = y
        // warp reduce sum for each row with only first warp
        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            for (int i = block_size - 1; i >= 0; --i) 
            {
                T partial = 0;
                for (int k = i + 1 + lane; k < block_size; k += 32) {
                    partial += sL[k * block_size + i] * sx[k];
                }

                T dot = warp_reduce_sum(partial);

                if (lane == 0) {
                    sx[i] = (sx[i] - dot) / sL[i * block_size + i];
                }
                __syncwarp();
            }
        }
        __syncthreads();
        for (int i = thread_id; i < block_size; i += NTHREADS) {
            x[x_offset + i] = sx[i];
        }
    }

    


    // Basic Cholesky factorization for a single matrix
    template<typename T>
    __forceinline__ __device__ void CholeskyFactorizeKernel(const T* A, T* L, int n)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = A[i * n + j];
                for (int k = 0; k < j; k++)
                    sum -= L[i * n + k] * L[j * n + k];
                // L_ii = sqrt(A_ii - sum_k(L_ik^2))
                if (i == j)
                    L[i * n + j] = sqrt(sum);
                // L_ij = (A_ij - sum_k(L_ik * L_jk)) / L_jj
                else
                    L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    template<typename T>
    __forceinline__ __device__ void UpperSolveInplaceKernel(const T* L, T* x, int n)
    {
        // Solve L^T x = b for x, where L is lower triangular
        // x_i = (b_i - sum_{j>i} L_ji x_j) / L_ii
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = x[i];
            for(int j = i + 1; j < n; j++)
                sum -= L[j * n + i] * x[j];
            x[i] = sum / L[i * n + i];
        }
    }

    template<typename T>
    __forceinline__ __device__ void LowerSolveInplaceKernel(const T* L, T* x, int n)
    {
        // Solve L x = b for x, where L is lower triangular
        // x_i = (b_i - sum_{j<i} L_ji x_j) / L_ii
        for (int i = 0; i < n; i++)
        {
            T sum = x[i];
            for(int j = 0; j < i; j++)
                sum -= L[i * n + j] * x[j];
            x[i] = sum / L[i * n + i];
        }
    }

    template<typename T>
    __global__ void BatchCholeskyFactorize(
        const T* A, T* L, 
        const int* block_sizes, 
        const int* block_offsets, 
        int num_blocks)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_blocks)
            return;
        int block_size = block_sizes[tid];
        int offset = block_offsets[tid];
        CholeskyFactorizeKernel(&A[offset], &L[offset], block_size);
    }

    template<typename T>
    __global__ void BatchCholeskySolve(
        const T* L, T* x, 
        const int* block_sizes, 
        const int* block_offsets,
        const int* x_offsets, 
        int num_blocks)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_blocks)
            return;
        int block_size = block_sizes[tid];
        int L_offset = block_offsets[tid];
        int x_offset = x_offsets[tid];
        LowerSolveInplaceKernel(&L[L_offset], &x[x_offset], block_size);
        UpperSolveInplaceKernel(&L[L_offset], &x[x_offset], block_size);
    }


    template<typename T>
    void CholeskyFactorizeSimplestHost(
        T* A,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        cuSafeCall((BatchCholeskyFactorize<T><<<blocks, threads>>>(A, A, block_sizes, block_offsets, num_blocks)));
    }

    template<typename T>
    void CholeskyFactorizeSingleTiledHost(
        T* A,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks)
    {

        // int dev = 0;
        // cudaGetDevice(&dev);

        // int max_smem_default = 0;
        // int max_smem_optin = 0;

        // cudaDeviceGetAttribute(&max_smem_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        // cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

        // printf("MaxSharedMemoryPerBlock       = %d bytes\n", max_smem_default);
        // printf("MaxSharedMemoryPerBlockOptin = %d bytes\n", max_smem_optin);

        constexpr int MAX_N = 96;
        constexpr int NTHREADS = 128;
        size_t smem_bytes = MAX_N * MAX_N * sizeof(T);
        const int blocks = num_blocks;

        bool check_validity = false;

        if (check_validity)
        {
            std::vector<int> h_sizes(num_blocks);
            cuSafeCall(cudaMemcpy(h_sizes.data(), block_sizes, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost));
            int max_size = 0;
            for (int s : h_sizes) max_size = (s > max_size) ? s : max_size;
            if (max_size > MAX_N)
            {
                std::printf("[CholeskyFactorizeHost] SingleTiled requires block_size <= %d, got %d\n", MAX_N, max_size);
                return;
            }
        }

        cuSafeCall(cudaFuncSetAttribute(
            CholeskyFactorizeSingleTile<MAX_N, NTHREADS, T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));

        cuSafeCall((CholeskyFactorizeSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                A, A, block_offsets, block_sizes, num_blocks)));
    }

    template<typename T>
    void CholeskyFactorizeUniformTiledHost(
        T* A,
        int block_size,
        int num_blocks)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;

        size_t smem_bytes = 4 * NTILES * NTILES * sizeof(T);
        cuSafeCall((CholeskyFactorizeUniformBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                A, block_size, num_blocks)));
    }

    template<typename T>
    void CholeskyFactorizePaddedTiledHost(
        T* A,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;

        size_t smem_bytes = 4 * NTILES * NTILES * sizeof(T);
        cuSafeCall((CholeskyFactorizeVariableBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                A, block_sizes, block_offsets, num_blocks)));
    }

    template<typename T>
    void CholeskyFactorizeWavefrontTiledHost(
        T* A,
        int block_size,
        int num_blocks,
        cudaStream_t stream)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;

        if(stream == nullptr)
        {
            std::printf("[CholeskySolveHost] WavefrontTiledHost need cuda stream input!");
            return;
        }
        
        const int num_tiles = (block_size + NTILES - 1) / NTILES;

        size_t smem_diag = 1 * NTILES * NTILES * sizeof(T);
        size_t smem_panel = 2 * NTILES * NTILES * sizeof(T);
        size_t smem_trail = 3 * NTILES * NTILES * sizeof(T);

        // three phase wavefront progress in same stream
        for(int k = 0; k < num_tiles; k++)
        {
            // phase A cholesky factorize A_kk(DiagPotrf)
            DiagPotrf<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_diag, stream>>>(
                A,          // in-place matrix buffer
                block_size, // 单个矩阵维度
                num_blocks,
                k); 

            // phase B panel phase solve L_ik(TRSM)
            // 当前需要并行计算的非对角tile
            const int panel_tiles = num_tiles - (k + 1);
            if (panel_tiles > 0)
            {
                dim3 grid_panel(num_blocks, panel_tiles, 1);
                PanelTrsm<NTILES, NTHREADS, T>
                    <<<grid_panel, NTHREADS, smem_panel, stream>>>(
                        A,
                        block_size,
                        num_blocks,
                        k);
            }

            // phase C update A_ij(GEMM)
            // 更新未来的A_ij
            if (panel_tiles > 0) {
                dim3 grid(num_blocks, panel_tiles, panel_tiles);

                PanelTrailGemm<NTILES, NTHREADS, T>
                    <<<grid, NTHREADS, smem_trail, stream>>>(
                        A, block_size, num_blocks, k);
            }
        }

        cuSafeCall(cudaStreamSynchronize(stream));

    }

    template<typename T>
    void CholeskySolveSimplestHost(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        cuSafeCall((BatchCholeskySolve<T><<<blocks, threads>>>(L, x, block_sizes, block_offsets, x_offsets, num_blocks)));
    }

    template<typename T>
    void CholeskySolveSingleTiledHost(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks)
    {
        constexpr int MAX_N = 96;
        constexpr int NTHREADS = 128;
        size_t smem_bytes = MAX_N * MAX_N * sizeof(T) + MAX_N * sizeof(T);
        const int blocks = num_blocks;

        // for debug
        bool check_validity = false;
        if (check_validity)
        {
            std::vector<int> h_sizes(num_blocks);
            cuSafeCall(cudaMemcpy(h_sizes.data(), block_sizes, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost));
            int max_size = 0;
            for (int s : h_sizes) max_size = (s > max_size) ? s : max_size;
            if (max_size > MAX_N)
            {
                std::printf("[CholeskySolveHost] SingleTiled requires block_size <= %d, got %d\n", MAX_N, max_size);
                return;
            }
        }

        cuSafeCall(cudaFuncSetAttribute(
            LowerSolveInplaceSingleTile<MAX_N, NTHREADS, T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));

        cuSafeCall((LowerSolveInplaceSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                L, x, block_offsets, block_sizes, x_offsets, num_blocks)));

        cuSafeCall(cudaFuncSetAttribute(
            UpperSolveInplaceSingleTile<MAX_N, NTHREADS, T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));

        cuSafeCall((UpperSolveInplaceSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                L, x, block_offsets, block_sizes, x_offsets, num_blocks)));
    }

    template<typename T>
    void CholeskySolveUniformTiledHost(
        const T* L,
        T* x,
        const int block_size,
        int num_blocks)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;

        size_t smem_bytes = 2 * (NTILES * NTILES + NTILES) * sizeof(T);
        cuSafeCall((LowerSolveUniformBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                L, x, block_size, num_blocks)));

        cuSafeCall((UpperSolveUniformBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                L, x, block_size, num_blocks)));
    }

    template<typename T>
    void CholeskySolvePaddedTiledHost(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;
        size_t smem_bytes = 2 * (NTILES * NTILES + NTILES) * sizeof(T);

        cuSafeCall((LowerSolveVariableBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                L, x, block_sizes, block_offsets, x_offsets, num_blocks)));

        cuSafeCall((UpperSolveVariableBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                L, x, block_sizes, block_offsets, x_offsets, num_blocks)));
    }

    template<typename T>
    void CholeskySolveWavefrontTiledHost(
        const T* L,
        T* x,
        const int block_size,
        int num_blocks,
        cudaStream_t stream)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;


        const int num_tiles = (block_size + NTILES - 1) / NTILES;

        // shared memory for cholesky solve
        size_t smem_diag = (NTILES * NTILES + NTILES) * sizeof(T);
        size_t smem_panel = (NTILES * NTILES + 2 * NTILES) * sizeof(T);

        // two phase wavefront progress in same stream(Lower Solve)
        for(int k = 0; k < num_tiles; k++)
        {
            // Solve y_k with L_kk y_k = b_k
            LowerDiagSolve<NTILES, NTHREADS, T>
                <<<num_blocks, NTHREADS, smem_diag, stream>>>
                (L, x, block_size, num_blocks, k);

            // Update b_i
            // right-looking更新的是未来的b_i(i > k)
            const int panel_tiles = num_tiles - (k + 1);
            if (panel_tiles > 0)
            {
                dim3 grid_panel(num_blocks, panel_tiles, 1);
                LowerUpdatePanel<NTILES, NTHREADS, T>
                    <<<grid_panel, NTHREADS, smem_panel, stream>>>(
                        L, x, block_size, num_blocks, k);
            }

        }

        // Upper Solve
        for(int k = num_tiles - 1; k >= 0; k--)
        {
            // Solve y_k with L_kk y_k = b_k
            UpperDiagSolve<NTILES, NTHREADS, T>
                <<<num_blocks, NTHREADS, smem_diag, stream>>>
                (L, x, block_size, num_blocks, k);

            // Update b_i
            // right-looking更新的是未来的b_i(i > k)
            const int panel_tiles = k;
            if (panel_tiles > 0)
            {
                dim3 grid_panel(num_blocks, panel_tiles, 1);
                UpperUpdatePanel<NTILES, NTHREADS, T>
                    <<<grid_panel, NTHREADS, smem_panel, stream>>>(
                        L, x, block_size, num_blocks, k);
            }
        }

        cuSafeCall(cudaStreamSynchronize(stream));
    }

    template<typename T>
    BatchedCholeskySolver<T>::BatchedCholeskySolver()
    {
        initialized_ = false;
        stream_ = nullptr;
    }

    template<typename T>
    BatchedCholeskySolver<T>::~BatchedCholeskySolver()
    {
        Release();
    }

    template<typename T>
    void BatchedCholeskySolver<T>::Release() 
    {
        factorize_graph_cache_.release();
        solve_graph_cache_.release();
        stream_ = nullptr;
        initialized_ = false;
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::Initialize(cudaStream_t stream)
    {
        // Re-init: clear old state first
        Release();

        initialized_ = true;
        stream_ = stream;
        return true;
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::IsInitialized() const
    {
        return initialized_;
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::Factorize(
        T* A,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks,
        CholeskyMethod method,
        int uniform_block_size,
        bool use_graph)
    {
        if (!initialized_)
        {
            std::printf("[BatchedCholeskySolver::Factorize] solver not initialized\n");
            return false;
        }
        if (A == nullptr || num_blocks <= 0)
        {
            std::printf("[BatchedCholeskySolver::Solve] invalid input pointers/num_blocks\n");
            return false;
        }

        // 现在统一 in-place：A 同时是输入和输出
        switch (method)
        {
        case CholeskyMethod::Simplest:
            if (block_sizes == nullptr || block_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskyFactorizeSimplestHost(A, block_sizes, block_offsets, num_blocks);
            return true;

        case CholeskyMethod::SingleTiled:
            if (block_sizes == nullptr || block_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskyFactorizeSingleTiledHost(A, block_sizes, block_offsets, num_blocks);
            return true;

        case CholeskyMethod::UniformTiled:
            if (uniform_block_size <= 0)
            {
                std::printf("[BatchedCholeskySolver::Factorize] UniformTiled requires uniform_block_size > 0\n");
                return false;
            }
            CholeskyFactorizeUniformTiledHost(A, uniform_block_size, num_blocks);
            return true;

        case CholeskyMethod::PaddedTiled:
            if (block_sizes == nullptr || block_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskyFactorizePaddedTiledHost(A, block_sizes, block_offsets, num_blocks);
            return true;

        case CholeskyMethod::WavefrontTiled:
            if (uniform_block_size <= 0)
            {
                std::printf("[BatchedCholeskySolver::Factorize] WavefrontTiled requires uniform_block_size > 0\n");
                return false;
            }
            // CholeskyFactorizeWavefrontTiledHost(A, uniform_block_size, num_blocks, stream_);
            if (use_graph)
                FactorizeWavefrontWithGraph(A, uniform_block_size, num_blocks);
            else
                CholeskyFactorizeWavefrontTiledHost(A, uniform_block_size, num_blocks, stream_);
            return true;

        default:
            std::printf("[BatchedCholeskySolver::Factorize] unknown method=%d\n", static_cast<int>(method));
            return false;
        }
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::Factorize(
        DevBlockArray<T>& A_blocks,
        const int* block_sizes,
        CholeskyMethod method,
        int uniform_block_size,
        bool use_graph)
    {
        // Note:
        // block_sizes stores matrix dimension n (NOT n*n).
        // A_blocks stores flattened per-block data.
        return Factorize(
            A_blocks.Begin(),
            block_sizes,
            A_blocks.Offsets().Begin(),
            A_blocks.NumBlocks(),
            method,
            uniform_block_size,
            use_graph);
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::Factorize(
        DevBlockArray<T>& A_blocks,
        const DevArr<int>& block_sizes,
        CholeskyMethod method,
        int uniform_block_size,
        bool use_graph)
    {
        return Factorize(
            A_blocks,
            block_sizes.Begin(),
            method,
            uniform_block_size,
            use_graph);
    }

    template<typename T>
    bool BatchedCholeskySolver<T>::FactorizeWavefrontWithGraph(
		T* A, 
		const int block_size, 
		int num_blocks)
    {
        constexpr int NTILES = 32;
        constexpr int NTHREADS = 128;

        if(stream_ == nullptr)
        {
            std::printf("[CholeskySolveHost] WavefrontTiledHost need cuda stream input!");
            return false;
        }

        const bool need_rebuild =
        !factorize_graph_cache_.built ||
        factorize_graph_cache_.A_ptr != A ||
        factorize_graph_cache_.block_size != block_size ||
        factorize_graph_cache_.num_blocks != num_blocks ||
        factorize_graph_cache_.stream != stream_;

        if (need_rebuild)
        {
            factorize_graph_cache_.release();
            const int num_tiles = (block_size + NTILES - 1) / NTILES;

            size_t smem_diag = 1 * NTILES * NTILES * sizeof(T);
            size_t smem_panel = 2 * NTILES * NTILES * sizeof(T);
            size_t smem_trail = 3 * NTILES * NTILES * sizeof(T);

            // 捕获当前 stream 上的 kernel 调度序列
            cuSafeCall(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

            // three phase wavefront progress in same stream
            for(int k = 0; k < num_tiles; k++)
            {
                // phase A cholesky factorize A_kk(DiagPotrf)
                DiagPotrf<NTILES, NTHREADS, T>
                <<<num_blocks, NTHREADS, smem_diag, stream_>>>(
                    A,          // in-place matrix buffer
                    block_size, // 单个矩阵维度
                    num_blocks,
                    k); 

                // phase B panel phase solve L_ik(TRSM)
                // 当前需要并行计算的非对角tile
                const int panel_tiles = num_tiles - (k + 1);
                if (panel_tiles > 0)
                {
                    dim3 grid_panel(num_blocks, panel_tiles, 1);
                    PanelTrsm<NTILES, NTHREADS, T>
                        <<<grid_panel, NTHREADS, smem_panel, stream_>>>(
                            A,
                            block_size,
                            num_blocks,
                            k);
                }

                // phase C update A_ij(GEMM)
                // 更新未来的A_ij
                if (panel_tiles > 0) {
                    dim3 grid(num_blocks, panel_tiles, panel_tiles);

                    PanelTrailGemm<NTILES, NTHREADS, T>
                        <<<grid, NTHREADS, smem_trail, stream_>>>(
                            A, block_size, num_blocks, k);
                }
            }

            cuSafeCall(cudaStreamEndCapture(stream_, &factorize_graph_cache_.graph));
            cuSafeCall(cudaGraphInstantiate(&factorize_graph_cache_.exec, factorize_graph_cache_.graph, nullptr, nullptr, 0));
            factorize_graph_cache_.A_ptr = A;
            factorize_graph_cache_.block_size = block_size;
            factorize_graph_cache_.num_blocks = num_blocks;
            factorize_graph_cache_.stream = stream_;
            factorize_graph_cache_.built = true;
        }   
        
        cuSafeCall(cudaGraphLaunch(factorize_graph_cache_.exec, stream_));
        return true;
    }
    
    template<typename T>
    bool BatchedCholeskySolver<T>::Solve(
        const T* L,
		T* x,
		const int* block_sizes,
		const int* block_offsets,
		const int* x_offsets,
		int num_blocks,
		CholeskyMethod method,
		int uniform_block_size)
    {
        if (!initialized_)
        {
            std::printf("[BatchedCholeskySolver::Solve] solver not initialized\n");
            return false;
        }
        if (L == nullptr || x == nullptr || num_blocks <= 0)
        {
            std::printf("[BatchedCholeskySolver::Solve] invalid input pointers/num_blocks\n");
            return false;
        }

        switch (method)
        {
        case CholeskyMethod::Simplest:
            if (block_sizes == nullptr || block_offsets == nullptr || x_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskySolveSimplestHost(L, x, block_sizes, block_offsets, x_offsets, num_blocks);
            return true;

        case CholeskyMethod::SingleTiled:
            if (block_sizes == nullptr || block_offsets == nullptr || x_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskySolveSingleTiledHost(L, x, block_sizes, block_offsets, x_offsets, num_blocks);
            return true;

        case CholeskyMethod::UniformTiled:
            if (uniform_block_size <= 0)
            {
                std::printf("[BatchedCholeskySolver::Solve] UniformTiled requires uniform_block_size > 0\n");
                return false;
            }
            CholeskySolveUniformTiledHost(L, x, uniform_block_size, num_blocks);
            return true;

        case CholeskyMethod::PaddedTiled:
            if (block_sizes == nullptr || block_offsets == nullptr || x_offsets == nullptr)
            {
                std::printf("[BatchedCholeskySolver::Solve] invalid input offset/size pointers\n");
                return false;
            }
            CholeskySolvePaddedTiledHost(L, x, block_sizes, block_offsets, x_offsets, num_blocks);
            return true;

        case CholeskyMethod::WavefrontTiled:
        if (uniform_block_size <= 0)
            {
                std::printf("[BatchedCholeskySolver::Solve] WavefrontTiled requires uniform_block_size > 0\n");
                return false;
            }
            CholeskySolveWavefrontTiledHost(L, x, uniform_block_size, num_blocks, stream_);
            return true;

        default:
            std::printf("[BatchedCholeskySolver::Solve] unknown method=%d\n", static_cast<int>(method));
            return false;
        }
    }

    template class dyno::BatchedCholeskySolver<float>;
    template class dyno::BatchedCholeskySolver<double>;
} // namespace dyno
