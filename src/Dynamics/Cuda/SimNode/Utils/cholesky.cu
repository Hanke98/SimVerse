#include "utils.h"
#include "cholesky.h"
#include <cstdio>


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
        int A_stride = block_size; // 因为是row-major，所以每行的步长就是列数
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        T* A_block = A + env_id * block_size * block_size;

        extern __shared__ T smem[]; // 大小为4 * NTILES * NTILES * sizeof(T)
        T* sA = smem; // 存A_kk Tile
        T* sB = sA + NTILES * NTILES; // 存A_ik Tile
        T* sC = sB + NTILES * NTILES; // 存A_kj Tile
        T* sD = sC + NTILES * NTILES; // 存A_ij Tile

        if(block_size % NTILES != 0)
        {
            // 忽略没有padding的情况
            return;
        }

        int tile_dim = block_size / NTILES;

        for(int k = 0; k < tile_dim; k++)
        {
            T* A_kk = tile<NTILES>(A_block, A_stride, k, k);

            // Step 1: Load A_kk tile into shared memory sA
            LoadTile<NTILES, NTHREADS, T>(A_kk, sA, A_stride, tile_stride);

            // Step 2: Subtract previous contributions from sA
            // A_kk = A_kk - sum_{j=0}^{k-1} L_kj * L_kj^T
            for(int j = 0; j < k; j++)
            {
                T* L_kj = tile<NTILES>(A_block, A_stride, k, j);
                // 加载非对角块L_kj到sB
                LoadTile<NTILES, NTHREADS, T>(L_kj, sB, A_stride, tile_stride, false);
                // 计算A_kk - L_kj * L_kj^T
                SyrkSubLower<NTILES, NTHREADS>(sA, sB, tile_stride);
            }

            // Step 3: Cholesky factorization on the tile sA, result stored back in sA
            PotrfTileLowerInplace<NTILES, NTHREADS>(sA, tile_stride);

            // Step 4: Store the result from sA back to A_kk
            StoreTile<NTILES, NTHREADS, T>(sA, A_kk, A_stride, tile_stride);

            // Step 5: Update the tiles below the diagonal in column k: A_ik = A_ik - L_ij * L_kj^T for i > k
            for(int i = k + 1; i < tile_dim; i++)
            {
                T* A_ik = tile<NTILES>(A_block, A_stride, i, k);
                // 加载A_ik到sB
                LoadTile<NTILES, NTHREADS, T>(A_ik, sB, A_stride, tile_stride, false);
                for(int j = 0; j < k; j++)
                {
                    T* L_ij = tile<NTILES>(A_block, A_stride, i, j);
                    T* L_kj = tile<NTILES>(A_block, A_stride, k, j);
                    // 加载L_ij到sC，加载L_kj到sD
                    LoadTile<NTILES, NTHREADS, T>(L_ij, sC, A_stride, tile_stride, false);
                    LoadTile<NTILES, NTHREADS, T>(L_kj, sD, A_stride, tile_stride, false);
                    // 计算A_ik = A_ik - L_ij * L_kj^T
                    GemmNTSub<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
                }

                // Step 6: Solve L_ik * L_kk^T = A_ik for L_ik, where L_kk is the Cholesky factor we just computed in sA
                TrsmRightLowerTranspose<NTILES, NTHREADS, T>(sA, sB, tile_stride);

                // Step 7: Store the result back to A_ik
                StoreTile<NTILES, NTHREADS, T>(sB, A_ik, A_stride, tile_stride, false); 
            }
        }
    }

    // 与上面的cholesky分解配套的Lower和Upper solve函数
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void LowerSolveUniformBlockTile(T* L, T* x, int block_size, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int L_stride = block_size; // 因为是row-major，所以每行的步长就是列数
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ T smem[]; // 大小为2 * (NTILES * NTILES  + NTILES) * sizeof(T)
        T* sA = smem; // 存L_ii Tile
        T* sB = sA + NTILES * NTILES; // 存y_i Tile
        T* sC = sB + NTILES; // 存L_ij Tile
        T* sD = sC + NTILES * NTILES; // 存y_j Tile

        if(block_size % NTILES != 0)
        {
            // 忽略没有padding的情况
            return;
        }

        int tile_dim = block_size / NTILES;
        for(int i = 0; i < tile_dim; i++)
        {
            // Step 1: Load L_ii tile into shared memory sA
            T* L_ii = tile<NTILES>(L_block, L_stride, i, i);
            LoadTile<NTILES, NTHREADS, T>(L_ii, sA, L_stride, tile_stride);

            // Step 2: Load y_i into sB
            T* y_i = x_block + i * NTILES; // y_i在global memory中的起始地址
            LoadVecTile<NTILES, NTHREADS, T>(y_i, sB);

            // Step 3: Subtract previous contributions from sB
            // b_i -= sum_{j=0}^{i-1} L_ij * y_j
            for(int j = 0; j < i; j++)
            {
                // Load L_ij into sC, Load y_j into sD
                T* L_ij = tile<NTILES>(L_block, L_stride, i, j);
                T* y_j = x_block + j * NTILES;
                LoadTile<NTILES, NTHREADS, T>(L_ij, sC, L_stride, tile_stride, false);
                LoadVecTile<NTILES, NTHREADS, T>(y_j, sD);
                // 计算b_i -= L_ij * y_j
                GemvSubLowerNTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            // Step 4: Solve L_ii * y_i = b for y_i
            TrsvLowerTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);

            // Step 5: Store the result back to y_i
            StoreVecTile<NTILES, NTHREADS, T>(sB, y_i);
        }
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void UpperSolveUniformBlockTile(T* L, T* x, int block_size, int num_blocks)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_blocks) 
        {
            return;
        }
        int L_stride = block_size; // 因为是row-major，所以每行的步长就是列数
        int tile_stride = NTILES; // tile在shared memory中是紧凑存储的，所以tile_stride等于NTILES

        T* L_block = L + env_id * block_size * block_size;
        T* x_block = x + env_id * block_size;

        extern __shared__ T smem[]; // 大小为2 * (NTILES * NTILES  + NTILES) * sizeof(T)
        T* sA = smem; // 存L_ii Tile, 按transpose使用
        T* sB = sA + NTILES * NTILES; // 存x_i Tile
        T* sC = sB + NTILES; // 存L_ji Tile, 按transpose使用
        T* sD = sC + NTILES * NTILES; // 存x_j Tile

        if(block_size % NTILES != 0)
        {
            // 忽略没有padding的情况
            return;
        }

        int tile_dim = block_size / NTILES;
        for(int i = tile_dim - 1; i >= 0; i--)
        {
            // Step 1: Load L_ii tile into shared memory sA（后续按转置方法使用）
            T* L_ii = tile<NTILES>(L_block, L_stride, i, i);
            LoadTile<NTILES, NTHREADS, T>(L_ii, sA, L_stride, tile_stride);

            // Step 2: Load x_i into sB
            T* x_i = x_block + i * NTILES;
            LoadVecTile<NTILES, NTHREADS, T>(x_i, sB);

            // Step 3: Subtract previous contributions from sB
            // b_i -= sum_{j=i+1}^{n-1} L_ji^T * x_j
            for(int j = i + 1; j < tile_dim; j++)
            {
                // Load L_ji into sC（后续按转置方法使用）, Load y_j into sD
                T* L_ji = tile<NTILES>(L_block, L_stride, j, i);
                T* x_j = x_block + j * NTILES;
                LoadTile<NTILES, NTHREADS, T>(L_ji, sC, L_stride, tile_stride, false);
                LoadVecTile<NTILES, NTHREADS, T>(x_j, sD);
                // 计算b_i -= L_ji^T * x_j
                GemvSubLowerTTile<NTILES, NTHREADS, T>(sB, sC, sD, tile_stride);
            }

            // Step 4: Solve L_ii^T * x_i = b for x_i
            TrsvUpperTileInplace<NTILES, NTHREADS, T>(sA, sB, tile_stride);

            // Step 5: Store the result back to x_i
            StoreVecTile<NTILES, NTHREADS, T>(sB, x_i);
        }
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
        extern __shared__ T sA[];
        // Load block diagonal environment matrix into shared memory
        for (int idx = threadIdx.x; idx < block_size * block_size; idx += NTHREADS) 
        {
            int r = idx / block_size;
            int c = idx % block_size;
            sA[r * block_size + c] = A[block_offset + r * block_size + c];
        }
        __syncthreads();
        for(int j = 0; j < block_size; j++)
        {
            // thread 0 computes the diagonal element L_ii
            if(thread_id == 0)
            {
                // L_jj = sqrt(A_jj - sum_j_{j<i}(L_ji^2))
                T sum = sA[j + j * block_size];
                for (int i = 0; i < j; ++i) 
                {
                    T v = sA[j * block_size + i];
                    sum -= v * v;
                }
                sA[j + j * block_size] = sqrt(sum);
            }
            __syncthreads();
            T L_ii = sA[j * block_size + j];
            // threads 1...i compute the off-diagonal elements L_ij for j < i
            // L_ij = (A_ij - sum_k_{k < j}(L_ik * L_jk)) / L_jj
            for (int i = j + 1 + thread_id; i < block_size; i += NTHREADS)
            {
                T sum = sA[i * block_size + j];
                for (int k = 0; k < j; ++k) 
                {
                    sum -= sA[i * block_size + k] * sA[j * block_size + k];
                }
                sA[i * block_size + j] = sum / L_ii;
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
                L[block_offset + r * block_size + c] = sA[r * block_size + c];
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
        
        extern __shared__ T smem[];
        T* sL = smem;                         // size: MAX_N * MAX_N
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
        
        extern __shared__ T smem[];
        T* sL = smem;                         // size: MAX_N * MAX_N
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

    // A: input symmetric positive definite matrices
    // L: output batch of lower triangular matrices
    // block_sizes: array of block sizes for each matrix, with size (num_blocks)
    // block_offsets: array of starting offsets for each block in the A and L arrays, with size (num_blocks)
    // num_blocks: number of environments in the batch
	template<typename T>
	__global__ void BatchBlockCholeskyFactorize(
	    const DArray<T> A, 
        DArray<T> L, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets, 
        int num_blocks)
	{
        return;
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
    void BatchCholeskyFactorizeHost(
        const DArray<T> A, 
        DArray<T> L, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets, 
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        BatchCholeskyFactorize<T><<<blocks, threads>>>(A.begin(), L.begin(), block_sizes.begin(), block_offsets.begin(), num_blocks);
        cudaDeviceSynchronize();
    }

    template<typename T>
    void BatchCholeskyFactorizeHost(
        const T* A, 
        T* L, 
        const int* block_sizes, 
        const int* block_offsets, 
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        BatchCholeskyFactorize<T><<<blocks, threads>>>(A, L, block_sizes, block_offsets, num_blocks);
        cudaDeviceSynchronize();
    }

    template<typename T>
    void BatchCholeskySolveHost(
        const DArray<T> L, 
        DArray<T> x, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets, 
        DArray<int> x_offsets, 
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        BatchCholeskySolve<T><<<blocks, threads>>>(L.begin(), x.begin(), block_sizes.begin(), block_offsets.begin(), x_offsets.begin(), num_blocks);
        cudaDeviceSynchronize();
    }


    template<typename T>
    void BatchCholeskySolveHost(
        const T* L, 
        T* x, 
        const int* block_sizes, 
        const int* block_offsets, 
        const int* x_offsets, 
        int num_blocks)
    {
        const int threads = 128;
        const int blocks = (num_blocks + threads - 1) / threads;
        BatchCholeskySolve<T><<<blocks, threads>>>(L, x, block_sizes, block_offsets, x_offsets, num_blocks);
        cudaDeviceSynchronize();
    }

    template<typename T>
    void BlockCholeskySingleTileHost(
        const DArray<T> A, 
        DArray<T> L, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets, 
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


        CUDA_CHECK(cudaFuncSetAttribute(
        CholeskyFactorizeSingleTile<MAX_N, NTHREADS, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
        
        CUDA_LAUNCH_AND_CHECK((CholeskyFactorizeSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                A.begin(),
                L.begin(),
                block_offsets.begin(),
                block_sizes.begin(),
                num_blocks)));
    }

    template<typename T>
    void BlockCholeskySolveSingleTileHost(
        const DArray<T> L, 
        DArray<T> x, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets,
        DArray<int> x_offsets, 
        int num_blocks)
    {
        constexpr int MAX_N = 96;
        constexpr int NTHREADS = 128;
        size_t smem_bytes = MAX_N * MAX_N * sizeof(T) + MAX_N * sizeof(T);
        const int blocks = num_blocks;

        CUDA_CHECK(cudaFuncSetAttribute(
        LowerSolveInplaceSingleTile<MAX_N, NTHREADS, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

        CUDA_LAUNCH_AND_CHECK((LowerSolveInplaceSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                L.begin(),
                x.begin(),
                block_offsets.begin(),
                block_sizes.begin(),
                x_offsets.begin(),
                num_blocks)));

        CUDA_CHECK(cudaFuncSetAttribute(
        UpperSolveInplaceSingleTile<MAX_N, NTHREADS, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

        CUDA_LAUNCH_AND_CHECK((UpperSolveInplaceSingleTile<MAX_N, NTHREADS, T>
            <<<blocks, NTHREADS, smem_bytes>>>(
                L.begin(),
                x.begin(),
                block_offsets.begin(),
                block_sizes.begin(),
                x_offsets.begin(),
                num_blocks)));
    }

    template<typename T>
    void UniformBlockCholeskyFactorizeWithTileHost(T* A, int uniform_block_size, int num_blocks)
    {
        constexpr int NTILES = 32; // tile size of 32x32
        constexpr int NTHREADS = 128; // number of threads per block
        size_t smem_bytes = 4 * NTILES * NTILES * sizeof(T); // shared memory size for 4 tiles

        CUDA_LAUNCH_AND_CHECK((CholeskyFactorizeUniformBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                A, uniform_block_size, num_blocks)));
    }

    template<typename T>
    void UniformBlockCholeskySolveWithTileHost(T* L, T* x, int uniform_block_size, int num_blocks)
    {
        constexpr int NTILES = 32; // tile size of 32x32
        constexpr int NTHREADS = 128; // number of threads per block
        size_t smem_bytes = 2 * (NTILES * NTILES  + NTILES) * sizeof(T); // shared memory size

        CUDA_LAUNCH_AND_CHECK((LowerSolveUniformBlockTile<NTILES, NTHREADS, T>
            <<<num_blocks, NTHREADS, smem_bytes>>>(
                L, x, uniform_block_size, num_blocks)));
        
        // CUDA_LAUNCH_AND_CHECK((UpperSolveUniformBlockTile<NTILES, NTHREADS, T>
        //     <<<num_blocks, NTHREADS, smem_bytes>>>(
        //         L, x, uniform_block_size, num_blocks)));
    }



    template void dyno::BatchCholeskyFactorizeHost<float>(
        const dyno::DArray<float>, dyno::DArray<float>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskyFactorizeHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>,dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskyFactorizeHost<float>(
        const float*, float*, const int*, const int*, int);

    template void dyno::BatchCholeskyFactorizeHost<double>(
        const double*, double*, const int*, const int*, int);

    template void dyno::BatchCholeskySolveHost<float>(
        const dyno::DArray<float>, dyno::DArray<float>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskySolveHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskySolveHost<float>(
        const float*, float*, const int*, const int*, const int*, int);

    template void dyno::BatchCholeskySolveHost<double>(
        const double*, double*, const int*, const int*, const int*, int);

    template void dyno::BlockCholeskySingleTileHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, int);
    
    template void dyno::BlockCholeskySolveSingleTileHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::UniformBlockCholeskyFactorizeWithTileHost<double>(
        double*, int, int);
    
    template void dyno::UniformBlockCholeskySolveWithTileHost<double>(
        double*, double*, int, int);
} // namespace dyno