#include "SimNode/Utils/utils.h"
#include "cholesky.h"




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

    // tile_offset表示tile在shared memory中每行的存储offset
    // A_offset表示A在global memory中每行的存储offset，不考虑padding的情况就是A的列数
    // 统一采用row-major存储
    // 模板参数，NTILES表示tile的行数，NTHREADS表示线程块大小，T表示数据类型
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void LoadTile(const T* A_kk, T* sA, int A_offset, int tile_offset)
    {
        // 线程块大小为 NTHREADS，每个线程加载一个元素
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES * NTILES; i += NTHREADS)
        {
            int row = i / NTILES;
            int col = i % NTILES;
            bool is_lower_triangular = (row >= col);
            // 只加载下三角部分，或者加载整个块但上三角部分置零
            if (is_lower_triangular)
                sA[row * tile_offset + col] = A_kk[row * A_offset + col];
        }
        __syncthreads();
    }

    template<unsigned NTILES, unsigned NTHREADS, class T>
    __device__ void StoreTile(const T* sA, T* L_kk, int L_offset, int tile_offset)
    {
        const int tid = threadIdx.x;
        for (int i = tid; i < NTILES * NTILES; i += NTHREADS)
        {
            int row = i / NTILES;
            int col = i % NTILES;
            bool is_lower_triangular = (row >= col);
            if (is_lower_triangular)
                L_kk[row * L_offset + col] = sA[row * tile_offset + col];
        }
        __syncthreads();
    }

    // 默认row-major
    // 这个函数是为了得到A中对应tile在global_memory中的起始地址
    template<unsigned NTILES, class T>
    __device__ T* tile(T* A, unsigned A_offset, unsigned i, unsigned j) 
    {
        return A + i * NTILES * A_offset + j * NTILES;
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

    // __launch_bounds__指定了每个block的最大线程数为NTHREADS，便于编译器优化
    template<unsigned NTILES, unsigned NTHREADS, class T>
    __global__  __launch_bounds__(NTHREADS) void CholeskyFactorizeTile(const T* A, T* L, const int* block_offsets, const int* block_sizes, int num_blocks)
    {
        
        
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
        const DArray<T> A, DArray<T> L, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets, 
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
        const DArray<T> L, DArray<T> x, 
        DArray<int> block_sizes, 
        DArray<int> block_offsets,
        DArray<int> x_offsets, 
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


    template void dyno::BatchCholeskyFactorizeHost<float>(
        const dyno::DArray<float>, dyno::DArray<float>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskyFactorizeHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>,dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskySolveHost<float>(
        const dyno::DArray<float>, dyno::DArray<float>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BatchCholeskySolveHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);

    template void dyno::BlockCholeskySingleTileHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, int);
    
    template void dyno::BlockCholeskySolveSingleTileHost<double>(
        const dyno::DArray<double>, dyno::DArray<double>, dyno::DArray<int>, dyno::DArray<int>, dyno::DArray<int>, int);
} // namespace dyno