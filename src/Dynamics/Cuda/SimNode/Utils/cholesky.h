#pragma once
#include "utils.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dyno
{
	// cholesky method
    enum class CholeskyMethod : int
    {
        Simplest = 0,
        SingleTiled = 1,
        UniformTiled = 2,
        PaddedTiled = 3,
		WavefrontTiled = 4
    };

	// host function
	template<typename T>
    void CholeskyFactorizeHost(
        const T* A,
        T* L,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks,
        CholeskyMethod method);

    template<typename T>
    void CholeskySolveHost(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks,
        CholeskyMethod method);

    // cuSolver/cuBLAS specialized interface:
    // A/x are contiguous batched buffers:
    // A: [num_blocks, block_size, block_size] (in-place factorization)
    // x  : [num_blocks, block_size]
    // Notes for row-major users:
    // cuSolver/cuBLAS follow column-major semantics. The numeric result is valid,
    // but matrix layout interpretation is transposed relative to row-major access.
    template<typename T>
    class CuSolverCholeskyRunner
    {
    public:
        CuSolverCholeskyRunner();
        ~CuSolverCholeskyRunner();

        CuSolverCholeskyRunner(const CuSolverCholeskyRunner&) = delete;
        CuSolverCholeskyRunner& operator=(const CuSolverCholeskyRunner&) = delete;

        bool Initialize();
        void Release();

        bool IsInitialized() const;
        int MaxNumBlocks() const;

        void Factorize(
            T* A,
            T** dA_ptr,
            int block_size,
            int num_blocks,
            int* d_info = nullptr,
            bool check_validity = false);

        void Solve(
            const T* L,
            T* x,
            const T** dA_ptr,
            T** dB_ptr,
            int block_size,
            int num_blocks,
            bool check_validity = false);

    private:
        static constexpr int kMaxNumBlocks = 10000;
        cusolverDnHandle_t solver_ = nullptr;
        cublasHandle_t blas_ = nullptr;
        int* d_info_ = nullptr;
        int max_num_blocks_ = 0;
        bool initialized_ = false;
    };

} // namespace dyno
