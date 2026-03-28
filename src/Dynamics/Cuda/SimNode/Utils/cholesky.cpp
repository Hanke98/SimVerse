#include "cholesky.h"
#include <cstdio>
#include <type_traits>

namespace dyno
{
    namespace
    {
        inline bool CheckCuSolverStatus(cusolverStatus_t status, const char* tag)
        {
            if (status != CUSOLVER_STATUS_SUCCESS)
            {
                std::printf("[CuSolver] %s failed with status = %d\n", tag, static_cast<int>(status));
                return false;
            }
            return true;
        }

        inline bool CheckCuBlasStatus(cublasStatus_t status, const char* tag)
        {
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::printf("[CuBLAS] %s failed with status = %d\n", tag, static_cast<int>(status));
                return false;
            }
            return true;
        }
    } // namespace

    template<typename T>
    CuSolverCholeskyRunner<T>::CuSolverCholeskyRunner() = default;

    template<typename T>
    CuSolverCholeskyRunner<T>::~CuSolverCholeskyRunner()
    {
        Release();
    }

    template<typename T>
    bool CuSolverCholeskyRunner<T>::Initialize()
    {
        Release();

        if (!CheckCuSolverStatus(cusolverDnCreate(&solver_), "cusolverDnCreate"))
        {
            Release();
            return false;
        }

        if (!CheckCuBlasStatus(cublasCreate(&blas_), "cublasCreate"))
        {
            Release();
            return false;
        }

        cuSafeCall(cudaMalloc(&d_info_, sizeof(int) * kMaxNumBlocks));

        max_num_blocks_ = kMaxNumBlocks;
        initialized_ = true;
        return true;
    }

    template<typename T>
    void CuSolverCholeskyRunner<T>::Release()
    {
        if (d_info_)
        {
            cuSafeCall(cudaFree(d_info_));
            d_info_ = nullptr;
        }

        if (blas_)
        {
            cublasDestroy(blas_);
            blas_ = nullptr;
        }
        if (solver_)
        {
            cusolverDnDestroy(solver_);
            solver_ = nullptr;
        }

        max_num_blocks_ = 0;
        initialized_ = false;
    }

    template<typename T>
    bool CuSolverCholeskyRunner<T>::IsInitialized() const
    {
        return initialized_;
    }

    template<typename T>
    int CuSolverCholeskyRunner<T>::MaxNumBlocks() const
    {
        return max_num_blocks_;
    }

    template<typename T>
    void CuSolverCholeskyRunner<T>::Factorize(
        T* A,
        T** dA_ptr,
        int block_size,
        int num_blocks,
        int* d_info,
        bool check_validity)
    {
        // if (!initialized_)
        // {
        //     std::printf("[CuSolverCholeskyRunner::Factorize] runner is not initialized\n");
        //     return;
        // }
        // if (A == nullptr || dA_ptr == nullptr)
        // {
        //     std::printf("[CuSolverCholeskyRunner::Factorize] A/dA_ptr is nullptr\n");
        //     return;
        // }
        // if (check_validity && (block_size <= 0 || num_blocks <= 0))
        // {
        //     std::printf("[CuSolverCholeskyRunner::Factorize] invalid block_size/num_blocks\n");
        //     return;
        // }
        // if (d_info == nullptr && check_validity && num_blocks > max_num_blocks_)
        // {
        //     std::printf("[CuSolverCholeskyRunner::Factorize] num_blocks exceeds internal d_info capacity\n");
        //     return;
        // }

        int* d_info_use = (d_info != nullptr) ? d_info : d_info_;

        bool ok = true;
        if constexpr (std::is_same_v<T, double>)
        {
            ok = CheckCuSolverStatus(
                cusolverDnDpotrfBatched(
                    solver_, CUBLAS_FILL_MODE_LOWER, block_size, dA_ptr, block_size, d_info_use, num_blocks),
                "cusolverDnDpotrfBatched");
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            ok = CheckCuSolverStatus(
                cusolverDnSpotrfBatched(
                    solver_, CUBLAS_FILL_MODE_LOWER, block_size, dA_ptr, block_size, d_info_use, num_blocks),
                "cusolverDnSpotrfBatched");
        }
        else
        {
            std::printf("[CuSolverCholeskyRunner::Factorize] only supports float/double\n");
            ok = false;
        }

        if (!ok)
        {
            std::printf("[CuSolverCholeskyRunner::Factorize] potrf batched failed\n");
        }
    }

    template<typename T>
    void CuSolverCholeskyRunner<T>::Solve(
        const T* L,
        T* x,
        const T** dA_ptr,
        T** dB_ptr,
        int block_size,
        int num_blocks,
        bool check_validity)
    {
        // if (!initialized_)
        // {
        //     std::printf("[CuSolverCholeskyRunner::Solve] runner is not initialized\n");
        //     return;
        // }
        // if (L == nullptr || x == nullptr || dA_ptr == nullptr || dB_ptr == nullptr)
        // {
        //     std::printf("[CuSolverCholeskyRunner::Solve] L/x/dA_ptr/dB_ptr is nullptr\n");
        //     return;
        // }
        // if (check_validity && (block_size <= 0 || num_blocks <= 0))
        // {
        //     std::printf("[CuSolverCholeskyRunner::Solve] invalid block_size/num_blocks\n");
        //     return;
        // }

        const T alpha = static_cast<T>(1);
        bool ok = true;
        if constexpr (std::is_same_v<T, double>)
        {
            ok = CheckCuBlasStatus(
                cublasDtrsmBatched(
                    blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    block_size, 1, &alpha, dA_ptr, block_size, dB_ptr, block_size, num_blocks),
                "cublasDtrsmBatched(N)");
            if (ok)
            {
                ok = CheckCuBlasStatus(
                    cublasDtrsmBatched(
                        blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        block_size, 1, &alpha, dA_ptr, block_size, dB_ptr, block_size, num_blocks),
                    "cublasDtrsmBatched(T)");
            }
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            ok = CheckCuBlasStatus(
                cublasStrsmBatched(
                    blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    block_size, 1, &alpha, dA_ptr, block_size, dB_ptr, block_size, num_blocks),
                "cublasStrsmBatched(N)");
            if (ok)
            {
                ok = CheckCuBlasStatus(
                    cublasStrsmBatched(
                        blas_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        block_size, 1, &alpha, dA_ptr, block_size, dB_ptr, block_size, num_blocks),
                    "cublasStrsmBatched(T)");
            }
        }
        else
        {
            std::printf("[CuSolverCholeskyRunner::Solve] only supports float/double\n");
            ok = false;
        }

        if (!ok)
        {
            std::printf("[CuSolverCholeskyRunner::Solve] trsm batched failed\n");
        }
    }

    template class CuSolverCholeskyRunner<float>;
    template class CuSolverCholeskyRunner<double>;
} // namespace dyno
