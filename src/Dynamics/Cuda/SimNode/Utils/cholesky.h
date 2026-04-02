#pragma once
#include "utils.h"
#include "SimBlockMatrix.h"
#include "SimBlockVector.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace dyno
{


	template<typename T>
	struct CholeskyGraphCache
	{
		cudaGraph_t graph = nullptr;
		cudaGraphExec_t exec = nullptr;

		T* A_ptr = nullptr;
		T* x_ptr = nullptr;
		int* is_converged = nullptr;
		int block_size = -1;
		int num_blocks = -1;
		cudaStream_t stream = nullptr;
		bool built = false;

		void release()
		{
			if (exec) { cuSafeCall(cudaGraphExecDestroy(exec)); exec = nullptr; }
			if (graph) { cuSafeCall(cudaGraphDestroy(graph)); graph = nullptr; }
			built = false;
			A_ptr = nullptr;
			x_ptr = nullptr;
			block_size = -1;
			num_blocks = -1;
			stream = nullptr;
		}
	};

	// cholesky method
    enum class CholeskyMethod : int
    {
        Simplest = 0,
        SingleTiled = 1,
        UniformTiled = 2,
        PaddedTiled = 3,
		WavefrontTiled = 4
    };

	template<typename T>
	class BatchedCholeskySolver
	{
	public:
		BatchedCholeskySolver();
		~BatchedCholeskySolver();

		BatchedCholeskySolver(const BatchedCholeskySolver&) = delete;
		BatchedCholeskySolver& operator=(const BatchedCholeskySolver&) = delete;

		// Allocate/reuse internal resources for up to these limits.
		bool Initialize(cudaStream_t stream = nullptr);

		void SetStream(cudaStream_t& stream){	stream_ = stream;	}

		void Release();

		bool IsInitialized() const;

		// In-place factorization: A -> L (stored in A)
		bool Factorize(
			T* A,
			int* is_converged,
			const int* block_sizes,
			const int* block_offsets,
			int num_blocks,
			CholeskyMethod method,
			int uniform_block_size = -1,
			bool use_graph = false);

        // overload for SimBlockMatrix:
        // infer block dimension n from rows/cols and require square blocks (rows == cols).
        bool Factorize(
            DevBlockMatrix<T>& A_blocks,
			DArray<int> is_converged,
            CholeskyMethod method,
            int uniform_block_size = -1,
            bool use_graph = false);

		bool FactorizeWavefrontWithGraph(
			T* A, 
			int* is_converged,
			const int block_size, 
			int num_blocks);

		// In-place solve on x: b -> x
		bool Solve(
			const T* L,
			T* x,
			int* is_converged,
			const int* block_sizes,
			const int* block_offsets,
			const int* x_offsets,
			int num_blocks,
			CholeskyMethod method,
			int uniform_block_size = -1);

		bool Solve(
            DevBlockMatrix<T>& L_blocks,
			DevBlockVector<T>& x_blocks,
			DArray<int> is_converged,
            CholeskyMethod method,
            int uniform_block_size = -1);

	private:
		bool initialized_ = false;
		cudaStream_t stream_ = nullptr;
		CholeskyGraphCache<T> factorize_graph_cache_;
		CholeskyGraphCache<T> solve_graph_cache_;
	};

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
