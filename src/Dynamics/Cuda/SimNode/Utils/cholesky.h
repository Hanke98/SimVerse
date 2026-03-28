#pragma once
#include "Array/Array.h"
#include "utils.h"

namespace dyno
{
	// cholesky method
	enum class CholeskyMethod : int
    {
        Simplest = 0,
        SingleTiled = 1,
        UniformTiled = 2,
        PaddedTiled = 3
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

	// block cholesky factorization
	template<typename T>
	__global__ void BatchBlockCholeskyFactorize(
	    const DArray<T> A, DArray<T> L, DArray<int> block_sizes, DArray<int> block_offsets, int num_blocks);

	// host function for batch Cholesky factorization
	template<typename T>
	void BatchCholeskyFactorizeHost(const DArray<T> A, DArray<T> L, DArray<int> block_sizes, DArray<int> block_offsets, int num_blocks);

    template<typename T>
	void BatchCholeskyFactorizeHost(const T* A, T* L, const int* block_sizes, const int* block_offsets, int num_blocks);

	// host function for batch Cholesky solve
	template<typename T>
	void BatchCholeskySolveHost(
	    const DArray<T> L, DArray<T> x, DArray<int> block_sizes, DArray<int> block_offsets, DArray<int> x_offsets, int num_blocks);

    template<typename T>
	void BatchCholeskySolveHost(
	    const T* L, T* x, const int* block_sizes, const int* block_offsets, const int* x_offsets, int num_blocks);

	// non-batched cholesky factorization for a single block
	template<typename T>
	void BlockCholeskyHost(const DArray<T> A, DArray<T> L);

	template<typename T>
	void BlockCholeskySingleTileHost(const DArray<T> A, DArray<T> L, DArray<int> block_sizes, DArray<int> block_offsets, int num_blocks);

	template<typename T>
	void BlockCholeskySolveSingleTileHost(
        const DArray<T> L, DArray<T> x, DArray<int> block_sizes, DArray<int> block_offsets, DArray<int> x_offsets, int num_blocks);

    template<typename T>
    void UniformBlockCholeskyFactorizeWithTileHost(T* A, int uniform_block_size, int num_blocks);

    template<typename T>
    void UniformBlockCholeskySolveWithTileHost(T* L, T* x, int uniform_block_size, int num_blocks);

    // block-wise Cholesky with non-uniform block sizes (tile path with zero/identity padding)
    template<typename T>
    void BatchBlockCholeskyFactorize(
        const T* A,
        T* L,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks);

    template<typename T>
    void BatchBlockCholeskyFactorizeHost(
        const T* A,
        T* L,
        const int* block_sizes,
        const int* block_offsets,
        int num_blocks);

    // block-wise Cholesky solve with non-uniform block sizes (tile path with padding)
    template<typename T>
    void BatchBlockCholeskySolve(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks);

    template<typename T>
    void BatchBlockCholeskySolveHost(
        const T* L,
        T* x,
        const int* block_sizes,
        const int* block_offsets,
        const int* x_offsets,
        int num_blocks);

	void TestI();
	void TestII();
    void TestIII();
    void TestIV();
    void TestVariableBlockCholeskyFactorize(int num_blocks = 50);
    void TestVariableBlockCholeskyFactorizeAndSolve(int num_blocks = 50);
    void TestLowerSolve();
    void TestUpperSolve();
} // namespace dyno
