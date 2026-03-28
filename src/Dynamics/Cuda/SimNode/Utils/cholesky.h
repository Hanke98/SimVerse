#pragma once
#include "Array/Array.h"
#include "utils.h"

namespace dyno
{
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

	void TestI();
	void TestII();
} // namespace dyno