#pragma once
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
} // namespace dyno
