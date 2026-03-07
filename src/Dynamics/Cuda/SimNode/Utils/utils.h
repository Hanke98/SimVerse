#pragma once

#include "Array/Array.h"
#include "Array/Array2D.h"

namespace dyno
{
    template<typename T>
    void FlattenArray2D(const DArray2D<T>& src,  DArray<T>& dst, int total_count,
        const DArray<int>& array_lengths, const DArray<int>& array_offsets);
    
    template<typename T>
    void FlattenArray2D(const CArray2D<T>& src,  CArray<T>& dst, int total_count,
        const CArray<int>& array_lengths, const CArray<int>& array_offsets);

#ifdef __CUDACC__
    template<typename T>
    __global__ void FlattenArray2DKernel(
        DArray2D<T> src,
        DArray<T> dst,
        DArray<int> array_lengths,
        DArray<int> array_offsets,
        int total_count)
    {
        int env_id = blockIdx.x;
        if (env_id >= array_lengths.size())
            return;

        int len = array_lengths[env_id];
        int offset = array_offsets[env_id];
        for (int local_id = threadIdx.x; local_id < len; local_id += blockDim.x)
        {
            int out_id = offset + local_id;
            if (out_id >= 0 && out_id < total_count)
                dst[out_id] = src(env_id, local_id);
        }
    }

    template<typename T>
    inline void FlattenArray2D(const DArray2D<T>& src, DArray<T>& dst, int total_count,
        const DArray<int>& array_lengths, const DArray<int>& array_offsets)
    {
        dst.resize(total_count);
        if (total_count <= 0 || array_lengths.size() == 0)
            return;

        FlattenArray2DKernel<<<array_lengths.size(), 128>>>(
            src,
            dst,
            array_lengths,
            array_offsets,
            total_count);
    }
#endif

    template<typename T>
    inline void FlattenArray2D(const CArray2D<T>& src, CArray<T>& dst, int total_count,
        const CArray<int>& array_lengths, const CArray<int>& array_offsets)
    {
        dst.resize(total_count);
        for (uint env_id = 0; env_id < array_lengths.size(); ++env_id)
        {
            int len = array_lengths[env_id];
            int offset = array_offsets[env_id];
            for (int local_id = 0; local_id < len; ++local_id)
            {
                int out_id = offset + local_id;
                if (out_id >= 0 && out_id < total_count)
                    dst[out_id] = src(env_id, local_id);
            }
        }
    }

}