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

}