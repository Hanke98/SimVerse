#include "utils.h"

namespace dyno
{

    template<typename T>
    void FlattenArray2D(const DArray2D<T>& src,  DArray<T>& dst, int total_count,
        const DArray<int>& array_lengths, const DArray<int>& array_offsets)
    {
        dst.resize(total_count);
        
    }

}