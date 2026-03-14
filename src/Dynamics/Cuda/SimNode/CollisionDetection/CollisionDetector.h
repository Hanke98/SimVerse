#pragma once

#include <Array/Array.h>
#include <Array/Array2D.h>

namespace dyno {

    template<typename TDataType>
    class ZJUCollisionDetector
    {
    public:
        ZJUCollisionDetector<TDataType>() {};
        ~ZJUCollisionDetector<TDataType>() {};


    
    protected:
        DArray<int>              broad_phase_flags;
        DArray<int>              broad_phase_offsets;
        DArray<int>              narrow_phase_flags;
        DArray<int>              narrow_phase_offsets;

        

    };

}