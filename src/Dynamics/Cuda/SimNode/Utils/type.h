#pragma once

#include "Vector.h"
#include "Array/Array.h"
#include "Field.h"

namespace dyno
{
    template<typename TDataType>
    struct EnvironmentInfos
    {
        using Real = typename TDataType::Real;
        int num_envs = 0;
        int max_constraints = 512;
        DArray<Vec3f> gravities;
        DArray<Real> timesteps;
    };
}