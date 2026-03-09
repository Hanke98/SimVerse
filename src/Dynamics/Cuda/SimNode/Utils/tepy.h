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
        int num_env = 0;
        FArray<Vec3f, DeviceType::GPU> gravities;
        FArray<Real, DeviceType::GPU> timesteps;
    };
}