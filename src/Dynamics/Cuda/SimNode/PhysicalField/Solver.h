#pragma once
#include "Object.h"
#include "DataTypes.h"

namespace dyno
{
    template<typename TDataType>
    class SolverBase
    {
    public:
        SolverBase() {};
        ~SolverBase() {};

        virtual void TimeIntegration() = 0;
    };
}
