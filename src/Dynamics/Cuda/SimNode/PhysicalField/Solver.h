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

        virtual void Init() = 0;
        virtual void Step() = 0;
        virtual void TimeIntegration() = 0;
    };
}
