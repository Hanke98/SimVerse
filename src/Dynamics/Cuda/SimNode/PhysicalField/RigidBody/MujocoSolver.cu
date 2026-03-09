#include "MujocoSolver.h"
#include <spdlog/spdlog.h>

namespace dyno
{
    template<typename TDataType>
    void MujocoSolver<TDataType>::TimeIntegration()
    {
        spdlog::info("MujocoSolver TimeIntegration called.");
    }

    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}