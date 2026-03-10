#include "SimModule.h"

namespace dyno
{
    IMPLEMENT_TCLASS(SimModule, TDataType)

    template<typename TDataType>
    SimModule<TDataType>::SimModule() : ComputeModule()
    {
        spdlog::info("SimModule constructor called.");

        
    }

    template<typename TDataType>
	SimModule<TDataType>::~SimModule()
	{
	}

    template<typename TDataType>
    void SimModule<TDataType>::compute()
    {
        spdlog::info("===================   Frame {} Started   ===================", frame);
        spdlog::info("SimModule compute function called.");
        AdvanceOneStep();
        UpdateRenderingData();
        
        spdlog::info("===================   Frame {} Ended     ===================\n", frame++);
    }

    template<typename TDataType>
    void SimModule<TDataType>::Init()
    {
        auto env_infos = in_env_infos.constDataPtr();
        auto rigid_body = in_rigid_body.constDataPtr();
        if (env_infos == nullptr || rigid_body == nullptr)
        {
            spdlog::warn("SimModule::Init called before env_infos/rigid_body are connected.");
            return;
        }

        solver = std::make_shared<MujocoSolver<TDataType>>(env_infos, rigid_body);
        solver->Init();

    }

    DEFINE_CLASS(SimModule);
}// namespace dyno