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
        
        
        spdlog::info("===================   Frame {} Ended     ===================\n", frame++);
    }

    template<typename TDataType>
    void SimModule<TDataType>::Init()
    {
        auto rigid_body = in_rigid_body.constDataPtr();
        if (rigid_body == nullptr)
        {
            spdlog::warn("SimModule::Init called before in_rigid_body is connected.");
            return;
        }

        solver = std::make_shared<MujocoSolver<TDataType>>(rigid_body);
    }

    DEFINE_CLASS(SimModule);
}// namespace dyno