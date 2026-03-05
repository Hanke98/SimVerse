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



    DEFINE_CLASS(SimModule);
}// namespace dyno