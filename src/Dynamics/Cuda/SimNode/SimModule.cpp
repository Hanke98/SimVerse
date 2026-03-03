#include "SimModule.h"

namespace dyno
{
    IMPLEMENT_TCLASS(SimModule, TDataType)

    template<typename TDataType>
    void SimModule<TDataType>::compute()
    {
        std::cout << "SimModule compute function called." << std::endl << std::flush;
    }



    DEFINE_CLASS(SimModule);
}// namespace dyno