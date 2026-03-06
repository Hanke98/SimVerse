#include "SimModule.h"

namespace dyno
{
    __global__ void AddOneToBatchQaccKernal(dyno::DArray2D<Real> batch_qacc)
    {
        int env_id = blockIdx.x;
        int dof_idx = threadIdx.x;

        if(env_id >= batch_qacc.nx() || dof_idx >= batch_qacc.ny())
            return;

        batch_qacc(env_id, dof_idx) += 1.0f;
        printf("Updated batch_qacc(%d, %d) = %f\n", env_id, dof_idx, batch_qacc(env_id, dof_idx));
    }

    template<typename TDataType>
    void SimModule<TDataType>::AdvanceOneStep()
    {
        spdlog::info("AdvanceOneStep function called.");


        // TODO: write a kernal to add 1.0 to each element in batch_qacc, and print the updated values in the kernel.
        auto rigid_body = in_rigid_body.constDataPtr();
        AddOneToBatchQaccKernal<<<rigid_body->batch_qacc.nx(), rigid_body->batch_qacc.ny()>>>
        (rigid_body->batch_qacc);
        cudaDeviceSynchronize();
    }

    DEFINE_CLASS(SimModule);
}