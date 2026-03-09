#include "SimModule.h"
#include "Utils/utils.h"

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

    __global__ void UpdatePrimiteiveRigidBodyRenderingDataKernel(DArray<Vec3f> pos_cache, DArray<Mat3f> rot_cache,
        DArray2D<Vec3f> batch_pos, DArray2D<Mat3f> batch_rot, DArray<int> batch_bodies, int num_envs)
    {
        int env_id = blockIdx.x;
        int local_bid = threadIdx.x;

        if(env_id >= num_envs || local_bid >= batch_bodies[env_id])
            return;


        
    }

}// for cuda kernels




namespace dyno
{
    template<typename TDataType>
    void SimModule<TDataType>::AdvanceOneStep()
    {
        spdlog::info("AdvanceOneStep function called.");


        // TODO: write a kernal to add 1.0 to each element in batch_qacc, and print the updated values in the kernel.
        auto rigid_body = in_rigid_body.constDataPtr();
        AddOneToBatchQaccKernal<<<rigid_body->batch_qacc.nx(), rigid_body->batch_qacc.ny()>>>
        (rigid_body->batch_qacc);
        cudaDeviceSynchronize();

        solver->TimeIntegration();
    }

    template<typename TDataType>
    void SimModule<TDataType>::UpdateRenderingData()
    {
        // 1. Update rigid body rendering data (primitive part)
        auto topo = TypeInfo::cast<DiscreteElements<TDataType>>(this->intopology()->getDataPtr());
        auto rigid_body = in_rigid_body.constDataPtr();

        if (topo == nullptr || rigid_body == nullptr)
        {
            spdlog::info("SimModule::UpdateRenderingData skipped: topology or rigid_body is null.");
            return;
        }

        int total_rigid_bodies = rigid_body->topo_pos_cache.size();
        if (total_rigid_bodies <= 0)
        {
            spdlog::info("SimModule::UpdateRenderingData skipped: topo cache is empty.");
            return;
        }

        // Re-flatten per-env simulation data into render-transform buffers.
        FlattenArray2D(
            rigid_body->batch_pos,
            rigid_body->topo_pos_cache,
            total_rigid_bodies,
            rigid_body->batch_bodies,
            rigid_body->batch_body_offset);

        FlattenArray2D(
            rigid_body->batch_rot,
            rigid_body->topo_rot_cache,
            total_rigid_bodies,
            rigid_body->batch_bodies,
            rigid_body->batch_body_offset);


        // TODO: 2. Update rigid body rendering data(mesh part)

        // TODO: 3. Update soft body rendering data

        topo->setPosition(rigid_body->topo_pos_cache);
        topo->setRotation(rigid_body->topo_rot_cache);
        topo->update();
    }

    DEFINE_CLASS(SimModule);
}