#include "SimNode.h"

namespace dyno {

    __global__ void PrintBatchNvKernel(dyno::DArray<int> batch_nv)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_nv.size())
        {
            printf("batch_nv[%d] = %d\n", idx, batch_nv[idx]);
        }
    }

    template<typename Real>
    __global__ void PrintBatchQaccKernel(dyno::DArray2D<Real> batch_qacc)
    {
        uint env_id = blockIdx.x;
        uint dof_idx = threadIdx.x;

        if (env_id < batch_qacc.nx() && dof_idx < batch_qacc.ny())
            printf("batch_qacc(%u, %u) = %f\n", env_id, dof_idx, batch_qacc(env_id, dof_idx));
    }

    template<typename TDataType>
    void SimNode<TDataType>::InitRigidBody(int num_env, int num_bodies)
    {
        spdlog::info("Start initializing rigid body state variables.");

        // batch_nv test
        std::vector<int> batch_nv_host;
        for(int eid = 0; eid < num_env; eid++)
        {
            batch_nv_host.push_back(num_bodies * (eid + 1));
            spdlog::info("batch_nv_host[{}] = {}", eid, batch_nv_host[eid]);
        }
            

        // batch_qacc test
        CArray2D<Real> batch_qacc_host(num_env, num_bodies * 6);
        for(int eid = 0; eid < num_env; eid++)
        {
            for(int bid = 0; bid < num_bodies; bid++)
                for(int i = 0; i < 6; i++)
                {
                    int dof_id = bid * 6 + i;
                    batch_qacc_host(eid, bid * 6 + i) = (10000*eid + dof_id) * 1.0f;
                    spdlog::info("batch_qacc_host({},{}) = {}", eid, bid * 6 + i, batch_qacc_host(eid, bid * 6 + i));
                }
                    
        }
            

        auto rigid_body = var_rigid_body.getValue();
        rigid_body.batch_nv.resize(num_env);
        rigid_body.batch_nv.assign(batch_nv_host);

        rigid_body.batch_qacc.resize(num_env, num_bodies * 6);
        rigid_body.batch_qacc.assign(batch_qacc_host);

        cuExecute(rigid_body.batch_nv.size(),
            PrintBatchNvKernel,
            rigid_body.batch_nv);

        // cuExecute2D(make_uint2(rigid_body.batch_qacc.nx(), rigid_body.batch_qacc.ny()),
        //     PrintBatchQaccKernel<typename TDataType::Real>,
        //     rigid_body.batch_qacc);
        PrintBatchQaccKernel<Real><<<rigid_body.batch_qacc.nx(), rigid_body.batch_qacc.ny()>>>(rigid_body.batch_qacc);

        var_rigid_body.setValue(rigid_body);
    }

    DEFINE_CLASS(SimNode);
}// dyno