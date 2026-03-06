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



        // ========================= TEST FUNCs =========================
        auto AddShapes = [&](RigidBody<TDataType>& rigid_bodies){
            const int slots_per_env = 2;
            CArray2D<int> shape_type_host(num_env, slots_per_env);
            CArray2D<int> shape_idx_host(num_env, slots_per_env);
            CArray2D<BoxInfo> boxes_host(num_env, slots_per_env);
            CArray2D<SphereInfo> spheres_host(num_env, slots_per_env);

            for (int eid = 0; eid < num_env; ++eid)
            {
                for (int sid = 0; sid < slots_per_env; ++sid)
                {
                    shape_type_host(eid, sid) = -1;
                    shape_idx_host(eid, sid) = -1;
                }
            }

            if (num_env > 0)
            {
                shape_type_host(0, 0) = 0;
                shape_type_host(0, 1) = 0;
                shape_idx_host(0, 0) = 0;
                shape_idx_host(0, 1) = 1;

                boxes_host(0, 0).center = Vector<Real, 3>(Real(-0.8), Real(0.0), Real(0.0));
                boxes_host(0, 0).halfLength = Vector<Real, 3>(Real(0.2), Real(0.3), Real(0.4));

                boxes_host(0, 1).center = Vector<Real, 3>(Real(0.8), Real(0.0), Real(0.0));
                boxes_host(0, 1).halfLength = Vector<Real, 3>(Real(0.5), Real(0.2), Real(0.2));
            }

            if (num_env > 1)
            {
                shape_type_host(1, 0) = 1;
                shape_type_host(1, 1) = 1;
                shape_idx_host(1, 0) = 0;
                shape_idx_host(1, 1) = 1;

                spheres_host(1, 0).center = Vector<Real, 3>(Real(-0.5), Real(0.0), Real(0.0));
                spheres_host(1, 0).radius = Real(0.25);

                spheres_host(1, 1).center = Vector<Real, 3>(Real(0.5), Real(0.0), Real(0.0));
                spheres_host(1, 1).radius = Real(0.45);
            }

            rigid_bodies.shape_type.assign(shape_type_host);
            rigid_bodies.shape_idx.assign(shape_idx_host);
            rigid_bodies.boxes.assign(boxes_host);
            rigid_bodies.spheres.assign(spheres_host);
        };




        // ========================= TEST FUNCs =========================


        spdlog::info("Start initializing rigid body state variables.");
        using Real = typename TDataType::Real;

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