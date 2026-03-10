#include "MujocoSolver.h"
#include <spdlog/spdlog.h>

namespace dyno
{
    __global__ void TimeIntegrationKernal(DArray2D<Vec3f> batch_pos, DArray2D<Mat3f> batch_rot, DArray<int> num_bodies, int num_envs)
    {
        int env_id = blockIdx.x;
        int body_idx = threadIdx.x;
        if (env_id >= num_envs)
            return;

        int env_self_bodies = num_bodies[env_id];
        if (body_idx >= env_self_bodies)
            return;

        Vec3f& pos = batch_pos(env_id, body_idx);
        pos += Vec3f(0.f, 0.003f, 0.f);

        Mat3f& rot = batch_rot(env_id, body_idx);
        if (env_id == 0 && body_idx == 0)
        {
            printf("Before rotation update: rot(%d, %d) = \n", env_id, body_idx);
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    printf("%f ", rot(i, j));
                printf("\n");
            }
        }
        Mat3f delta_rot = Mat3f::identityMatrix();
        delta_rot(1, 1) = cosf(1.f);
        delta_rot(1, 2) = -sinf(1.f);
        delta_rot(2, 1) = sinf(1.f);
        delta_rot(2, 2) = cosf(1.f);
        rot = delta_rot * rot;


        printf("Env %d, Body %d, Position: (%f, %f, %f)\n", env_id, body_idx, pos.x, pos.y, pos.z);
    }
}


namespace dyno
{
    template<typename TDataType>
    void MujocoSolver<TDataType>::Init()
    {
        spdlog::info("[MujocoSolver Solver] Starting initialization.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        for(int eid = 0; eid < env_infos->num_envs; eid++)
        {
            spdlog::info("Environment {}: ", eid);
            
        }

        spdlog::info("[MujocoSolver Solver] Initialization complete. Number of environments: {}", env_infos->num_envs);



        spdlog::info("[MujocoSolver Solver] Finished initialization.");
    }


    template<typename TDataType>
    void MujocoSolver<TDataType>::TimeIntegration()
    {
        spdlog::info("[MujocoSolver Solver] TimeIntegration called.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        TimeIntegrationKernal<<<env_infos->num_envs, 16>>>(
            rigid_body_system->batch_pos, rigid_body_system->batch_rot, rigid_body_system->batch_bodies, env_infos->num_envs);
        cudaDeviceSynchronize();
    }


    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}