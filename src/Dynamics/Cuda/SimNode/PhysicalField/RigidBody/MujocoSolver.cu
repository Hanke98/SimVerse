#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include "../../Utils/utils.h"

namespace dyno
{
    __global__ void TimeIntegrationKernel(DArray2D<Vec3f> batch_pos, DArray2D<Mat3f> batch_rot, DArray<int> num_bodies, int num_envs)
    {
        int env_id = blockIdx.x;
        int body_idx = threadIdx.x;
        if (env_id >= num_envs)
            return;

        int env_self_bodies = num_bodies[env_id];
        if (body_idx >= env_self_bodies)
            return;

        Vec3f& pos = batch_pos(env_id, body_idx);
        pos -= Vec3f(0.f, 0.003f, 0.f);

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
        delta_rot(1, 1) = cosf(0.5f);
        delta_rot(1, 2) = -sinf(0.5f);
        delta_rot(2, 1) = sinf(0.5f);
        delta_rot(2, 2) = cosf(0.5f);
        rot = delta_rot * rot;


        printf("Env %d, Body %d, Position: (%f, %f, %f)\n", env_id, body_idx, pos.x, pos.y, pos.z);
    }

    template<typename TDataType>
    __global__ void DofCountAndBuildIndexKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        auto& q_index = rigid_body_system.q_offset;
        auto& q_num = rigid_body_system.q_lengths;
        auto& qpos_index = rigid_body_system.qpos_offset;
        auto& qpos_num = rigid_body_system.qpos_lengths;

        printf("Env %d: num_bodies = %d\n", env_id, env_self_bodies);
        int nv = 0;         // num of generalised DoFs for this env.
        int nqpos = 0;   // num of qpos for this env.

        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int is_static = rigid_body_system.is_static(env_id, bid);

            if (parent_idx == -1)
            {
                if(is_static)
                    continue;   // Static root body, no DoFs

                q_index(env_id, bid) = nv;
                q_num(env_id, bid) = 6;
                qpos_index(env_id, bid) = nqpos;
                qpos_num(env_id, bid) = 7;
                nv += 6;
                nqpos += 7;
            }
            else
            {
                // TODO: For articulated bodies
            }
            printf("Env %d, Body %d, NV: %d\n", env_id, bid, nv);
            printf("Env %d, Body %d, Nqpos: %d\n", env_id, bid, nqpos);
            printf("Env %d, Body %d, cube q_index: %d\n", env_id, bid, q_index(env_id, bid));
            printf("Env %d, Body %d, cube q_num: %d\n", env_id, bid, q_num(env_id, bid));
            printf("Env %d, Body %d, cube q_pos_index: %d\n", env_id, bid, qpos_index(env_id, bid));
            printf("Env %d, Body %d, cube q_pos_num: %d\n", env_id, bid, qpos_num(env_id, bid));
        }

        

    }

    template<typename TDataType>
    __global__ void InitQposKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        auto& qpos = rigid_body_system.batch_qpos;
        auto& qpos_index = rigid_body_system.qpos_offset;
        const auto& pos = rigid_body_system.batch_pos;
        const auto& quat = rigid_body_system.batch_quat;
        
        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int is_static = rigid_body_system.is_static(env_id, bid);
            const int qpos_start = qpos_index(env_id, bid);

            if(parent_idx == -1)
            {
                if(is_static)
                    continue;

                for(int i = 0; i < 3; i++)
                    qpos(env_id, qpos_start + i) = pos(env_id, bid)[i];
                qpos(env_id, qpos_start + 3) = quat(env_id, bid).x;
                qpos(env_id, qpos_start + 4) = quat(env_id, bid).y;
                qpos(env_id, qpos_start + 5) = quat(env_id, bid).z;
                qpos(env_id, qpos_start + 6) = quat(env_id, bid).w;
            }
            else
            {

            }
            
            printf("Env %d, Body %d, q_pos: %f %f %f %f %f %f %f\n", env_id, bid, qpos(env_id, qpos_start), qpos(env_id, qpos_start + 1), qpos(env_id, qpos_start + 2), qpos(env_id, qpos_start + 3), qpos(env_id, qpos_start + 4), qpos(env_id, qpos_start + 5), qpos(env_id, qpos_start + 6));
        }
    }

    template<typename TDataType>
    __global__ void BuildRootIndexKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const auto& num_bodies = rigid_body_system.batch_bodies[env_id];
        auto& root_idx = rigid_body_system.root_idx;
        for(int bid = 0; bid < num_bodies; bid++)   // It is necessary to ensure that when an object is loaded, the parent comes before the child.
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            rigid_body_system.subtree_mass(env_id, bid) = rigid_body_system.batch_mass(env_id, bid);
            parent_idx == -1 ? root_idx(env_id, bid) = bid : root_idx(env_id, bid) = root_idx(env_id, parent_idx);
            printf("Env %d, Body %d, Root Index: %d\n", env_id, bid, root_idx(env_id, bid));
        }
    }

    template<typename TDataType>
    __global__ void CalculateSubtreeMassKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;
        
        const int num_bodies = rigid_body_system.batch_bodies[env_id];
        auto& subtree_mass = rigid_body_system.subtree_mass;
        const auto& mass_vec = rigid_body_system.batch_mass;

        for(int bid = num_bodies-1; bid >= 0; bid--)
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            if(parent_idx != -1)
                subtree_mass(env_id, parent_idx) += subtree_mass(env_id, bid);
            printf("Env %d, Body %d, Subtree Mass: %f\n", env_id, bid, subtree_mass(env_id, bid));
        }
    }

    template<typename TDataType>
    __global__ void ForwardKinematicsKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = rigid_body_system.batch_bodies[env_id];
        const auto& quat_world = rigid_body_system.batch_quat;
        auto& rot_world = rigid_body_system.batch_rot;


        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            if(parent_idx == -1)
            {
                rot_world(env_id, bid) = quat_world(env_id, bid).toMatrix3x3();
            }
            else
            {
                ;
            }
        }
    }

    template<typename TDataType>
    __global__ void SubtreeComKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = rigid_body_system.batch_bodies[env_id];

        int bidx = threadIdx.x;
        if (bidx >= num_bodies)
            return;

        auto& subtree_com = rigid_body_system.subtree_com;
        const auto& mass = rigid_body_system.subtree_mass;
        const auto& pos = rigid_body_system.batch_pos;

        subtree_com(env_id, bidx) = mass(env_id, bidx) * pos(env_id, bidx);
        

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

        const int num_envs = env_infos->num_envs;
        rigid_body_system->max_bodies = GetMaxValue(rigid_body_system->batch_bodies, num_envs);
        const int max_bodies = rigid_body_system->max_bodies;
        const int max_nv = max_bodies * 6;

        CArray<int> env_num_bodies(num_envs);

        env_num_bodies.assign(rigid_body_system->batch_bodies);

        rigid_body_system->batch_nv.resize(num_envs);
        rigid_body_system->batch_nv.reset();
        
        rigid_body_system->q_lengths.resize(num_envs, max_bodies);
        rigid_body_system->q_offset.resize(num_envs, max_bodies);

        rigid_body_system->qpos_lengths.resize(num_envs, max_bodies);
        rigid_body_system->qpos_offset.resize(num_envs, max_bodies);

        rigid_body_system->root_idx.resize(num_envs, max_bodies);
        rigid_body_system->subtree_mass.resize(num_envs, max_bodies);
        rigid_body_system->subtree_com.resize(num_envs, max_bodies);


        spdlog::info("[MujocoSolver Solver] Max number of bodies across environments: {}", rigid_body_system->max_bodies);
        // 1.  Calculate the DoFs and establish index
        DofCountAndBuildIndexKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // 2. malloc the solver states based on the DoF count
        rigid_body_system->batch_qacc.resize(num_envs, max_nv);
        rigid_body_system->batch_qvel.resize(num_envs, max_nv);

        rigid_body_system->batch_qM.resize(num_envs, max_nv * max_nv);
        rigid_body_system->batch_cdof.resize(num_envs, max_nv * 6);
        rigid_body_system->batch_cdofdot.resize(num_envs, max_nv * 6);

        rigid_body_system->batch_qpos.resize(num_envs, max_bodies * 7);
        rigid_body_system->dof_frictionloss.resize(num_envs, max_nv);

        rigid_body_system->batch_q_inner_force.resize(num_envs, max_nv);
        rigid_body_system->batch_q_ex_force.resize(num_envs, max_nv);
        rigid_body_system->batch_ex_acc.resize(num_envs, max_nv);

        spdlog::info("[MujocoSolver Solver] Allocated solver state arrays based on DoF counts.");
        // 3. Initialize the qpos
        InitQposKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // 4. Build root index and calculate subtree mass
        BuildRootIndexKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        CalculateSubtreeMassKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        spdlog::info("[MujocoSolver Solver] Initialization complete. Number of environments: {}", env_infos->num_envs);



        spdlog::info("[MujocoSolver Solver] Finished initialization.");
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::Step()
    {
        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;
        
        // 1. Reset forces and accelerations
        rigid_body_system->batch_q_inner_force.reset();
        rigid_body_system->batch_q_ex_force.reset();
        rigid_body_system->batch_ex_acc.reset();

        // Update forward kinematics and subtree com
        ForwardKinematics();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::TimeIntegration()
    {
        spdlog::info("[MujocoSolver Solver] TimeIntegration called.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        TimeIntegrationKernel<<<env_infos->num_envs, 16>>>(
            rigid_body_system->batch_pos, rigid_body_system->batch_rot, rigid_body_system->batch_bodies, env_infos->num_envs);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ForwardKinematics()
    {
        spdlog::info("[MujocoSolver Solver] Start forward kinematics.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        ForwardKinematicsKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        SubtreeComKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        spdlog::info("[MujocoSolver Solver] Finished forward kinematics.");
    }


    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}