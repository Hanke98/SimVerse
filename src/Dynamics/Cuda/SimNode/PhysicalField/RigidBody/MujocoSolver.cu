#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include "../../Utils/utils.h"

namespace dyno
{
    template<typename TDataType>
    __global__ void TimeIntegrationKernel(RigidBody<TDataType> rigid_body_system, DArray<Real> dts, int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        const Real dt = dts[env_id];
        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int& is_static = rigid_body_system.is_static(env_id, bid);
            
            const int& qpos_start = rigid_body_system.qpos_offset(env_id, bid);
            const int& q_start = rigid_body_system.q_offset(env_id, bid);
            auto& qpos = rigid_body_system.batch_qpos;
            const auto& qvel = rigid_body_system.batch_qvel;


            if(parent_idx == -1)
            {
                if(is_static)
                    continue;

                for(int i = 0; i < 3; i++)
                    qpos(env_id, qpos_start + i) += qvel(env_id, q_start + i) * dt;

                Vec3f w = Vec3f(qvel(env_id, q_start + 3), qvel(env_id, q_start + 4), qvel(env_id, q_start + 5));
                Quat<Real> quat = Quat<Real>(qpos(env_id, qpos_start + 3), qpos(env_id, qpos_start + 4), qpos(env_id, qpos_start + 5), qpos(env_id, qpos_start + 6));
                Real angle = w.norm();
                Vec3f axis = w.normalize();
                
                Quat<Real> qrot = QuatFromAxisAngle<Real>(axis, angle);
                
                quat.normalize();
                Quat<Real> quat_new = quat * qrot;
                qpos(env_id, qpos_start + 3) = quat_new.x;
                qpos(env_id, qpos_start + 4) = quat_new.y;
                qpos(env_id, qpos_start + 5) = quat_new.z;
                qpos(env_id, qpos_start + 6) = quat_new.w;
            }
            else
            {
                ;
            }


            auto& pos = rigid_body_system.batch_pos;
            pos(env_id, bid).x = qpos(env_id, qpos_start); 
            pos(env_id, bid).y = qpos(env_id, qpos_start + 1);
            pos(env_id, bid).z = qpos(env_id, qpos_start + 2);
            printf("Env %d, Body %d, Position: (%f, %f, %f)\n", 
                env_id, bid, pos(env_id, bid).x, pos(env_id, bid).y, pos(env_id, bid).z);
        }

        


        
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
        rigid_body_system.batch_nv[env_id] = nv;
        

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
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = rigid_body_system.batch_bodies[env_id];
        auto& subtree_com = rigid_body_system.subtree_com;
        const auto& mass = rigid_body_system.subtree_mass;
        const auto& subtree_mass = rigid_body_system.subtree_mass;
        const auto& pos = rigid_body_system.batch_pos;


        for(int bidx = 0; bidx < num_bodies; bidx++)
            subtree_com(env_id, bidx) = mass(env_id, bidx) * pos(env_id, bidx);

        for(int bidx = num_bodies-1; bidx >= 0; bidx--)
        {
            const int parent_idx = rigid_body_system.parent_idx(env_id, bidx);
            if(parent_idx != -1)
                subtree_com(env_id, parent_idx) += subtree_com(env_id, bidx);
        }

        for(int bidx = 0; bidx < num_bodies; bidx++)
            subtree_com(env_id, bidx) /= subtree_mass(env_id, bidx);

    }

    template<typename TDataType>
    __global__ void ComputeCdofKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        int bid = blockDim.x * blockIdx.y + threadIdx.x;
        if(bid >= env_self_bodies)
            return;

        auto& cdof = rigid_body_system.batch_cdof;
        const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
        const Vec3f& pos = rigid_body_system.batch_pos(env_id, bid);
        const Mat3f rot = rigid_body_system.batch_rot(env_id, bid);
        const Vec3f& subtree_com = rigid_body_system.subtree_com(env_id, bid);

        if(parent_idx != -1)    // joint attached to parent
        {

        }
        else
        {
            if(rigid_body_system.is_static(env_id, bid))
                return;
            const int q_start = rigid_body_system.q_offset(env_id, bid);
            for(int i = 0; i < 3; i++)  // linear velocity
                cdof(env_id, (q_start + i) * 6 + 3 + i) = 1.f;
            
            Vec3f offset = subtree_com - pos;
            // For angular velocity
            for(int i = 0; i < 3; i++)
            {
                Vec3f rot_axis = rot.col(i);
                Vec3f trans_part = cross(rot_axis, offset);
                for(int j = 0; j < 3; j++)
                {
                    cdof(env_id, (q_start + i + 3) * 6 + 3 + j) = trans_part[j];
                    cdof(env_id, (q_start + i + 3) * 6 + j) = rot_axis[j];
                }
            }
        }
    }

    template<typename TDataType>
    __global__ void SubTreeInertialKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {

    }

    template<typename TDataType>
    __global__ void ComputeGeneralizedInertialMatrixKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        int bid = blockDim.x * blockIdx.y + threadIdx.x;
        if(bid >= env_self_bodies)
            return;

        const int is_static = rigid_body_system.is_static(env_id, bid);
        if(is_static)
            return;
        
        auto& batch_qM = rigid_body_system.batch_qM;
        const auto& mass = rigid_body_system.batch_mass(env_id, bid);
        const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
        const int q_start = rigid_body_system.q_offset(env_id, bid);
        const int nv = rigid_body_system.batch_nv[env_id];

        if(parent_idx != -1)
        {
            ;
        }
        else
        {
            for(int i = 0; i < 6; i++)
                for(int j = 0; j < 6; j++)
                    batch_qM(env_id, (q_start + i) * nv + (q_start + j)) = 0.f;
            
            // Trick, cube
            batch_qM(env_id, (q_start + 0) * nv + (q_start + 0)) = mass;
            batch_qM(env_id, (q_start + 1) * nv + (q_start + 1)) = mass;
            batch_qM(env_id, (q_start + 2) * nv + (q_start + 2)) = mass;

            batch_qM(env_id, (q_start + 3) * nv + (q_start + 3)) = mass * (0.8f * 0.8f + 0.8f * 0.8f) / 12.f;
            batch_qM(env_id, (q_start + 4) * nv + (q_start + 4)) = mass * (0.8f * 0.8f + 0.8f * 0.8f) / 12.f;
            batch_qM(env_id, (q_start + 5) * nv + (q_start + 5)) = mass * (0.8f * 0.8f + 0.8f * 0.8f) / 12.f;

        }

        printf("Env %d, Body %d, qM diagonal: %f %f %f %f %f %f\n", env_id, bid,
            batch_qM(env_id, (q_start + 0) * nv + (q_start + 0)),
            batch_qM(env_id, (q_start + 1) * nv + (q_start + 1)),
            batch_qM(env_id, (q_start + 2) * nv + (q_start + 2)),
            batch_qM(env_id, (q_start + 3) * nv + (q_start + 3)),
            batch_qM(env_id, (q_start + 4) * nv + (q_start + 4)),
            batch_qM(env_id, (q_start + 5) * nv + (q_start + 5)));


        
    }

    template<typename TDataType>
    __global__ void TrickAddGravityKernel(RigidBody<TDataType> rigid_body_system, const DArray<Vec3f> gravities, int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;

        if(env_id >= num_envs)
            return;

        auto& q_inner_force = rigid_body_system.batch_q_inner_force;

        q_inner_force(env_id, 0) = gravities[env_id].x;
        q_inner_force(env_id, 1) = gravities[env_id].y;
        q_inner_force(env_id, 2) = gravities[env_id].z;

        printf("Env: %d, q_inner_force: %f, %f, %f\n", env_id, q_inner_force(env_id, 0), q_inner_force(env_id, 1), q_inner_force(env_id, 2));

    }

    template<typename TDataType>    // only for test, only for all bodies without constraints 
    __global__ void TrickMassMatInverse(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        
        int bid = threadIdx.x;
        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        if(bid >= env_self_bodies)
            return;

        auto& batch_qM = rigid_body_system.batch_qM;
        const int q_start = rigid_body_system.q_offset(env_id, bid);
        const int nv = rigid_body_system.batch_nv[env_id];
        for(int i = 0; i < 6; i++)
            for(int j = 0; j < 6; j++)
            {
                int idx = (q_start + i) * nv + (q_start + j);
                if (i == j)
                    batch_qM(env_id, idx) = 1.f / batch_qM(env_id, idx);
                else
                    batch_qM(env_id, idx) = 0.f;

            }
    }

    template<typename TDataType>
    __global__ void UpdateGeneralizedVelKernel(RigidBody<TDataType> rigid_body_system, DArray<Real> timesteps, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        const auto& num_nv = rigid_body_system.batch_nv[env_id];

        int dof_idx = threadIdx.x;
        if(dof_idx >= num_nv)
            return;

        const Real& dt = timesteps[env_id];
        auto& qvel = rigid_body_system.batch_qvel;
        const auto& qacc = rigid_body_system.batch_qacc;

        printf("Env: %d, dof_idx: %d, qacc: %f\n", env_id, dof_idx, qacc(env_id, dof_idx));

        qvel(env_id, dof_idx) += qacc(env_id, dof_idx) * dt;
    }


    __device__ void CubeCollitionWithGround(const Vec3f& pos, const Mat3f& rot, const BoxInfo& box,
        BatchCollisionConstraints& collision_constraints, int env_id, int bid)
    {
        Vec3f ground_normal = Vec3f(0.f, 1.f, 0.f);

        for(int i = 0; i < 8; i++)
        {
            int sx = (i & 1) ? 1 : -1;
            int sy = (i & 2) ? 1 : -1;
            int sz = (i & 4) ? 1 : -1;
            
            Vec3f vertex_origin = Vec3f(sx * box.halfLength.x, sy * box.halfLength.y, sz * box.halfLength.z);
            Vec3f vertex_trans = rot * vertex_origin;
            vertex_trans += pos;

            if(vertex_trans.y > 0.f)
                continue;

            auto& idx = collision_constraints.num_constraints[env_id];
            
            collision_constraints.body_idxs(env_id, idx) = Pair(bid, -1);
            collision_constraints.depth(env_id, idx) = -vertex_trans.y;
            collision_constraints.normal(env_id, idx) = ground_normal;
            collision_constraints.point(env_id, idx) = Vec3f(vertex_trans.x, 0.5f * vertex_trans.y, vertex_trans.z);
            idx += 1;
        }
    }

    template<typename TDataType>
    __global__ void CollisonDetectionKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        auto& collision_constraints = rigid_body_system.collision_constraints;
        const auto& collision_paras = rigid_body_system.collision_paras;
    
        const int num_bodies = rigid_body_system.batch_bodies[env_id];

        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int& is_static = rigid_body_system.is_static(env_id, bid);
            if(is_static)
                continue;

            const Vec3f& pos = rigid_body_system.batch_pos(env_id, bid);
            const Mat3f& rot = rigid_body_system.batch_rot(env_id, bid);

            const int shape_type = rigid_body_system.shape_type(env_id, bid);
            const int shape_idx = rigid_body_system.shape_idx(env_id, bid);
            if(shape_type == 1)   // cube
            {
                CubeCollitionWithGround(pos, rot, rigid_body_system.boxes(env_id, shape_idx), collision_constraints, env_id, bid);
            }
        }
    
    }
}


namespace dyno
{
    template<typename TDataType>
    void MujocoSolver<TDataType>::Init()
    {
        spdlog::info("[MujocoSolver Solver] Starting initialization.");

        auto& collision_paras = this->rigid_body->collision_paras;
        collision_paras.time_const = 0.02f;
        collision_paras.damp_ratio = 1.f;
        collision_paras.dmax = 0.95f;
        collision_paras.dmin = 0.9f;
        collision_paras.midpoint = 0.5f;
        collision_paras.power = 2;


        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        const int num_envs = env_infos->num_envs;
        rigid_body_system->max_bodies = GetMaxValue(rigid_body_system->batch_bodies, num_envs);
        const int max_bodies = rigid_body_system->max_bodies;
        const int max_nv = max_bodies * 6;

        CArray<int> env_num_bodies(num_envs);

        env_num_bodies.assign(rigid_body_system->batch_bodies);


        INIT_DYNO_ARRAY(rigid_body_system->batch_nv, num_envs);

        INIT_DYNO_ARRAY2D(rigid_body_system->q_lengths, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->q_offset, num_envs, max_bodies);

        INIT_DYNO_ARRAY2D(rigid_body_system->qpos_lengths, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->qpos_offset, num_envs, max_bodies);

        INIT_DYNO_ARRAY2D(rigid_body_system->root_idx, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_mass, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_com, num_envs, max_bodies);


        spdlog::info("[MujocoSolver Solver] Max number of bodies across environments: {}", rigid_body_system->max_bodies);
        // 1.  Calculate the DoFs and establish index
        DofCountAndBuildIndexKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // 2. malloc the solver states based on the DoF count
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qacc, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qvel, num_envs, max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qM, num_envs, max_nv * max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdof, num_envs, max_nv * 6);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdofdot, num_envs, max_nv * 6);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qpos, num_envs, max_bodies * 7);
        INIT_DYNO_ARRAY2D(rigid_body_system->dof_frictionloss, num_envs, max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_inner_force, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_ex_force, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_ex_acc, num_envs, max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_crb, num_envs, max_bodies * 10);

        INIT_DYNO_ARRAY(rigid_body_system->collision_constraints.num_constraints, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.body_idxs, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.depth, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.normal, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.point, num_envs, 1024);
        


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

        MakeConstraints();

        // TODO: compute comvel
        // TODO: compute RNE

        TrickAddGravityKernel<TDataType><<<1, 1>>>(*rigid_body_system, env_infos->gravities, env_infos->num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->batch_q_ex_force.assign(rigid_body_system->batch_q_inner_force);

        TrickMassMatInverse<TDataType><<<32, 512>>>(*rigid_body_system, env_infos->num_envs);
        cudaDeviceSynchronize();

        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_q_ex_force,
            rigid_body_system->batch_ex_acc, rigid_body_system->batch_nv, rigid_body_system->batch_nv, env_infos->num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->batch_qacc.assign(rigid_body_system->batch_ex_acc);

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::TimeIntegration()
    {
        spdlog::info("[MujocoSolver Solver] TimeIntegration called.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        UpdateGeneralizedVelKernel<TDataType><<<32, 512>>>(*rigid_body_system, env_infos->timesteps, env_infos->num_envs);
        cudaDeviceSynchronize();

        TimeIntegrationKernel<<<32, 512>>>(*rigid_body_system, env_infos->timesteps, env_infos->num_envs);
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

        ComputeCdofKernel<TDataType><<<dim3(num_envs, 32), 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Crb 
        rigid_body_system->batch_crb.reset();
        // 1. Calculate the global inertia matrix of each rigid body when the center of mass of the corresponding kinematic tree is taken as the reference point.
        // TODO:  
        // 2. Calculate the global inertia matrix of each sub-tree.
        // TODO:


        // 3. Construct the system inertia matrix in the generalized coordinate system.
        rigid_body_system->batch_qM.reset();
        ComputeGeneralizedInertialMatrixKernel<TDataType><<<dim3(num_envs, 32), 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();


        spdlog::info("[MujocoSolver Solver] Finished forward kinematics.");
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::MakeConstraints()
    {
        spdlog::info("[MujocoSolver Solver] MakeConstraints called.");
        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
        // 1. collision constraints
        auto& collision_constraints = rigid_body_system->collision_constraints;
        collision_constraints.num_constraints.reset();
        const auto& collision_paras = rigid_body_system->collision_paras;

        CollisonDetectionKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
    }

    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}