#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include "../../Utils/utils.h"
#include "Algorithm.h"
#include <Eigen/Dense>

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
                Real w_norm = w.norm();
                Quat<Real> qrot;

                if (w_norm > 1e-8)
                {
                    Vec3f axis = w / w_norm;
                    Real angle = w_norm * dt;
                    qrot = QuatFromAxisAngle<Real>(axis, angle);
                }
                else
                    qrot = Quat<Real>(0, 0, 0, 1);
                
                quat.normalize();
                Quat<Real> quat_new = quat * qrot;
                qpos(env_id, qpos_start + 3) = quat_new.x;
                qpos(env_id, qpos_start + 4) = quat_new.y;
                qpos(env_id, qpos_start + 5) = quat_new.z;
                qpos(env_id, qpos_start + 6) = quat_new.w;
            }
            else    // TODO: handle joint
            {
                ;
            }


            auto& pos = rigid_body_system.batch_pos;
            auto& quat = rigid_body_system.batch_quat;
            auto& rot_mat = rigid_body_system.batch_rot;
            pos(env_id, bid).x = qpos(env_id, qpos_start); 
            pos(env_id, bid).y = qpos(env_id, qpos_start + 1);
            pos(env_id, bid).z = qpos(env_id, qpos_start + 2);
            quat(env_id, bid) = Quat<Real>(qpos(env_id, qpos_start + 3), qpos(env_id, qpos_start + 4), qpos(env_id, qpos_start + 5), qpos(env_id, qpos_start + 6));
            rot_mat(env_id, bid) = quat(env_id, bid).toMatrix3x3();

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
            auto& is_isolated = rigid_body_system.is_isolated;

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
                is_isolated(env_id, bid) = 1;
            }
            else
            {
                is_isolated(env_id, parent_idx) = 0;    // parent is not isolated if it has children
                is_isolated(env_id, bid) = 0;           // non-root body is not isolated

                const int& joint_type = rigid_body_system.joint_type(env_id, bid);
                if(joint_type < 3)  // hinge or slide
                {
                    q_index(env_id, bid) = nv;
                    q_num(env_id, bid) = 1;
                    qpos_index(env_id, bid) = nqpos;
                    qpos_num(env_id, bid) = 1;
                    nv += 1;
                    nqpos += 1;
                }
                else
                {
                    q_index(env_id, bid) = nv;
                    q_num(env_id, bid) = 3;
                    qpos_index(env_id, bid) = nqpos;
                    qpos_num(env_id, bid) = 4;
                    nv += 3;
                    nqpos += 4;
                }
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
                const auto& joint_qpos = rigid_body_system.joint_qpos;
                const auto& joint_qpos_start = rigid_body_system.joint_qpos_offset(env_id, bid);
                if(rigid_body_system.joint_type(env_id, bid) < 3)
                    qpos(env_id, qpos_start) = joint_qpos(env_id, joint_qpos_start);
                else
                    for(int i = 0; i < 4; i++)
                        qpos(env_id, qpos_start + i) = joint_qpos(env_id, joint_qpos_start + i);
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
        {
            subtree_com(env_id, bidx) /= subtree_mass(env_id, bidx);
            printf("Env %d, Body %d, Subtree COM: (%f, %f, %f)\n", env_id, bidx, subtree_com(env_id, bidx).x, subtree_com(env_id, bidx).y, subtree_com(env_id, bidx).z);
        }
            

    }

    template<typename TDataType>
    __global__ void ComputeCdofKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        int bid = threadIdx.x;
        if(bid >= env_self_bodies)
            return;

        auto& cdof = rigid_body_system.batch_cdof;
        const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
        const Vec3f& pos = rigid_body_system.batch_pos(env_id, bid);
        const Mat3f rot = rigid_body_system.batch_rot(env_id, bid);
        const Vec3f& subtree_com = rigid_body_system.subtree_com(env_id, bid);

        const int q_start = rigid_body_system.q_offset(env_id, bid);
        int cdof_start = q_start * 6;   // each body has 6 cdof (3 for linear, 3 for angular)

        if(parent_idx != -1)    // joint attached to parent
        {

        }
        else
        {
            if(rigid_body_system.is_static(env_id, bid))
                return;
            
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
            printf("Env %d, Body %d, cdof:\n", env_id, bid);
            for(int i = 0; i < 6; i++)
                printf("  cdof[%d]: %f %f, %f %f, %f %f\n", i, cdof(env_id, (q_start + i) * 6 + 0),
                    cdof(env_id, (q_start + i) * 6 + 1), cdof(env_id, (q_start + i) * 6 + 2),
                    cdof(env_id, (q_start + i) * 6 + 3), cdof(env_id, (q_start + i) * 6 + 4), 
                    cdof(env_id, (q_start + i) * 6 + 5));
            
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

        Real height = rigid_body_system.boxes(env_id, bid).halfLength.y * 2.f;
        Real width = rigid_body_system.boxes(env_id, bid).halfLength.z * 2.f;
        Real depth = rigid_body_system.boxes(env_id, bid).halfLength.x * 2.f;

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

            batch_qM(env_id, (q_start + 3) * nv + (q_start + 3)) = mass * (height * height + width * width) / 12.f;
            batch_qM(env_id, (q_start + 4) * nv + (q_start + 4)) = mass * (width * width + depth * depth) / 12.f;
            batch_qM(env_id, (q_start + 5) * nv + (q_start + 5)) = mass * (depth * depth + height * height) / 12.f;

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

        const int num_bodies = rigid_body_system.batch_bodies[env_id];
        const Vec3f g = gravities[env_id];

        for(int bid = 0; bid < num_bodies; bid++)
        {
            if(rigid_body_system.is_static(env_id, bid))
                continue;

            if(rigid_body_system.parent_idx(env_id, bid) != -1)
                continue;

            const int q_start = rigid_body_system.q_offset(env_id, bid);
            const Real mass = rigid_body_system.batch_mass(env_id, bid);

            // Follow the test convention in pseudocode: q_inner_force is set to -m*g.
            q_inner_force(env_id, q_start + 0) = -mass * g.x;
            q_inner_force(env_id, q_start + 1) = -mass * g.y;
            q_inner_force(env_id, q_start + 2) = -mass * g.z;
            
        }
    }

    // template<typename TDataType>    // only for test, only for all bodies without constraints 
    // __global__ void TrickMassMatInverse(RigidBody<TDataType> rigid_body_system, int num_envs)
    // {
    //     int env_id = blockIdx.x;
    //     if(env_id >= num_envs)
    //         return;
        
    //     int bid = threadIdx.x;
    //     int env_self_bodies = rigid_body_system.batch_bodies[env_id];
    //     if(bid >= env_self_bodies)
    //         return;

    //     auto& batch_qM = rigid_body_system.batch_qM;
    //     auto& batch_qM_inv = rigid_body_system.batch_qM_inv;
    //     const int q_start = rigid_body_system.q_offset(env_id, bid);
    //     const int nv = rigid_body_system.batch_nv[env_id];
    //     for(int i = 0; i < 6; i++)
    //         for(int j = 0; j < 6; j++)
    //         {
    //             int idx = (q_start + i) * nv + (q_start + j);
    //             if (i == j)
    //                 batch_qM_inv(env_id, idx) = 1.f / batch_qM(env_id, idx);
    //             else
    //                 batch_qM_inv(env_id, idx) = 0.f;

    //         }
    // }

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

            auto& idx = collision_constraints.collision_nums[env_id];
            
            collision_constraints.body_idxs(env_id, idx) = Pair(bid, -1);
            collision_constraints.depth(env_id, idx) = -vertex_trans.y;
            collision_constraints.normal(env_id, idx) = ground_normal;
            collision_constraints.point(env_id, idx) = Vec3f(vertex_trans.x, 0.5f * vertex_trans.y, vertex_trans.z);
            collision_constraints.mu(env_id, idx) = 0.6f;
            idx += 1;
        }
    }

    template<typename TDataType>
    __global__ void CountConstraintNums(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        auto& num_each_constraint = rigid_body_system.num_each_constraint;
        auto& constraints = rigid_body_system.collision_constraints;

        const auto& collisions = rigid_body_system.collision_constraints;

        num_each_constraint[env_id] = Vec4i(0, 0, 0, collisions.collision_nums[env_id] * 4);
        
        auto& offsets = rigid_body_system.constraint_offset[env_id];
        auto& num_constraints = rigid_body_system.num_constraints[env_id];

        offsets[0] = 0;
        for(int i = 1; i < 4; i++)
            offsets[i] = offsets[i - 1] + num_each_constraint[env_id][i - 1];

        num_constraints = offsets[3] + num_each_constraint[env_id][3];


        printf("Env: %d, Num constraints(total: %d): %d, %d, %d, %d\n", env_id, num_constraints, num_each_constraint[env_id].x, num_each_constraint[env_id].y, num_each_constraint[env_id].z, num_each_constraint[env_id].w);
        printf("Env: %d, Constraint offsets: %d, %d, %d, %d\n", env_id, offsets[0], offsets[1], offsets[2], offsets[3]);
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
            if(shape_type == 0)   // cube
            {
                CubeCollitionWithGround(pos, rot, rigid_body_system.boxes(env_id, shape_idx), collision_constraints, env_id, bid);
            }

            if(collision_constraints.collision_nums[env_id] > 0)
            {
                printf("Env %d, Body %d, Collision Num: %d\n", env_id, bid, collision_constraints.collision_nums[env_id]);
                for(int cidx = 0; cidx < collision_constraints.collision_nums[env_id]; cidx++)
                {
                    printf("  Collision %d: depth = %f, normal = (%f, %f, %f), point = (%f, %f, %f)\n", cidx,
                        collision_constraints.depth(env_id, cidx),
                        collision_constraints.normal(env_id, cidx).x,
                        collision_constraints.normal(env_id, cidx).y,
                        collision_constraints.normal(env_id, cidx).z,
                        collision_constraints.point(env_id, cidx).x,
                        collision_constraints.point(env_id, cidx).y,
                        collision_constraints.point(env_id, cidx).z);
                }
            }
        }
    
    }

    template<typename TDataType>
    __device__ void ComputeJac(DArray2D<Real>& dst_jac, const Vec3f& c_point, const RigidBody<TDataType>& rigid_body_system, int env_id, int bid, int cidx)
    {
        const int root_idx = rigid_body_system.root_idx(env_id, bid);
        Vec3f offset = c_point - rigid_body_system.subtree_com(env_id, root_idx);

        const auto& q_offset = rigid_body_system.q_offset(env_id, bid);
        const auto& cdof = rigid_body_system.batch_cdof;
        const int num_nv = rigid_body_system.batch_nv[env_id];
        const int jac_offset = cidx * 6;
        const int nv = rigid_body_system.batch_nv[env_id];
        if(rigid_body_system.is_static(env_id, bid))
            return;

        const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
        if(parent_idx == -1)
        {
            for(int i = 0; i < 6; i++)
            {
                Vec3f cdof_angular = Vec3f(cdof(env_id, (q_offset + i) * 6 + 0), cdof(env_id, (q_offset + i) * 6 + 1), cdof(env_id, (q_offset + i) * 6 + 2));
                Vec3f d = cross(cdof_angular, offset);
                MatrixAt(dst_jac, env_id, cidx, 0, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 0);
                MatrixAt(dst_jac, env_id, cidx, 1, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 1);
                MatrixAt(dst_jac, env_id, cidx, 2, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 2);
                
                MatrixAt(dst_jac, env_id, cidx, 3, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 3) + d.x;
                MatrixAt(dst_jac, env_id, cidx, 4, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 4) + d.y;
                MatrixAt(dst_jac, env_id, cidx, 5, q_offset + i, Vec2i(6, nv)) = cdof(env_id, (q_offset + i) * 6 + 5) + d.z;
                
            }
        }
        else
        {
            ;
        }
    }

    template<typename TDataType>
    __global__ void ContactConstraintJacobianKernel(RigidBody<TDataType> rigid_body_system, int num_envs, DArray2D<Real> Jac_temp1, DArray2D<Real> Jac_temp2)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        auto& collisions = rigid_body_system.collision_constraints;
        const int num_collisions = collisions.collision_nums[env_id];
        const int num_nv = rigid_body_system.batch_nv[env_id];

        int cidx = threadIdx.x;
        if(cidx >= num_collisions)
            return;

        const Vec3f& normal = collisions.normal(env_id, cidx);
        const Vec3f& c_point = collisions.point(env_id, cidx);
        int a_idx = collisions.body_idxs(env_id, cidx).first;
        int b_idx = collisions.body_idxs(env_id, cidx).second;
        auto& J = rigid_body_system.batch_J;

        Vec3f t = abs(normal.y) < 0.5 ? Vec3f(0.f, 1.f, 0.f) : Vec3f(0.f, 0.f, 1.f);
        Vec3f y = t - dot(t, normal) * normal;
        y.normalize();
        Vec3f z = cross(normal, y);

        Mat3f c_basis;
        c_basis.setCol(0, normal);
        c_basis.setCol(1, y);
        c_basis.setCol(2, z);


        ComputeJac(Jac_temp1, c_point, rigid_body_system, env_id, a_idx, cidx);
        if(b_idx != -1)
            ComputeJac(Jac_temp2, c_point, rigid_body_system, env_id, b_idx, cidx);
        else
        {
            for(int i = 0; i < 6; i++)
                for(int j = 0; j < num_nv; j++)
                    MatrixAt(Jac_temp2, env_id, cidx, i, j, Vec2i(6, num_nv)) = 0;
        }

        int test_idx = 0;
        if(cidx == test_idx)
        {
            printf("collision point: (%f, %f, %f)\n", c_point.x, c_point.y, c_point.z);
            for(int i = 0; i < 6; i++)
            {
                printf("JacA: ");
                for(int j = 0; j < num_nv; j++)
                    printf("%f\t", Jac_temp1(env_id, (cidx * 6 + i) * num_nv + j));
                printf("\n");
            }
        }
       

        // 用jacA的buffer来放jacp, 用jacB的buffer来放jacdif

        for(int j = 0; j < 3; j++)
            for(int k = 0; k < num_nv; k++)
                MatrixAt(Jac_temp1, env_id, cidx, j, k, Vec2i(3, num_nv)) = MatrixAt(Jac_temp1, env_id, cidx, j + 3, k, Vec2i(6, num_nv)) - MatrixAt(Jac_temp2, env_id, cidx, j + 3, k, Vec2i(6, num_nv));

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < num_nv; j++)
            {
                Real sum = 0;
                for(int k = 0; k < 3; k++)
                    sum += c_basis(k, i) * MatrixAt(Jac_temp1, env_id, cidx, k, j, Vec2i(3, num_nv));
                MatrixAt(Jac_temp2, env_id, cidx, i, j, Vec2i(3, num_nv)) = sum;
            }

        if(cidx == test_idx)
        {
            for(int i = 0; i < 3; i++)
            {
                printf("JacDif: ");
                for(int j = 0; j < num_nv; j++)
                    printf("%f\t", Jac_temp2(env_id, (cidx * 3 + i) * num_nv + j));
                printf("\n");
            }
        }

        const Real mu = collisions.mu(env_id, cidx);
        const int row0 = 4 * cidx;
        for(int i = 0; i < num_nv; i++)
        {
            const Real jn = MatrixAt(Jac_temp2, env_id, cidx, 0, i, Vec2i(3, num_nv));
            const Real jt1 = MatrixAt(Jac_temp2, env_id, cidx, 1, i, Vec2i(3, num_nv));
            const Real jt2 = MatrixAt(Jac_temp2, env_id, cidx, 2, i, Vec2i(3, num_nv));

            MatrixAt(J, env_id, row0 + 0, 0, i, Vec2i(1, num_nv)) = jn + mu * jt1;
            MatrixAt(J, env_id, row0 + 1, 0, i, Vec2i(1, num_nv)) = jn - mu * jt1;
            MatrixAt(J, env_id, row0 + 2, 0, i, Vec2i(1, num_nv)) = jn + mu * jt2;
            MatrixAt(J, env_id, row0 + 3, 0, i, Vec2i(1, num_nv)) = jn - mu * jt2;
        }
        
    }

    template<typename TDataType>
    __global__ void PrintJacobian(RigidBody<TDataType> rigid_body_system, int env_id)
    {
        if(threadIdx.x != 0)
            return;
        const auto& J = rigid_body_system.batch_J;
        const int num_constraints = rigid_body_system.num_constraints[env_id];
        const int num_nv = rigid_body_system.batch_nv[env_id];

        for(int i = 0; i < num_constraints; i++)
        {
            for(int j = 0; j < num_nv; j++)
                printf("%f\t", J(env_id, i * num_nv + j));
            printf("\n");
        }
    }

    __device__ Vec4f ComputeKBIP(Real error, const CollisionConstraintParas& collision_paras)
    {
        const Real& dmax = collision_paras.dmax;
        const Real& dmin = collision_paras.dmin;
        const Real& time_const = collision_paras.time_const;
        const Real& damp_ratio = collision_paras.damp_ratio;
        const Real& midpoint = collision_paras.midpoint;
        const Real& width = collision_paras.width;
        const Real& power = collision_paras.power;

        Real K = 1.f / (dmax * dmax * time_const * time_const * damp_ratio * damp_ratio);
        Real B = 2.f / (dmax * time_const);

        Real x = error / width;
        Real sign = 1.f;
        x < 0.f ? sign = -1.f : sign = 1.f;
        x *= sign;

        if(x > 1.f)
            return Vec4f(K, B, dmax, 0.f);
        else if(x < 0.f)
            return Vec4f(K, B, dmin, 0.f);

        Real y, yP;
        if(x < midpoint)
        {
            Real tmp = 1.f / powf(midpoint, power - 1.f);
            y = tmp * powf(x, power);
            yP = tmp * power * powf(x, power - 1.f);
        }
        else
        {
            Real tmp = 1.f / powf(1.f - midpoint, power - 1.f);
            y = 1.f - tmp * powf(1.f - x, power);
            yP = tmp * power * powf(1.f - x, power - 1.f);
        }

        Real I = dmin + y * (dmax - dmin);
        Real P = sign * yP * (dmax - dmin) / width;

        return Vec4f(K, B, I, P);
    }

    template<typename TDataType>
    __global__ void ComputeContactAref(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_envs)
            return;


        const int contact_idx = threadIdx.x;
        if(contact_idx >= rigid_body_system.collision_constraints.collision_nums[env_id])
            return;

        const int constraint_start = rigid_body_system.constraint_offset[env_id][3];
        const int num_constraint = rigid_body_system.num_each_constraint[env_id][3];


        const auto& collision_constraints = rigid_body_system.collision_constraints;
        const auto& depth = collision_constraints.depth(env_id, contact_idx);
        const auto& constraint_vels = rigid_body_system.batch_constraint_vel;

        Vec4f KBIP = ComputeKBIP(depth, rigid_body_system.collision_paras);
        
        auto& imp = rigid_body_system.batch_imp;
        auto& aref = rigid_body_system.batch_aref;

        for(int i = 0; i < 4; i++)
        {
            int idx = constraint_start + contact_idx * 4 + i;
            Real K = KBIP[0];
            Real B = KBIP[1];
            Real I = KBIP[2];
            

            imp(env_id, idx) = I;
            aref(env_id, idx) = -B * constraint_vels(env_id, idx) + K * I * depth;
        }
    }

    __device__ void RotateJacobianRow(DArray2D<Real>& jac_src, DArray2D<Real>& jac_dst, int mat_idx, int num_cols, int sys_id)
    {
        for(int i = 0; i < num_cols; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                MatrixAt(jac_dst, sys_id, mat_idx, j, i, Vec2i(6, num_cols)) = MatrixAt(jac_src, sys_id, mat_idx, j + 3, i, Vec2i(6, num_cols));
                MatrixAt(jac_dst, sys_id, mat_idx, j + 3, i, Vec2i(6, num_cols)) = MatrixAt(jac_src, sys_id, mat_idx, j, i, Vec2i(6, num_cols));
            }
        }
    }

    template<typename TDataType>
    __global__ void ComputeDiagJMinvJT(RigidBody<TDataType> rigid_body_system, int num_envs, DArray2D<Real> J_temp, DArray2D<Real> J_temp2)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int bid = threadIdx.x;
        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        if(bid >= env_self_bodies)
            return;

        
        if(rigid_body_system.is_static(env_id, bid))
        {
            rigid_body_system.batch_weight_inv(env_id, bid) = 0.f;
            return;
        }

        const int nv = rigid_body_system.batch_nv[env_id];
        if(nv <= 0)
        {
            rigid_body_system.batch_weight_inv(env_id, bid) = 0.f;
            return;
        }

        for(int r = 0; r < 6; r++)
            for(int c = 0; c < nv; c++)
            {
                MatrixAt(J_temp, env_id, bid, r, c, Vec2i(6, nv)) = 0.f;
                MatrixAt(J_temp2, env_id, bid, r, c, Vec2i(6, nv)) = 0.f;
            }

        const auto& pos = rigid_body_system.batch_pos(env_id, bid);
        ComputeJac(J_temp, pos, rigid_body_system, env_id, bid, bid);
        RotateJacobianRow(J_temp, J_temp2, bid, nv, env_id);    // J_temp2 now is jac

        // batch_qM_inv stores the in-place Cholesky factor L of qM from Step().
        // For each Jacobian row j, solve qM * x = j^T via:
        //   1) L * y = j^T (forward substitution)
        //   2) L^T * x = y (back substitution)
        // Then x^T equals j * qM^{-1}.
        const auto& L = rigid_body_system.batch_qM_inv;
        for(int r = 0; r < 6; r++)
        {
            for(int i = 0; i < nv; i++)
            {
                Real sum = MatrixAt(J_temp2, env_id, bid, r, i, Vec2i(6, nv));
                for(int k = 0; k < i; k++)
                    sum -= MatrixAt(L, env_id, i, k, Vec2i(nv, nv)) * MatrixAt(J_temp, env_id, bid, r, k, Vec2i(6, nv));

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                MatrixAt(J_temp, env_id, bid, r, i, Vec2i(6, nv)) = sum / lii;
            }

            for(int i = nv - 1; i >= 0; i--)
            {
                Real sum = MatrixAt(J_temp, env_id, bid, r, i, Vec2i(6, nv));
                for(int k = i + 1; k < nv; k++)
                    sum -= MatrixAt(L, env_id, k, i, Vec2i(nv, nv)) * MatrixAt(J_temp, env_id, bid, r, k, Vec2i(6, nv));

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                MatrixAt(J_temp, env_id, bid, r, i, Vec2i(6, nv)) = sum / lii;
            }
        }

        Real a00 = 0.f;
        Real a11 = 0.f;
        Real a22 = 0.f;
        for(int k = 0; k < nv; k++)
        {
            a00 += MatrixAt(J_temp2, env_id, bid, 0, k, Vec2i(6, nv)) * MatrixAt(J_temp, env_id, bid, 0, k, Vec2i(6, nv));
            a11 += MatrixAt(J_temp2, env_id, bid, 1, k, Vec2i(6, nv)) * MatrixAt(J_temp, env_id, bid, 1, k, Vec2i(6, nv));
            a22 += MatrixAt(J_temp2, env_id, bid, 2, k, Vec2i(6, nv)) * MatrixAt(J_temp, env_id, bid, 2, k, Vec2i(6, nv));
        }

        rigid_body_system.batch_weight_inv(env_id, bid) = (a00 + a11 + a22) / 3.f;
        
        
    }

    template<typename TDataType>
    __global__ void ComputeContact_dAKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        const int contact_idx = threadIdx.x;
        const int num_contacts = rigid_body_system.collision_constraints.collision_nums[env_id];
        if(contact_idx >= num_contacts)
            return;

        const auto& collision_constraints = rigid_body_system.collision_constraints;
        const auto& weight_inv = rigid_body_system.batch_weight_inv;
        int body_A_idx = collision_constraints.body_idxs(env_id, contact_idx).first;
        int body_B_idx = collision_constraints.body_idxs(env_id, contact_idx).second;
        Real mu = collision_constraints.mu(env_id, contact_idx);

        Real w = body_B_idx != -1 ? weight_inv(env_id, body_A_idx) + weight_inv(env_id, body_B_idx) : weight_inv(env_id, body_A_idx);
        Real tmp = w * (1 + mu * mu);

        for(int i = 0; i < 4; i++)
            rigid_body_system.batch_dA(env_id, contact_idx * 4 + i) = tmp;
    }

    template<typename TDataType>
    __global__ void ComputeRKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int cidx = threadIdx.x;
        const int num_constraints = rigid_body_system.num_constraints[env_id];
        if(cidx >= num_constraints)
            return;

        const int collition_offset = rigid_body_system.constraint_offset[env_id][3];
        auto& R = rigid_body_system.batch_D;
        const auto& imp = rigid_body_system.batch_imp;
        const auto& dA = rigid_body_system.batch_dA;

        
        R(env_id, cidx) = (1.f - imp(env_id, cidx)) * dA(env_id, cidx) / imp(env_id, cidx);
        if(cidx >= collition_offset)
        {
            const Real mu = rigid_body_system.collision_constraints.mu(env_id, (cidx - collition_offset) / 4);
            R(env_id, cidx) *= 2.f * mu * mu;
        }
    }

    template<typename TDataType>
    __global__ void ComputeDKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int cidx = threadIdx.x;
        const int num_constraints = rigid_body_system.num_constraints[env_id];
        if(cidx >= num_constraints)
            return;

        // auto& D = ri
        auto& D = rigid_body_system.batch_D;
        D(env_id, cidx) = 1.f / D(env_id, cidx);
    }

    template<typename TDataType>
    __global__ void ContactAndJointLimitEnergyKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int num_constraints = rigid_body_system.num_constraints[env_id];
        const int constraint_start = rigid_body_system.constraint_offset[env_id][2];

        int cidx = threadIdx.x + constraint_start;
        if(cidx >= num_constraints)
            return ;

        auto& constraint_force = rigid_body_system.batch_constraint_force;
        auto& constraint_energy = rigid_body_system.batch_constraint_energy;
        const Real& Jaref_cidx = rigid_body_system.batch_Jaref(env_id, cidx);
        const Real& D_cidx = rigid_body_system.batch_D(env_id, cidx);

        constraint_force(env_id, cidx) = - D_cidx * Jaref_cidx;
        rigid_body_system.batch_unquads(env_id, cidx) = 0;
        if(Jaref_cidx > 0)
        {
            constraint_force(env_id, cidx) = 0.f;
            rigid_body_system.batch_unquads(env_id, cidx) = 1;
        }
        else
            constraint_energy(env_id, cidx) += 0.5f * D_cidx * Jaref_cidx * Jaref_cidx;
    }

    template<typename TDataType>
    __global__ void InertialEnergyKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int num_nv = rigid_body_system.batch_nv[env_id];
        const int dof_idx = threadIdx.x;
        if(dof_idx >= num_nv)
            return;

        const auto& Ma = rigid_body_system.batch_Ma(env_id, dof_idx);
        const auto& q_ex_force = rigid_body_system.batch_q_ex_force(env_id, dof_idx);
        const auto& q_acc = rigid_body_system.batch_qacc(env_id, dof_idx);
        const auto& q_ex_acc = rigid_body_system.batch_q_ex_acc(env_id, dof_idx);
        auto& energy = rigid_body_system.batch_energy[env_id];

        atomicAdd(&energy, 0.5f * (Ma - q_ex_force) * (q_acc - q_ex_acc));
    }

    template<typename TDataType>
    __global__ void ReduceConstraintEnergyKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        __shared__ Real sh_sum[256];

        const int tid = threadIdx.x;
        const int num_constraints = rigid_body_system.num_constraints[env_id];
        const auto& constraint_energy = rigid_body_system.batch_constraint_energy;

        Real local_sum = 0.f;
        for(int cidx = tid; cidx < num_constraints; cidx += blockDim.x)
            local_sum += constraint_energy(env_id, cidx);

        sh_sum[tid] = local_sum;
        __syncthreads();

        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                sh_sum[tid] += sh_sum[tid + stride];
            __syncthreads();
        }

        if(tid == 0)
            rigid_body_system.batch_energy[env_id] = sh_sum[0];
    }

    template<typename TDataType>
    __global__ void BuildHessianKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        const int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int nv = rigid_body_system.batch_nv[env_id];
        const int num_constraints = rigid_body_system.num_constraints[env_id];

        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.z * blockDim.x + threadIdx.x;

        if(row >= nv || col > row)
            return;

        Real sum = 0.f;
        const auto& J = rigid_body_system.batch_J;
        const auto& D = rigid_body_system.batch_D;
        const auto& unquads = rigid_body_system.batch_unquads;

        // H = qM + J^T * diag(D_eff) * J, D_eff[cidx] = 0 for unquad constraints.
        for(int cidx = 0; cidx < num_constraints; cidx++)
        {
            if(unquads(env_id, cidx) != 0)
                continue;

            const Real d = D(env_id, cidx);
            const Real jr = MatrixAt(J, env_id, cidx, row, Vec2i(num_constraints, nv));
            const Real jc = MatrixAt(J, env_id, cidx, col, Vec2i(num_constraints, nv));
            sum += jr * d * jc;
        }

        const Real h = MatrixAt(rigid_body_system.batch_qM, env_id, row, col, Vec2i(nv, nv)) + sum;
        MatrixAt(rigid_body_system.batch_H, env_id, row, col, Vec2i(nv, nv)) = h;
        if(col != row)
            MatrixAt(rigid_body_system.batch_H, env_id, col, row, Vec2i(nv, nv)) = h;
    }

    template<typename TDataType>
    __global__ void UpdateGradientKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        const int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int dof_idx = threadIdx.x;
        const int nv = rigid_body_system.batch_nv[env_id];
        if(dof_idx >= nv)
            return;

        const int num_constraints = rigid_body_system.num_constraints[env_id];
        const auto& J = rigid_body_system.batch_J;
        const auto& constraint_force = rigid_body_system.batch_constraint_force;

        Real jt_f = 0.f;
        for(int cidx = 0; cidx < num_constraints; cidx++)
        {
            const Real j = MatrixAt(J, env_id, cidx, dof_idx, Vec2i(num_constraints, nv));
            jt_f += j * constraint_force(env_id, cidx);
        }

        rigid_body_system.batch_grad(env_id, dof_idx) =
            - rigid_body_system.batch_Ma(env_id, dof_idx)
            + rigid_body_system.batch_q_ex_force(env_id, dof_idx)
            + jt_f;
    }

    template<typename TDataType>
    __global__ void ComputeScale(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;


        int qidx = threadIdx.x;
        int nv = rigid_body_system.batch_nv[env_id];
        if(qidx >= nv)
            return;

        auto& qM_diag = rigid_body_system.batch_qM_diag_elem;
        qM_diag(env_id, qidx) = rigid_body_system.batch_qM(env_id, qidx * nv + qidx);

        __syncthreads();

        if(qidx == 0)
        {
            Real sum_qM_diag = 0.f;
            for(int i = 0; i < nv; i++)
                sum_qM_diag += qM_diag(env_id, i);
            rigid_body_system.batch_scale[env_id] = 1.f / sum_qM_diag;
        }
    }

    template<typename TDataType>
    __global__ void BatchNewtonIterationKernel(RigidBody<TDataType> rigid_body_system, int num_envs, int max_iters)
    {
        int env_idx = blockIdx.x;
        if(env_idx >= num_envs)
            return;

        __shared__ bool converged; 

        int num_nv = rigid_body_system.batch_nv[env_idx];
        int nc = rigid_body_system.num_constraints[env_idx];

        int iter = 0;

        while(iter < max_iters)
        {
            ;
        }


    }

    template<typename TDataType>
    __global__ void SearchAlphaKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        auto& alpha = rigid_body_system.batch_alpha[env_id];
        alpha = 1.f;

        auto& dx = rigid_body_system.batch_dx;


    }
    
    template<typename TDataType>
    __global__ void CheckConvergenceKernel(RigidBody<TDataType> rigid_body_system, int num_envs, Real impr, Real eps)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const auto& scale = rigid_body_system.batch_scale[env_id];
        const auto& grad = rigid_body_system.batch_grad;

        Real grad_square_sum = 0.f;
        for(int i = 0; i < rigid_body_system.batch_nv[env_id]; i++)
            grad_square_sum += grad(env_id, i) * grad(env_id, i);
        grad_square_sum = scale * sqrt(grad_square_sum);

        Real improvment = scale * (rigid_body_system.batch_energy_ref[env_id] - rigid_body_system.batch_energy[env_id]); 

        if(grad_square_sum < eps || improvment < impr)
            rigid_body_system.is_converged[env_id] = 1;
    }
 
}


namespace dyno
{
    template<typename TDataType>
    void MujocoSolver<TDataType>::Init()
    {
        spdlog::info("[MujocoSolver Solver] Starting initialization.");

        const int max_joint_qpos = 128;

        auto& collision_paras = this->rigid_body->collision_paras;
        collision_paras.time_const = 0.02f;
        collision_paras.damp_ratio = 1.f;
        collision_paras.dmax = 0.95f;
        collision_paras.dmin = 0.9f;
        collision_paras.width = 0.001;
        collision_paras.midpoint = 0.5f;
        collision_paras.power = 2;


        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        const int num_envs = env_infos->num_envs;
        const int num_max_constraints = env_infos->max_constraints;
        rigid_body_system->max_bodies = GetMaxValue(rigid_body_system->batch_bodies, num_envs);
        const int max_bodies = rigid_body_system->max_bodies;
        const int max_nv = max_bodies * 6;

        CArray<int> env_num_bodies(num_envs);

        env_num_bodies.assign(rigid_body_system->batch_bodies);


        INIT_DYNO_ARRAY(rigid_body_system->batch_nv, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->is_isolated, num_envs, max_bodies);
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
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_aref, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_imp, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_Jaref, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY(rigid_body_system->batch_energy, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->batch_energy_ref, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_constraint_energy, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_unquads, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_H, num_envs, max_nv * max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_dx, num_envs, max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qM, num_envs, max_nv * max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qM_inv, num_envs, max_nv * max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qM_diag_elem, num_envs, max_nv);
        INIT_DYNO_ARRAY(rigid_body_system->batch_scale, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdof, num_envs, max_nv * 6);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdofdot, num_envs, max_nv * 6);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qpos, num_envs, max_bodies * 7);
        INIT_DYNO_ARRAY2D(rigid_body_system->dof_frictionloss, num_envs, max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_inner_force, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_ex_force, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_ex_acc, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_Ma, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_grad, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_weight_inv, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_dof_weight_inv, num_envs, max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_dA, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_D, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY(rigid_body_system->is_converged, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->sys_alpha, num_envs);
        

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_crb, num_envs, max_bodies * 10);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_J, num_envs, num_max_constraints * max_nv);
        INIT_DYNO_ARRAY(rigid_body_system->num_constraints, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->num_each_constraint, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->constraint_offset, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_constraint_vel, num_envs, num_max_constraints);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_constraint_force, num_envs, num_max_constraints);

        INIT_DYNO_ARRAY(rigid_body_system->collision_constraints.collision_nums, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.body_idxs, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.depth, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.normal, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.point, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.mu, num_envs, 1024);

        INIT_DYNO_ARRAY2D(rigid_body_system->Mat_temp1, num_envs, num_max_constraints * max_nv);
        INIT_DYNO_ARRAY2D(rigid_body_system->Mat_temp2, num_envs, num_max_constraints * max_nv);

        INIT_DYNO_ARRAY2D(rigid_body_system->joint_type, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->joint_qpos, num_envs, max_joint_qpos);
        INIT_DYNO_ARRAY2D(rigid_body_system->joint_qpos_offset, num_envs, max_bodies);

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
        rigid_body_system->batch_q_ex_acc.reset();
        rigid_body_system->is_converged.reset();
        // Update forward kinematics and subtree com
        ForwardKinematics();

        MakeConstraints();

        // TODO: compute comvel
        // TODO: compute RNE


        TrickAddGravityKernel<TDataType><<<32, 512>>>(*rigid_body_system, env_infos->gravities, env_infos->num_envs);
        cudaDeviceSynchronize();

        // q_ex_force = -q_inner_force
        SumArray2D<<<32, 128>>>(rigid_body_system->batch_q_ex_force, rigid_body_system->batch_q_inner_force,
            rigid_body_system->batch_q_ex_force, env_infos->num_envs, rigid_body_system->batch_nv, false);
        cudaDeviceSynchronize();

        // Solve qM * q_ex_acc = q_ex_force by Cholesky factorization instead of explicitly forming qM^{-1}.
        // BatchCholeskySolveVarSizeKernel factorizes in-place, so copy qM to a temporary buffer first.
        rigid_body_system->batch_qM_inv.assign(rigid_body_system->batch_qM);
        
        BatchCholeskySolveVarSizeKernel<<<env_infos->num_envs, 1>>>(
            rigid_body_system->batch_qM_inv,
            rigid_body_system->batch_q_ex_force,
            rigid_body_system->batch_q_ex_acc,
            rigid_body_system->batch_nv,
            rigid_body_system->max_bodies * 6,
            env_infos->num_envs,
            rigid_body_system->is_converged);
        cudaDeviceSynchronize();

        rigid_body_system->batch_qacc.assign(rigid_body_system->batch_q_ex_acc);

        NewtonSolver();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::NewtonSolver()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        MakeJacobian();

        ComputeAref();

        ComputeRD();

        // Iterate 0
        ComputeEnergy();
        BuildHessian();
        UpdateGradient();
        SolveSystem();

        ComputeScale<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        
        // rigid_body_system->sys_alpha.reset();

        // TODO: iterate more times
        int iter = 0;
        while(iter < 10)
        {
            // TODO: line search


            // Update qacc      qacc += α * dx
            SumArray2D<<<32, 128>>>(rigid_body_system->batch_qacc, rigid_body_system->batch_dx,
                rigid_body_system->batch_qacc, num_envs, rigid_body_system->batch_nv, rigid_body_system->is_converged);
            cudaDeviceSynchronize();

            // Update Ma        Ma += α * qM * dx
            BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_dx, rigid_body_system->batch_Ma,
                rigid_body_system->batch_nv, rigid_body_system->batch_nv, num_envs, true, rigid_body_system->is_converged);
            cudaDeviceSynchronize();
            // Update Jaref     Jaref += α * J * dx
            BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_dx, rigid_body_system->batch_Jaref,
                rigid_body_system->num_constraints, rigid_body_system->batch_nv, num_envs, true, rigid_body_system->is_converged);
            cudaDeviceSynchronize();
            spdlog::info("qacc in newton");
            PrintVector<<<1, 1>>>(rigid_body_system->batch_qacc, 0, 6);
            cuSynchronize();
            spdlog::info("Ma in newton");
            PrintVector<<<1, 1>>>(rigid_body_system->batch_Ma, 0, 6);
            cuSynchronize();
            spdlog::info("Jaref in newton");
            PrintVector<<<1, 1>>>(rigid_body_system->batch_Jaref, 0, 4);
            cuSynchronize();

            rigid_body_system->batch_energy_ref.assign(rigid_body_system->batch_energy);
            ComputeEnergy();
            BuildHessian();
            UpdateGradient();
            SolveSystem();
            

            CheckConvergenceKernel<<<32, 128>>>(*rigid_body_system, num_envs, 1e-8f, 1e-8f);
            cudaDeviceSynchronize();
            Reduction<int> reduce_converged;
            int total_converged = reduce_converged.accumulate(rigid_body_system->is_converged.begin(), num_envs);
            spdlog::info("[MujocoSolver Solver] Iteration {}, converged environments: {}/{}", iter, total_converged, num_envs);
            if(total_converged == num_envs)
                break;

            iter++;
        }
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

        spdlog::info("Qvel:");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qvel, 0, 6);
        cudaDeviceSynchronize();
        spdlog::info("QPOS:");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qpos, 0, 7);
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

        ComputeCdofKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
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
        collision_constraints.collision_nums.reset();
        const auto& collision_paras = rigid_body_system->collision_paras;


        CollisonDetectionKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::MakeJacobian()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        // 碰撞检测已完成
        
        CountConstraintNums<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Function2Pt::plus(rigid_body_system->num_constraints, rigid_body_system->num_topo_invariant_constraints, rigid_body_system->collision_constraints.collision_nums);
        
        rigid_body_system->batch_J.reset();

        ContactConstraintJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs, rigid_body_system->Mat_temp1, rigid_body_system->Mat_temp2);
        cudaDeviceSynchronize();

        spdlog::info("Jacobian: ");
        PrintJacobian<TDataType><<<1, 1>>>(*rigid_body_system, 0);
        cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeAref()
    {
        // constraint_vel = J * qvel
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_qvel, rigid_body_system->batch_constraint_vel,
            rigid_body_system->num_constraints, rigid_body_system->batch_nv, num_envs);
        cudaDeviceSynchronize();

        // TODO: 等式约束
        // TODO: 摩擦损失约束
        // TODO: 关节限位约束

        ComputeContactAref<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
        printf("Aref:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_aref, 0, 16);
        cudaDeviceSynchronize();


        // Compute constraint residuals Jaref
        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_qacc,
            rigid_body_system->batch_Ma, rigid_body_system->batch_nv, rigid_body_system->batch_nv, num_envs);
        cudaDeviceSynchronize();

        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_qacc,
            rigid_body_system->batch_Jaref, rigid_body_system->num_constraints, rigid_body_system->batch_nv, num_envs);
        cudaDeviceSynchronize();

        SumArray2D<<<32, 128>>>(rigid_body_system->batch_Jaref, rigid_body_system->batch_aref, rigid_body_system->batch_Jaref,
            num_envs, rigid_body_system->num_constraints, false);
        cudaDeviceSynchronize();

        printf("Jaref:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_Jaref, 0, 16);
        cudaDeviceSynchronize();


    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeRD()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
    
        ComputeDiagJMinvJT<TDataType><<<32, 512>>>(*rigid_body_system, num_envs, rigid_body_system->Mat_temp1, rigid_body_system->Mat_temp2);
        cudaDeviceSynchronize();

        // TODO: compute diag(JM^(-1)J^T) for joints

        rigid_body_system->batch_dA.reset();

        // TODO: handling equality constraints
        // TODO: handling friction loss constraints
        // TODO: handling joint limit constraints
        ComputeContact_dAKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        ComputeRKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        printf("R:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_D, 0, 16);
        cudaDeviceSynchronize();

        ComputeDKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        printf("D:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_D, 0, 16);
        cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeEnergy()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
        
        rigid_body_system->batch_energy.reset();
        rigid_body_system->batch_constraint_energy.reset();
        // 1. equality constraint energy
        // 2. friction loss constraint energy

        // 3. contact constraint energy and joint limit constraint energy
        ContactAndJointLimitEnergyKernel<TDataType><<<num_envs, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        ReduceConstraintEnergyKernel<TDataType><<<num_envs, 256>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
        spdlog::info("Constraint force:");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_constraint_force, 0, 16);
        cudaDeviceSynchronize();

        spdlog::info("s: ");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_energy, 1);
        cudaDeviceSynchronize();

        InertialEnergyKernel<TDataType><<<num_envs, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // PrintVector<<<1, 1>>>(rigid_body_system->batch_energy, 1);
        // cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::BuildHessian()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_H.reset();

        const int max_nv = rigid_body_system->max_bodies * 6;
        dim3 block(16, 16, 1);
        dim3 grid(num_envs,
            (max_nv + block.y - 1) / block.y,
            (max_nv + block.x - 1) / block.x);

        BuildHessianKernel<TDataType><<<grid, block>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
        printf("Hessian:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_H, 0, 6 * 6);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::UpdateGradient()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_grad.reset();

        UpdateGradientKernel<TDataType><<<num_envs, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        printf("Gradient:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_grad, 0, 6);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::SolveSystem()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
        rigid_body_system->batch_dx.reset();

        auto& H = rigid_body_system->batch_H;
        auto& grad = rigid_body_system->batch_grad;
        auto& x = rigid_body_system->batch_dx; // reuse qacc as solution

        
        BatchCholeskySolveVarSizeKernel<<<num_envs, 1>>>(H, grad, x, rigid_body_system->batch_nv, rigid_body_system->max_bodies * 6, num_envs, rigid_body_system->is_converged);
        cudaDeviceSynchronize();

        printf("dx (solution):\n");
        PrintVector<<<1, 1>>>(x, 0, 6);
        cudaDeviceSynchronize();
    }

    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}