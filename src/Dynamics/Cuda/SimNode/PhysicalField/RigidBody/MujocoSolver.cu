#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include "../../Utils/utils.h"
#include "Algorithm.h"

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
            // printf("Env %d, Body %d, cdof:\n", env_id, bid);
            // for(int i = 0; i < 6; i++)
            //     printf("  cdof[%d]: %f %f, %f %f, %f %f\n", i, cdof(env_id, (q_start + i) * 6 + 0),
            //         cdof(env_id, (q_start + i) * 6 + 1), cdof(env_id, (q_start + i) * 6 + 2),
            //         cdof(env_id, (q_start + i) * 6 + 3), cdof(env_id, (q_start + i) * 6 + 4), 
            //         cdof(env_id, (q_start + i) * 6 + 5));
            
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

            auto& idx = collision_constraints.collision_nums[env_id];
            
            collision_constraints.body_idxs(env_id, idx) = Pair(bid, -1);
            collision_constraints.depth(env_id, idx) = -vertex_trans.y;
            collision_constraints.normal(env_id, idx) = ground_normal;
            collision_constraints.point(env_id, idx) = Vec3f(vertex_trans.x, 0.5f * vertex_trans.y, vertex_trans.z);
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

        int test_idx = 3;
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
        const int num_max_constraints = env_infos->max_constraints;
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

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_J, num_envs, num_max_constraints * max_nv);
        INIT_DYNO_ARRAY(rigid_body_system->num_constraints, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->num_each_constraint, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->constraint_offset, num_envs);

        INIT_DYNO_ARRAY(rigid_body_system->collision_constraints.collision_nums, num_envs);
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

        MakeJacobian();

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
        
        DArray2D<Real> Mat_temp1(num_envs, rigid_body_system->batch_J.size() / num_envs);   // num_constraints * nv
        DArray2D<Real> Mat_temp2(num_envs, rigid_body_system->batch_J.size() / num_envs);

        ContactConstraintJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs, Mat_temp1, Mat_temp2);
        cudaDeviceSynchronize();
    }

    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}