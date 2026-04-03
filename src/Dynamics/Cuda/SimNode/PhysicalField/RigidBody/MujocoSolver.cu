#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h> 

#include "../../Utils/utils.h"
#include "Algorithm.h"
#include <Eigen/Dense>
// #include "kernel.cuh"

#define NV_TMP 256

namespace dyno
{
    __global__ void TimeIntegrationKernel(
        DArray<int> batch_bodies,
        DevArr2D<int> parent_idx,
        DevArr2D<int> is_static,
        DevArr2D<int> qpos_offset,
        DevArr2D<int> q_offset,
        DevArr2D<Real> batch_qpos,
        DevArr2D<Real> batch_qvel,
        DevArr2D<int> joint_type,
        DArray<Real> dts,
        int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = batch_bodies[env_id];
        const Real dt = dts[env_id];
        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int parent = parent_idx(env_id, bid);
            const int is_static_body = is_static(env_id, bid);

            const int qpos_start = qpos_offset(env_id, bid);
            const int q_start = q_offset(env_id, bid);


            if(parent == -1)
            {
                if(is_static_body)
                    continue;

                for(int i = 0; i < 3; i++)
                    batch_qpos(env_id, qpos_start + i) += batch_qvel(env_id, q_start + i) * dt;

                Vec3f w = Vec3f(batch_qvel(env_id, q_start + 3), batch_qvel(env_id, q_start + 4), batch_qvel(env_id, q_start + 5));
                Quat<Real> quat = Quat<Real>(batch_qpos(env_id, qpos_start + 3), batch_qpos(env_id, qpos_start + 4), batch_qpos(env_id, qpos_start + 5), batch_qpos(env_id, qpos_start + 6));
                Real w_norm = w.norm();
                Quat<Real> qrot;

                if (w_norm > 1e-8)
                {
                    Vec3f axis = w / w_norm;
                    Real angle = w_norm * dt;
                    qrot = QuatFromAxisAngle(axis, angle);
                }
                else
                    qrot = Quat<Real>(0, 0, 0, 1);

                quat.normalize();
                Quat<Real> quat_new = quat * qrot;
                batch_qpos(env_id, qpos_start + 3) = quat_new.x;
                batch_qpos(env_id, qpos_start + 4) = quat_new.y;
                batch_qpos(env_id, qpos_start + 5) = quat_new.z;
                batch_qpos(env_id, qpos_start + 6) = quat_new.w;
            }
            else
            {
                const int joint_type_body = joint_type(env_id, bid);
                if(joint_type_body < 3)  // Hinge or Slide
                    batch_qpos(env_id, qpos_start) += batch_qvel(env_id, q_start) * dt;
                else
                {
                    Vec3f w = Vec3f(batch_qvel(env_id, q_start), batch_qvel(env_id, q_start + 1), batch_qvel(env_id, q_start + 2));
                    Quat<Real> quat = Quat<Real>(batch_qpos(env_id, qpos_start), batch_qpos(env_id, qpos_start + 1), batch_qpos(env_id, qpos_start + 2), batch_qpos(env_id, qpos_start + 3));
                    Real w_norm = w.norm();
                    Quat<Real> qrot = Quat<Real>(0, 0, 0, 1);

                    if (w_norm > 1e-8)
                    {
                        Vec3f axis = w / w_norm;
                        Real angle = w_norm * dt;
                        qrot = QuatFromAxisAngle(axis, angle);
                    }

                    quat.normalize();
                    Quat<Real> quat_new = quat * qrot;
                    batch_qpos(env_id, qpos_start) = quat_new.x;
                    batch_qpos(env_id, qpos_start + 1) = quat_new.y;
                    batch_qpos(env_id, qpos_start + 2) = quat_new.z;
                    batch_qpos(env_id, qpos_start + 3) = quat_new.w;
                }

            }
        }
    }

    __global__ void DofCountAndBuildIndexKernel(
        DArray<int> batch_bodies,
        DevArr2D<int> q_index,
        DevArr2D<int> q_num,
        DevArr2D<int> qpos_index,
        DevArr2D<int> parent_idx_all,
        DevArr2D<int> is_static_all,
        DevArr2D<int> is_isolated,
        DevArr2D<int> joint_type_all,
        DArray<int>   batch_nv,
        int num_envs, DArray<int> qpos_num, DArray<int> num_groups)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = batch_bodies[env_id];
        int group_count = 0;

        printf("Env %d: num_bodies = %d\n", env_id, env_self_bodies);
        int nv = 0;         // num of generalised DoFs for this env.
        int nqpos = 0;   // num of qpos for this env.

        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int parent_idx = parent_idx_all(env_id, bid);
            const int is_static = is_static_all(env_id, bid);
            printf("parent_id: %d, is_static: %d\n", parent_idx, is_static);

            if (parent_idx == -1)
            {
                is_isolated(env_id, bid) = 1;
                group_count++;
                if(is_static)
                    continue;   // Static root body, no DoFs

                q_index(env_id, bid) = nv;
                q_num(env_id, bid) = 6;
                qpos_index(env_id, bid) = nqpos;
                nv += 6;
                nqpos += 7;
            }
            else
            {
                is_isolated(env_id, parent_idx) = 0;    // parent is not isolated if it has children
                is_isolated(env_id, bid) = 0;           // non-root body is not isolated

                const int& joint_type = joint_type_all(env_id, bid);
                if(joint_type < 3)  // hinge or slide
                {
                    q_index(env_id, bid) = nv;
                    q_num(env_id, bid) = 1;
                    qpos_index(env_id, bid) = nqpos;
                    nv += 1;
                    nqpos += 1;
                }
                else
                {
                    q_index(env_id, bid) = nv;
                    q_num(env_id, bid) = 3;
                    qpos_index(env_id, bid) = nqpos;
                    nv += 3;
                    nqpos += 4;
                }
            }
            printf("Env %d, Body %d, NV: %d\n", env_id, bid, nv);
            printf("Env %d, Body %d, Nqpos: %d\n", env_id, bid, nqpos);
            printf("Env %d, Body %d, cube q_index: %d\n", env_id, bid, q_index(env_id, bid));
            printf("Env %d, Body %d, cube q_num: %d\n", env_id, bid, q_num(env_id, bid));
            printf("Env %d, Body %d, cube q_pos_index: %d\n", env_id, bid, qpos_index(env_id, bid));
        }
        batch_nv[env_id] = nv;
        qpos_num[env_id] = nqpos;
        num_groups[env_id] = group_count;
        

    }

    __global__ void BuildGlobalDofOffsetKernel(
    DArray<int> batch_bodies,          // [env]
    DevArr2D<int> q_lengths,           // [env, body]
    DevArr2D<int> batch_nv_offset,     // [env, body] 输出：全局q起始offset
    DArray<int> batch_nv,              // [env] 可选：做一致性检查
    int num_envs)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;

        int global_q = 0;
        for (int env_id = 0; env_id < num_envs; ++env_id)
        {
            const int num_bodies = batch_bodies[env_id];

            for (int bid = 0; bid < num_bodies; ++bid)
            {
                int qn = q_lengths(env_id, bid);
                if (qn < 0) qn = 0; 
                batch_nv_offset(env_id, bid) = global_q; // 这个body的全局q起点
                global_q += qn;
            }
        }
    }


    __global__ void FillGroupKernel(
        DArray<int> batch_bodies,
        DevArr2D<int> parent_idx,
        DevArr2D<Pair<int, int>> groups,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        auto group_ptr = groups.BlockPtr(env_id);

        int gid = 0;
        int group_begin = -1;

        for (int bid = 0; bid < num_bodies; ++bid)
        {
            if (parent_idx(env_id, bid) == -1)
            {
                if (group_begin != -1)
                {
                    group_ptr[gid] = Pair<int, int>(group_begin, bid - group_begin);
                    ++gid;
                }
                group_begin = bid;
            }
        }

        if (group_begin != -1)
        {
            group_ptr[gid] = Pair<int, int>(group_begin, num_bodies - group_begin);
        }
    }

    __global__ void FillFlattenMappingInfoKernel(
        DArray<int> batch_bodies,
        DArray<int> batch_bodies_offset,
        DevArr2D<int> q_length,
        DevArr2D<int> nv_offset,
        DevArr2D<Pair<int, int>> groups,
        DArray<int> flatten_group_to_env,
        DArray<int> flatten_body_to_env,
        DArray<Pair<int, int>> flatten_q_to_env_body,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        auto group_ptr = groups.BlockPtr(env_id);

        for (int local_gid = 0; local_gid < groups.BlockSize(env_id); ++local_gid)
        {
            flatten_group_to_env[local_gid + groups.BlockOffset(env_id)] = env_id;
            printf("Env %d, Group %d, flatten_group_id: %d\n", env_id, local_gid, local_gid + groups.BlockOffset(env_id));
        }

        for (int bid = 0; bid < num_bodies; ++bid)
        {
            flatten_body_to_env[bid + batch_bodies_offset[env_id]] = env_id;
            printf("Env %d, Body %d, flatten_body_id: %d\n", env_id, bid, bid + batch_bodies_offset[env_id]);
            const int q_num = q_length(env_id, bid);
            const int q_off = nv_offset(env_id, bid);
            printf("Env %d, Body %d, q_num: %d, q_off: %d\n", env_id, bid, q_num, q_off);
            for (int i = 0; i < q_num; ++i)
            {
                flatten_q_to_env_body[q_off + i] = Pair<int, int>(env_id, bid);
                printf("Env %d, Body %d, num_bodies: %d, q_num: %d, flatten_q_id: %d\n", env_id, bid, num_bodies, q_num, q_off + i);
            }
        }

    }


    __global__ void InitInertiaKernel(DArray<int> batch_bodies, DevArr2D<Vec3f> batch_inertia,
        DevArr2D<Real> batch_mass, DArray2D<SphereInfo> spheres, DArray2D<BoxInfo> boxes,
        DArray2D<CapsuleInfo> capsules,
        DArray2D<int> shape_type, DArray2D<int> shape_idx, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = batch_bodies[env_id];
        for(int bid = 0; bid < env_self_bodies; bid++) {
            Real body_mass = batch_mass(env_id, bid);

            switch(shape_type(env_id, bid)) {
                case 0: {
                    Real radius = spheres(env_id, shape_idx(env_id, bid)).radius;
                    Real I = (2.0f / 5.0f) * body_mass * radius * radius;
                    batch_inertia(env_id, bid).x = I;
                    batch_inertia(env_id, bid).y = I;
                    batch_inertia(env_id, bid).z = I;
                    break;
                }

                case 1: {
                    Vec3f halfSize = boxes(env_id, shape_idx(env_id, bid)).halfLength;
                    batch_inertia(env_id, bid).x = (body_mass / 3.f) * (halfSize.y * halfSize.y + halfSize.z * halfSize.z);
                    batch_inertia(env_id, bid).y = (body_mass / 3.f) * (halfSize.x * halfSize.x + halfSize.z * halfSize.z);
                    batch_inertia(env_id, bid).z = (body_mass / 3.f) * (halfSize.x * halfSize.x + halfSize.y * halfSize.y);
                    break;
                }

                case 2: {
                    Real radius = capsules(env_id, shape_idx(env_id, bid)).radius;
                    Real halfLength = capsules(env_id, shape_idx(env_id, bid)).halfLength;
                    Real sphere_mass = 4.f * body_mass * radius / (4.f * radius + 6.f * halfLength);
                    Real cylinder_mass = body_mass - sphere_mass;
                    Real sphere_inertia = 2.f / 5.f * sphere_mass * radius * radius;

                    batch_inertia(env_id, bid).x = cylinder_mass * (3.f * radius * radius + 4.f * halfLength * halfLength) / 12.f
                                                + sphere_inertia + sphere_mass * halfLength * (3.f * radius + 4.f * halfLength) / 4.f;
                    batch_inertia(env_id, bid).y = batch_inertia(env_id, bid).x;
                    batch_inertia(env_id, bid).z = cylinder_mass * radius * radius / 2.f + sphere_inertia;
                    break;
                }
                default: ;
            }
        }
    }

    __global__ void InitQposKernel(DArray<int> batch_bodies, DevArr2D<Real> batch_qpos,
        DevArr2D<int> qpos_offset, DArray2D<Vec3f> batch_pos, DArray2D<Quat<Real>> batch_quat,
        DevArr2D<Real> joint_qpos, DevArr2D<int> joint_qpos_offset,
        DevArr2D<int> parent_idx, DevArr2D<int> joint_type, DevArr2D<int> is_static, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = batch_bodies[env_id];

        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int qpos_start = qpos_offset(env_id, bid);

            if(parent_idx(env_id, bid) == -1)
            {
                if(is_static(env_id, bid))
                    continue;

                for(int i = 0; i < 3; i++)
                    batch_qpos(env_id, qpos_start + i) = batch_pos(env_id, bid)[i];
                batch_qpos(env_id, qpos_start + 3) = batch_quat(env_id, bid).x;
                batch_qpos(env_id, qpos_start + 4) = batch_quat(env_id, bid).y;
                batch_qpos(env_id, qpos_start + 5) = batch_quat(env_id, bid).z;
                batch_qpos(env_id, qpos_start + 6) = batch_quat(env_id, bid).w;
            }
            else
            {
                int jqpos_start = joint_qpos_offset(env_id, bid);
                if(joint_type(env_id, bid) < 3)
                    batch_qpos(env_id, qpos_start) = joint_qpos(env_id, jqpos_start);
                else
                    for(int i = 0; i < 4; i++)
                        batch_qpos(env_id, qpos_start + i) = joint_qpos(env_id, jqpos_start + i);
            }

            printf("Env %d, Body %d, q_pos: %f %f %f %f %f %f %f\n", env_id, bid,
                batch_qpos(env_id, qpos_start), batch_qpos(env_id, qpos_start + 1),
                batch_qpos(env_id, qpos_start + 2), batch_qpos(env_id, qpos_start + 3),
                batch_qpos(env_id, qpos_start + 4), batch_qpos(env_id, qpos_start + 5),
                batch_qpos(env_id, qpos_start + 6));
        }
    }

    __global__ void BuildRootIndexKernel(
        DArray<int> batch_bodies,
        DevArr2D<int> root_idx,
        DevArr2D<Real> subtree_mass,
        DevArr2D<Real> batch_mass,
        DevArr2D<int> parent_idx,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int pidx = parent_idx(env_id, bid);
            subtree_mass(env_id, bid) = batch_mass(env_id, bid);
            root_idx(env_id, bid) = (pidx == -1) ? bid : root_idx(env_id, pidx);
            printf("Env %d, Body %d, Root Index: %d\n", env_id, bid, root_idx(env_id, bid));
        }
    }

    __global__ void CalculateSubtreeMassKernel(
        DArray<int> batch_bodies,
        DevArr2D<Real> subtree_mass,
        DevArr2D<int> parent_idx,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        for(int bid = num_bodies - 1; bid >= 0; bid--)
        {
            const int pidx = parent_idx(env_id, bid);
            if(pidx != -1)
                subtree_mass(env_id, pidx) += subtree_mass(env_id, bid);

            printf("Env %d, Body %d, Subtree Mass: %f\n", env_id, bid, subtree_mass(env_id, bid));
        }
    }

    __global__ void UpdateGeneralizedVelKernel(
        DArray<int> batch_nv,
        DevArr2D<Real> batch_qvel,
        DevArr2D<Real> batch_qacc,
        DArray<Real> timesteps,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_nv = batch_nv[env_id];
        int dof_idx = threadIdx.x;
        if(dof_idx >= num_nv)
            return;

        const Real dt = timesteps[env_id];
        batch_qvel(env_id, dof_idx) += batch_qacc(env_id, dof_idx) * dt;
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
    __global__ void UpdateAnchorConstarints(
        DArray<Vec4i> num_each_constraint,
        BatchAnchorConstraints anchor_constraints,
        DArray2D<Mat3f> batch_rot,
        DArray2D<Vec3f> batch_pos,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        if(anchor_idx >= num_each_constraint[env_id][0] / 3)
            return;

        const int body_A_idx = anchor_constraints.body_idxs(env_id, anchor_idx).first;
        const int body_B_idx = anchor_constraints.body_idxs(env_id, anchor_idx).second;

        const Mat3f& rot_A = batch_rot(env_id, body_A_idx);
        const Mat3f& rot_B = batch_rot(env_id, body_B_idx);
        const Vec3f& local_anchor_A = anchor_constraints.anchor_A_local(env_id, anchor_idx);
        const Vec3f& local_anchor_B = anchor_constraints.anchor_B_local(env_id, anchor_idx);
        const Vec3f& pos_A = batch_pos(env_id, body_A_idx);
        const Vec3f& pos_B = batch_pos(env_id, body_B_idx);
        Vec3f& global_anchor_A = anchor_constraints.anchor_A_world(env_id, anchor_idx);
        Vec3f& global_anchor_B = anchor_constraints.anchor_B_world(env_id, anchor_idx);
        Vec3f& pos_err = anchor_constraints.anchor_error(env_id, anchor_idx);

        global_anchor_A = rot_A * local_anchor_A + pos_A;
        global_anchor_B = rot_B * local_anchor_B + pos_B;
        pos_err = global_anchor_A - global_anchor_B;
    }

    template<typename TDataType>
    __global__ void UpdateJointLimitConstraints(
        BatchJointLimitConstraints joint_limits,
        DevArr2D<int> joint_type,
        DevArr2D<int> joint_qpos_offset,
        DevArr2D<Real> joint_qpos,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int jl_idx = threadIdx.x;
        if(jl_idx >= joint_limits.ref_nums[env_id])
            return;

        const int bid = joint_limits.joint_idx(env_id, jl_idx);
        const int jt = joint_type(env_id, bid);
        const int is_upper = joint_limits.is_upper(env_id, jl_idx);
        const int qpos_idx = joint_qpos_offset(env_id, bid);
        const Real limit = joint_limits.limit(env_id, jl_idx);
        auto& limit_err = joint_limits.limit_error(env_id, jl_idx);
        auto& is_active = joint_limits.is_active(env_id, jl_idx);
        auto& limit_extern = joint_limits.limit_extern(env_id, jl_idx);
        is_active = 0;

        if(jt < 3)
        {
            Real dist = joint_qpos(env_id, qpos_idx) - limit;
            if((is_upper && dist > 0.f) || (!is_upper && dist < 0.f))
            {
                limit_err = dist;
                is_active = 1;
            }
        }
        else
        {
            Quat<Real> quat = Quat<Real>(joint_qpos(env_id, qpos_idx), joint_qpos(env_id, qpos_idx + 1), joint_qpos(env_id, qpos_idx + 2), joint_qpos(env_id, qpos_idx + 3));
            quat.normalize();

            Vec3f angle_vel;
            Real angle;
            Quat2Vel(quat, angle, angle_vel, 1.f);
            Real dist = limit - angle;
            if(dist < 0.f)
            {
                limit_err = dist;
                is_active = 1;
                limit_extern = angle_vel.normalize();
            }
        }
        printf("Env %d, Joint Limit Constraint %d, is_upper: %d, limit: %f, qpos: %f, dist: %f, is_active: %d\n",
            env_id, jl_idx, is_upper, limit, joint_qpos(env_id, qpos_idx), limit_err, is_active);
    }

    template<typename TDataType>
    __global__ void CountConstraintNums(
        DArray<Vec4i> num_each_constraint,
        BatchJointLimitConstraints joint_limits,
        BatchCollisionConstraints collisions,
        DArray<Vec4i> constraint_offset,
        DArray<int> num_constraints,
        int num_envs)
    {
        int env_id = threadIdx.x;
        if(env_id >= num_envs)
            return;

        int active_jl_num = 0;
        for(int i = 0; i < joint_limits.ref_nums[env_id]; i++)
        {
            if(joint_limits.is_active(env_id, i))
                joint_limits.active_mapping(env_id, active_jl_num++) = i;
        }

        num_each_constraint[env_id][2] = active_jl_num;
        num_each_constraint[env_id][3] = collisions.collision_nums[env_id] * 4;

        auto& offsets = constraint_offset[env_id];
        auto& total = num_constraints[env_id];

        total = 0;
        for(int i = 0; i < 4; i++)
            total += num_each_constraint[env_id][i];

        offsets[2] = num_each_constraint[env_id][0] + num_each_constraint[env_id][1];
        offsets[3] = offsets[2] + num_each_constraint[env_id][2];
    }

    template<typename TDataType>
    __global__ void CollisonDetectionKernel(
        BatchCollisionConstraints collision_constraints,
        DArray<int> batch_bodies,
        DevArr2D<int> is_static,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Mat3f> batch_rot,
        DArray2D<int> shape_type,
        DArray2D<int> shape_idx,
        DArray2D<BoxInfo> boxes,
        int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        for(int bid = 0; bid < num_bodies; bid++)
        {
            if(is_static(env_id, bid))
                continue;

            if(shape_type(env_id, bid) == 1)
            {
                const Vec3f& pos = batch_pos(env_id, bid);
                const Mat3f& rot = batch_rot(env_id, bid);
                const int sidx = shape_idx(env_id, bid);
                CubeCollitionWithGround(pos, rot, boxes(env_id, sidx), collision_constraints, env_id, bid);
            }
        }
    }

    __device__ void ComputeJacLocal(Real* dst_jac, const Vec3f& c_point,
        const DevArr2D<int>& root_idx, const DevArr2D<Vec3f>& subtree_com, const DevArr2D<Real>& batch_cdof,
        const DArray<int>& batch_nv, const DevArr2D<int>& is_static, const DevArr2D<int>& q_offset,
        const DevArr2D<int>& q_lengths, const DevArr2D<int>& parent_idx, int env_id, int bid)
    {
        const int root = root_idx(env_id, bid);
        Vec3f offset = c_point - subtree_com(env_id, root);
        const int nv = batch_nv[env_id];

        for(int i = 0; i < 6 * nv; i++)
            dst_jac[i] = 0.f;

        if(is_static(env_id, bid))
            return;

        int j = bid;
        while(j != -1)
        {
            int q_start = q_offset(env_id, j);
            int q_num = q_lengths(env_id, j);

            for(int k = q_num - 1; k >= 0; k--)
            {
                int q_idx = q_start + k;
                Vec3f cdof_angular = Vec3f(batch_cdof(env_id, q_idx * 6), batch_cdof(env_id, q_idx * 6 + 1), batch_cdof(env_id, q_idx * 6 + 2));
                Vec3f d = cross(cdof_angular, offset);
                for(int i = 0; i < 6; i++)
                    dst_jac[i * nv + q_idx] = batch_cdof(env_id, q_idx * 6 + i);
                for(int i = 3; i < 6; i++)
                    dst_jac[i * nv + q_idx] += d[i - 3];
            }
            j = parent_idx(env_id, j);
        }
    }

    template<typename TDataType>
    __global__ void ContactConstraintJacobianKernel(
        BatchCollisionConstraints collisions,
        DArray<int> batch_nv,
        DArray<Vec4i> constraint_offset,
        DevMat2D<Real> batch_J,
        DevArr2D<int> root_idx,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<Real> batch_cdof,
        DevArr2D<int> is_static,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<int> parent_idx,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_collisions = collisions.collision_nums[env_id];
        const int num_nv = batch_nv[env_id];
        int cidx = threadIdx.x;
        if(cidx >= num_collisions)
            return;

        const Vec3f& normal = collisions.normal(env_id, cidx);
        const Vec3f& c_point = collisions.point(env_id, cidx);
        int a_idx = collisions.body_idxs(env_id, cidx).first;
        int b_idx = collisions.body_idxs(env_id, cidx).second;

        Vec3f t = abs(normal.y) < 0.5 ? Vec3f(0.f, 1.f, 0.f) : Vec3f(0.f, 0.f, 1.f);
        Vec3f y = t - dot(t, normal) * normal;
        y.normalize();
        Vec3f z = cross(normal, y);

        Mat3f c_basis;
        c_basis.setCol(0, normal);
        c_basis.setCol(1, y);
        c_basis.setCol(2, z);

        Real jacA[6 * NV_TMP];
        Real jacB[6 * NV_TMP];

        ComputeJacLocal(jacA, c_point, root_idx, subtree_com, batch_cdof, batch_nv, is_static, q_offset, q_lengths, parent_idx, env_id, a_idx);
        if(b_idx != -1)
            ComputeJacLocal(jacB, c_point, root_idx, subtree_com, batch_cdof, batch_nv, is_static, q_offset, q_lengths, parent_idx, env_id, b_idx);
        else
            for(int i = 0; i < 6 * num_nv; i++) jacB[i] = 0.f;

        for(int j = 0; j < 3; j++)
            for(int k = 0; k < num_nv; k++)
                jacA[j * num_nv + k] = jacA[(j + 3) * num_nv + k] - jacB[(j + 3) * num_nv + k];

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < num_nv; j++)
            {
                Real sum = 0;
                for(int k = 0; k < 3; k++)
                    sum += c_basis(k, i) * jacA[k * num_nv + j];
                jacB[i * num_nv + j] = sum;
            }

        const Real mu = collisions.mu(env_id, cidx);
        const int row0 = constraint_offset[env_id][3] + 4 * cidx;
        for(int i = 0; i < num_nv; i++)
        {
            const Real j0j = jacB[i];
            const Real j1j = jacB[num_nv + i];
            const Real j2j = jacB[2 * num_nv + i];

            batch_J(env_id, row0, i) = j0j + mu * j1j;
            batch_J(env_id, row0 + 1, i) = j0j - mu * j1j;
            batch_J(env_id, row0 + 2, i) = j0j + mu * j2j;
            batch_J(env_id, row0 + 3, i) = j0j - mu * j2j;
        }
    }

    template<typename TDataType>
    __global__ void AnchorConstraintJacobianKernel(
        BatchAnchorConstraints batch_anchor,
        DArray<Vec4i> num_each_constraint,
        DArray<int> batch_nv,
        DevMat2D<Real> batch_J,
        DevArr2D<int> root_idx,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<Real> batch_cdof,
        DevArr2D<int> is_static,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<int> parent_idx,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        const int anchor_nums = num_each_constraint[env_id][0] / 3;
        if(anchor_idx >= anchor_nums)
            return;

        const int a_idx = batch_anchor.body_idxs(env_id, anchor_idx).first;
        const int b_idx = batch_anchor.body_idxs(env_id, anchor_idx).second;
        const Vec3f& anchor_A_global = batch_anchor.anchor_A_world(env_id, anchor_idx);
        const Vec3f& anchor_B_global = batch_anchor.anchor_B_world(env_id, anchor_idx);
        const int nv = batch_nv[env_id];

        Real jacA[6 * NV_TMP];
        Real jacB[6 * NV_TMP];
        ComputeJacLocal(jacA, anchor_A_global, root_idx, subtree_com, batch_cdof, batch_nv, is_static, q_offset, q_lengths, parent_idx, env_id, a_idx);
        ComputeJacLocal(jacB, anchor_B_global, root_idx, subtree_com, batch_cdof, batch_nv, is_static, q_offset, q_lengths, parent_idx, env_id, b_idx);

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < nv; j++)
            {
                int row = anchor_idx * 3 + i;
                batch_J(env_id, row, j) = jacA[(i + 3) * nv + j] - jacB[(i + 3) * nv + j];
            }
    }

    template<typename TDataType>
    __global__ void FrictionLossJacobianKernel(
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DArray<int> batch_nv,
        DevMat2D<Real> batch_J,
        BatchFrictionLossConstraints friction_loss_constraints,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;
        const int constraint_num = num_each_constraint[env_id][1];
        if(cidx >= constraint_num)
            return;

        const int constraint_start = constraint_offset[env_id][1];
        const int nv = batch_nv[env_id];
        const int nv_idx = friction_loss_constraints.dof_idxs(env_id, cidx);
        batch_J(env_id, constraint_start + cidx, nv_idx) = 1.f;
    }

    template<typename TDataType>
    __global__ void JointLimitJacobianKernel(
        DArray<Vec4i> num_each_constraint,
        BatchJointLimitConstraints constraints,
        DevArr2D<int> q_offset,
        DevArr2D<int> joint_type,
        DArray<Vec4i> constraint_offset,
        DArray<int> batch_nv,
        DevMat2D<Real> batch_J,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;
        const int constraint_num = num_each_constraint[env_id][2];
        if(cidx >= constraint_num)
            return;

        const int jl_cid = constraints.active_mapping(env_id, cidx);
        const int bid = constraints.joint_idx(env_id, jl_cid);
        const int q_start = q_offset(env_id, bid);
        const int jt = joint_type(env_id, bid);
        const Real pos_err = constraints.limit_error(env_id, jl_cid);
        const int row_idx = constraint_offset[env_id][2] + cidx;
        const int nv = batch_nv[env_id];

        if(jt < 3)
            batch_J(env_id, row_idx, q_start) = pos_err < 0.f ? 1.f : -1.f;
        else
        {
            const Vec3f& axis = constraints.limit_extern(env_id, jl_cid);
            for(int i = 0; i < 3; i++)
                batch_J(env_id, row_idx, q_start + i) = -axis[i];
        }
    }

    template<typename TDataType>
    __global__ void PrintJacobian(DevArr2D<Real> batch_J, DArray<int> num_constraints, DArray<int> batch_nv, int env_id)
    {
        if(threadIdx.x != 0)
            return;

        const int nc = num_constraints[env_id];
        const int nv = batch_nv[env_id];
        for(int i = 0; i < nc; i++)
        {
            for(int j = 0; j < nv; j++)
                printf("%f\t", batch_J(env_id, i * nv + j));
            printf("\n");
        }
    }

    __device__ Vec4f ComputeKBIP(Real error, Real dmax, Real dmin, Real time_const,
        Real damp_ratio, Real midpoint, Real width, Real power)
    {
        Real K = 1.f / (dmax * dmax * time_const * time_const * damp_ratio * damp_ratio);
        Real B = 2.f / (dmax * time_const);

        Real x = error / width;
        Real sign = x < 0.f ? -1.f : 1.f;
        x *= sign;

        if(x > 1.f)
            return Vec4f(K, B, dmax, 0.f);
        if(x < 0.f)
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
    __global__ void ComputeAnchorAref(
        BatchAnchorConstraints anchor_constraints,
        DArray<Vec4i> num_each_constraint,
        DevArr2D<Real> batch_constraint_vel,
        DevArr2D<Real> batch_aref,
        DevArr2D<Real> batch_imp,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        if(anchor_idx >= num_each_constraint[env_id][0] / 3)
            return;

        const auto& dmax = anchor_constraints.dmax(env_id, anchor_idx);
        const auto& dmin = anchor_constraints.dmin(env_id, anchor_idx);
        const auto& time_const = anchor_constraints.time_const(env_id, anchor_idx);
        const auto& damp_ratio = anchor_constraints.damp_ratio(env_id, anchor_idx);
        const auto& midpoint = anchor_constraints.midpoint(env_id, anchor_idx);
        const auto& width = anchor_constraints.width(env_id, anchor_idx);
        const auto& power = anchor_constraints.power(env_id, anchor_idx);
        const Vec3f& pos_err = anchor_constraints.anchor_error(env_id, anchor_idx);

        Real pos_err_norm = pos_err.norm();
        Vec4f KBIP = ComputeKBIP(pos_err_norm, dmax, dmin, time_const, damp_ratio, midpoint, width, power);
        Real K = KBIP[0];
        Real B = KBIP[1];
        Real I = KBIP[2];
        for(int i = 0; i < 3; i++)
        {
            batch_imp(env_id, anchor_idx * 3 + i) = I;
            batch_aref(env_id, anchor_idx * 3 + i) = -B * batch_constraint_vel(env_id, anchor_idx * 3 + i) - K * I * pos_err[i];
        }
    }

    template<typename TDataType>
    __global__ void ComputeFrictionLossAref(
        BatchFrictionLossConstraints friction_loss_constraints,
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_constraint_vel,
        DevArr2D<Real> batch_imp,
        DevArr2D<Real> batch_aref,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int fidx = threadIdx.x;
        if(fidx >= num_each_constraint[env_id][1])
            return;

        const int c_offset = constraint_offset[env_id][1];
        const auto& dmax = friction_loss_constraints.dmax(env_id, fidx);
        const auto& dmin = friction_loss_constraints.dmin(env_id, fidx);
        const auto& time_const = friction_loss_constraints.time_const(env_id, fidx);
        const auto& damp_ratio = friction_loss_constraints.damp_ratio(env_id, fidx);
        const auto& midpoint = friction_loss_constraints.midpoint(env_id, fidx);
        const auto& width = friction_loss_constraints.width(env_id, fidx);
        const auto& power = friction_loss_constraints.power(env_id, fidx);

        Vec4f KBIP = ComputeKBIP(0.f, dmax, dmin, time_const, damp_ratio, midpoint, width, power);
        batch_imp(env_id, c_offset + fidx) = KBIP[2];
        batch_aref(env_id, c_offset + fidx) = -KBIP[1] * batch_constraint_vel(env_id, c_offset + fidx);
    }

    template<typename TDataType>
    __global__ void ComputeJointLimitAref(
        BatchJointLimitConstraints joint_limit_constraints,
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_constraint_vel,
        DevArr2D<Real> batch_imp,
        DevArr2D<Real> batch_aref,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;
        if(cidx >= num_each_constraint[env_id][2])
            return;

        const int c_offset = constraint_offset[env_id][2];
        const int jl_cidx = joint_limit_constraints.active_mapping(env_id, cidx);
        const auto& dmax = joint_limit_constraints.dmax(env_id, jl_cidx);
        const auto& dmin = joint_limit_constraints.dmin(env_id, jl_cidx);
        const auto& time_const = joint_limit_constraints.time_const(env_id, jl_cidx);
        const auto& damp_ratio = joint_limit_constraints.damp_ratio(env_id, jl_cidx);
        const auto& midpoint = joint_limit_constraints.midpoint(env_id, jl_cidx);
        const auto& width = joint_limit_constraints.width(env_id, jl_cidx);
        const auto& power = joint_limit_constraints.power(env_id, jl_cidx);
        const auto& pos_err = joint_limit_constraints.limit_error(env_id, jl_cidx);

        Real pos_err_abs = abs(pos_err);
        Vec4f KBIP = ComputeKBIP(pos_err_abs, dmax, dmin, time_const, damp_ratio, midpoint, width, power);
        Real K = KBIP[0];
        Real B = KBIP[1];
        Real I = KBIP[2];
        batch_imp(env_id, c_offset + cidx) = I;
        batch_aref(env_id, c_offset + cidx) = -B * batch_constraint_vel(env_id, c_offset + cidx) + K * I * pos_err_abs;
    }

    template<typename T>
    __device__ void ComputeContactParas(const BatchCollisionConstraints& sys_contact_paras, const Pair<int, int>& body_idxs, int env_id, const T& wa, const T& wb,
        T& time_const, T& damp_ratio, T& dmax, T& dmin, T& midpoint, T& width, T& power)
    {
        int a_idx = body_idxs.first;
        int b_idx = body_idxs.second;

        const Real& a_time_const = sys_contact_paras.time_const(env_id, a_idx);
        const Real& a_damp_ratio = sys_contact_paras.damp_ratio(env_id, a_idx);
        const Real& a_dmax = sys_contact_paras.dmax(env_id, a_idx);
        const Real& a_dmin = sys_contact_paras.dmin(env_id, a_idx);
        const Real& a_midpoint = sys_contact_paras.midpoint(env_id, a_idx);
        const Real& a_width = sys_contact_paras.width(env_id, a_idx);
        const Real& a_power = sys_contact_paras.power(env_id, a_idx);

        if(b_idx != -1)
        {
            const Real& b_time_const = sys_contact_paras.time_const(env_id, b_idx);
            const Real& b_damp_ratio = sys_contact_paras.damp_ratio(env_id, b_idx);
            const Real& b_dmax = sys_contact_paras.dmax(env_id, b_idx);
            const Real& b_dmin = sys_contact_paras.dmin(env_id, b_idx);
            const Real& b_midpoint = sys_contact_paras.midpoint(env_id, b_idx);
            const Real& b_width = sys_contact_paras.width(env_id, b_idx);
            const Real& b_power = sys_contact_paras.power(env_id, b_idx);

            Real w_a = wa / (wa + wb);
            Real w_b = wb / (wa + wb);

            time_const = w_a * a_time_const + w_b * b_time_const;
            damp_ratio = w_a * a_damp_ratio + w_b * b_damp_ratio;
            dmax = w_a * a_dmax + w_b * b_dmax;
            dmin = w_a * a_dmin + w_b * b_dmin;
            midpoint = w_a * a_midpoint + w_b * b_midpoint;
            width = w_a * a_width + w_b * b_width;
            power = w_a * a_power + w_b * b_power;
        }
        else
        {
            time_const = a_time_const;
            damp_ratio = a_damp_ratio;
            dmax = a_dmax;
            dmin = a_dmin;
            midpoint = a_midpoint;
            width = a_width;
            power = a_power;
        }
    }

    template<typename TDataType>
    __global__ void ComputeContactAref(
        BatchCollisionConstraints collision_constraints,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_constraint_vel,
        DevArr2D<Real> contact_weights,
        DevArr2D<Real> batch_imp,
        DevArr2D<Real> batch_aref,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int contact_idx = threadIdx.x;
        if(contact_idx >= collision_constraints.collision_nums[env_id])
            return;

        const int constraint_start = constraint_offset[env_id][3];
        const auto& depth = collision_constraints.depth(env_id, contact_idx);
        const auto& body_idxs = collision_constraints.body_idxs(env_id, contact_idx);
        const int a_idx = body_idxs.first;
        const int b_idx = body_idxs.second;

        const Real& wa = contact_weights(env_id, a_idx);
        Real wb = b_idx == -1 ? 0.f : contact_weights(env_id, b_idx);

        Real time_const, damp_ratio, dmax, dmin, midpoint, width, power;
        ComputeContactParas(collision_constraints, body_idxs, env_id, wa, wb, time_const, damp_ratio, dmax, dmin, midpoint, width, power);
        Vec4f KBIP = ComputeKBIP(depth, dmax, dmin, time_const, damp_ratio, midpoint, width, power);

        for(int i = 0; i < 4; i++)
        {
            int idx = constraint_start + contact_idx * 4 + i;
            Real K = KBIP[0];
            Real B = KBIP[1];
            Real I = KBIP[2];
            batch_imp(env_id, idx) = I;
            batch_aref(env_id, idx) = -B * batch_constraint_vel(env_id, idx) + K * I * depth;
        }
    }

    __device__ void RotateJacobianRow(Real* jac_src, Real* jac_dst, int num_cols)
    {
        for(int i = 0; i < num_cols; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                jac_dst[j * num_cols + i] = jac_src[(j + 3) * num_cols + i];
                jac_dst[(j + 3) * num_cols + i] = jac_src[j * num_cols + i];
            }
        }
    }

    template<typename TDataType>
    __global__ void ComputeDiagJMinvJTForBodies(
        DArray<int> batch_bodies,
        DevArr2D<int> is_static,
        DArray<int> batch_nv,
        DevArr2D<Vec3f> batch_global_com_pos,
        DevArr2D<int> root_idx,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<Real> batch_cdof,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<int> parent_idx,
        DevMat2D<Real> batch_qM_L,
        DevArr2D<Real> batch_weight_inv,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int bid = threadIdx.x;
        int env_self_bodies = batch_bodies[env_id];
        if(bid >= env_self_bodies)
            return;

        if(is_static(env_id, bid))
        {
            batch_weight_inv(env_id, bid) = 0.f;
            return;
        }

        const int nv = batch_nv[env_id];
        if(nv <= 0)
        {
            batch_weight_inv(env_id, bid) = 0.f;
            return;
        }

        Real jac[6 * NV_TMP];
        Real j_tmp[6 * NV_TMP];
        ComputeJacLocal(j_tmp, batch_global_com_pos(env_id, bid), root_idx, subtree_com, batch_cdof, batch_nv, is_static, q_offset, q_lengths, parent_idx, env_id, bid);
        RotateJacobianRow(j_tmp, jac, nv);

        const auto& L = batch_qM_L;
        for(int r = 0; r < 6; r++)
        {
            for(int i = 0; i < nv; i++)
            {
                Real sum = jac[r * nv + i];
                for(int k = 0; k < i; k++)
                    sum -= L(env_id, i, k) * j_tmp[r * nv + k];

                const Real lii = L(env_id, i, i);
                j_tmp[r * nv + i] = sum / lii;
            }

            for(int i = nv - 1; i >= 0; i--)
            {
                Real sum = j_tmp[r * nv + i];
                for(int k = i + 1; k < nv; k++)
                    sum -= L(env_id, k, i) * j_tmp[r * nv + k];

                const Real lii = L(env_id, i, i);
                j_tmp[r * nv + i] = sum / lii;
            }
        }

        Real a00 = 0.f;
        Real a11 = 0.f;
        Real a22 = 0.f;
        for(int k = 0; k < nv; k++)
        {
            a00 += jac[0 * nv + k] * j_tmp[0 * nv + k];
            a11 += jac[1 * nv + k] * j_tmp[1 * nv + k];
            a22 += jac[2 * nv + k] * j_tmp[2 * nv + k];
        }

        batch_weight_inv(env_id, bid) = (a00 + a11 + a22) / 3.f;
    }

    template<typename TDataType>
    __global__ void ComputeDiagJMinvJTForJoints(
        DArray<int> batch_bodies,
        DArray<int> batch_nv,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<int> joint_type,
        DevMat2D<Real> batch_qM_L,
        DevArr2D<Real> batch_dof_weight_inv,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int bid = threadIdx.x;
        int num_bodies = batch_bodies[env_id];
        if(bid >= num_bodies)
            return;

        const int nv = batch_nv[env_id];
        const int q_start = q_offset(env_id, bid);
        const int jt = joint_type(env_id, bid);
        if(jt == 0)
            return;

        const auto& L = batch_qM_L;
        auto& dof_weight_inv = batch_dof_weight_inv;

        if(jt < 3)
        {
            Real x[NV_TMP];
            for(int i = 0; i < nv; i++)
                x[i] = (i == q_start) ? 1.f : 0.f;

            for(int i = 0; i < nv; i++)
            {
                Real sum = x[i];
                for(int k = 0; k < i; k++)
                    sum -= L(env_id, i, k) * x[k];
                const Real lii = L(env_id, i, i);
                x[i] = sum / lii;
            }

            for(int i = nv - 1; i >= 0; i--)
            {
                Real sum = x[i];
                for(int k = i + 1; k < nv; k++)
                    sum -= L(env_id, k, i) * x[k];
                const Real lii = L(env_id, i, i);
                x[i] = sum / lii;
            }

            dof_weight_inv(env_id, q_start) = x[q_start];
        }
        else
        {
            Real w_sum = 0.f;
            for(int axis = 0; axis < 3; axis++)
            {
                const int qidx = q_start + axis;
                Real x[NV_TMP];
                for(int i = 0; i < nv; i++)
                    x[i] = (i == qidx) ? 1.f : 0.f;

                for(int i = 0; i < nv; i++)
                {
                    Real sum = x[i];
                    for(int k = 0; k < i; k++)
                        sum -= L(env_id, i, k) * x[k];
                    const Real lii = L(env_id, i, i);
                    x[i] = sum / lii;
                }

                for(int i = nv - 1; i >= 0; i--)
                {
                    Real sum = x[i];
                    for(int k = i + 1; k < nv; k++)
                        sum -= L(env_id, k, i) * x[k];
                    const Real lii = L(env_id, i, i);
                    x[i] = sum / lii;
                }

                w_sum += x[qidx];
            }

            const Real w = w_sum / 3.f;
            dof_weight_inv(env_id, q_start + 0) = w;
            dof_weight_inv(env_id, q_start + 1) = w;
            dof_weight_inv(env_id, q_start + 2) = w;
        }
    }

    template<typename TDataType>
    __global__ void ComputeAnchor_dAKernel(
        DArray<Vec4i> num_each_constraint,
        BatchAnchorConstraints anchor_constraints,
        DevArr2D<Real> batch_weight_inv,
        DevArr2D<Real> batch_dA,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        if(anchor_idx >= num_each_constraint[env_id][0] / 3)
            return;

        const int body_A_idx = anchor_constraints.body_idxs(env_id, anchor_idx).first;
        const int body_B_idx = anchor_constraints.body_idxs(env_id, anchor_idx).second;
        Real w = batch_weight_inv(env_id, body_A_idx) + batch_weight_inv(env_id, body_B_idx);

        for(int i = 0; i < 3; i++)
            batch_dA(env_id, anchor_idx * 3 + i) = w;
    }

    template<typename TDataType>
    __global__ void ComputeFrictionLoss_dAKernel(
        DArray<Vec4i> num_each_constraint,
        BatchFrictionLossConstraints friction_loss_constraints,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_dof_weight_inv,
        DevArr2D<Real> batch_dA,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int fidx = threadIdx.x;
        if(fidx >= num_each_constraint[env_id][1])
            return;

        const int dof_idx = friction_loss_constraints.dof_idxs(env_id, fidx);
        const int c_offset = constraint_offset[env_id][1];
        batch_dA(env_id, c_offset + fidx) = batch_dof_weight_inv(env_id, dof_idx);
    }

    template<typename TDataType>
    __global__ void ComputeJointLimit_dAKernel(
        DArray<Vec4i> num_each_constraint,
        BatchJointLimitConstraints joint_limit_constraints,
        DevArr2D<int> q_offset,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_dof_weight_inv,
        DevArr2D<Real> batch_dA,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int jidx = threadIdx.x;
        if(jidx >= num_each_constraint[env_id][2])
            return;

        const int jl_cidx = joint_limit_constraints.active_mapping(env_id, jidx);
        const int body_idx = joint_limit_constraints.joint_idx(env_id, jl_cidx);
        const int q_start = q_offset(env_id, body_idx);
        const int c_offset = constraint_offset[env_id][2];
        batch_dA(env_id, c_offset + jidx) = batch_dof_weight_inv(env_id, q_start);
    }

    template<typename TDataType>
    __global__ void ComputeContact_dAKernel(
        BatchCollisionConstraints collision_constraints,
        DevArr2D<Real> batch_weight_inv,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_dA,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int contact_idx = threadIdx.x;
        const int num_contacts = collision_constraints.collision_nums[env_id];
        if(contact_idx >= num_contacts)
            return;

        const int c_offset = constraint_offset[env_id][3];
        int body_A_idx = collision_constraints.body_idxs(env_id, contact_idx).first;
        int body_B_idx = collision_constraints.body_idxs(env_id, contact_idx).second;
        Real mu = collision_constraints.mu(env_id, contact_idx);

        Real w = body_B_idx != -1 ? batch_weight_inv(env_id, body_A_idx) + batch_weight_inv(env_id, body_B_idx) : batch_weight_inv(env_id, body_A_idx);
        Real tmp = w * (1 + mu * mu);

        for(int i = 0; i < 4; i++)
            batch_dA(env_id, c_offset + contact_idx * 4 + i) = tmp;
    }

    template<typename TDataType>
    __global__ void ComputeRKernel(
        DArray<int> num_constraints,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_imp,
        DevArr2D<Real> batch_dA,
        BatchCollisionConstraints collision_constraints,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int cidx = threadIdx.x;
        const int nc = num_constraints[env_id];
        if(cidx >= nc)
            return;

        const int collision_offset = constraint_offset[env_id][3];
        batch_D(env_id, cidx) = (1.f - batch_imp(env_id, cidx)) * batch_dA(env_id, cidx) / batch_imp(env_id, cidx);
        if(cidx >= collision_offset)
        {
            const Real mu = collision_constraints.mu(env_id, (cidx - collision_offset) / 4);
            batch_D(env_id, cidx) *= 2.f * mu * mu;
        }
    }

    template<typename TDataType>
    __global__ void ComputeDKernel(DArray<int> num_constraints, DevArr2D<Real> batch_D, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int cidx = threadIdx.x;
        const int nc = num_constraints[env_id];
        if(cidx >= nc)
            return;

        batch_D(env_id, cidx) = 1.f / batch_D(env_id, cidx);
    }

    template<typename TDataType>
    __global__ void AnchorEnergyKernel(
        DArray<int> is_converged,
        DArray<Vec4i> num_each_constraint,
        DevArr2D<Real> batch_constraint_force,
        DevArr2D<Real> batch_constraint_energy,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_Jaref,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs || is_converged[env_id])
            return;

        int anchor_num = num_each_constraint[env_id][0];
        int cidx = threadIdx.x;
        if(cidx >= anchor_num)
            return;

        const Real D = batch_D(env_id, cidx);
        const Real Jaref = batch_Jaref(env_id, cidx);
        batch_constraint_force(env_id, cidx) = -D * Jaref;
        batch_constraint_energy(env_id, cidx) = 0.5f * D * Jaref * Jaref;
    }

    template<typename TDataType>
    __global__ void FrictionLossEnergyKernel(
        DArray<int> is_converged,
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_constraint_force,
        DevArr2D<Real> batch_constraint_energy,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_Jaref,
        BatchFrictionLossConstraints friction_loss_constraints,
        DevArr2D<int> batch_unquads,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs || is_converged[env_id])
            return;

        int c_num = num_each_constraint[env_id][1];
        int cidx = threadIdx.x;
        if(cidx >= c_num)
            return;

        int c_offset = constraint_offset[env_id][1];
        const int ridx = c_offset + cidx;
        Real D = batch_D(env_id, ridx);
        Real Jaref = batch_Jaref(env_id, ridx);
        Real dof_frictionloss = friction_loss_constraints.dof_frictionloss(env_id, cidx);

        batch_constraint_force(env_id, ridx) = -D * Jaref;
        Real R_dof_fl = dof_frictionloss / D;

        if(Jaref <= -R_dof_fl)
        {
            batch_constraint_energy(env_id, ridx) = -0.5f * R_dof_fl * dof_frictionloss - dof_frictionloss * Jaref;
            batch_constraint_force(env_id, ridx) = dof_frictionloss;
            batch_unquads(env_id, ridx) = 1;
        }
        else if(Jaref >= R_dof_fl)
        {
            batch_constraint_energy(env_id, ridx) = -0.5f * R_dof_fl * dof_frictionloss + dof_frictionloss * Jaref;
            batch_constraint_force(env_id, ridx) = -dof_frictionloss;
            batch_unquads(env_id, ridx) = 1;
        }
        else
        {
            batch_constraint_energy(env_id, ridx) = 0.5f * D * Jaref * Jaref;
        }
    }

    template<typename TDataType>
    __global__ void ContactAndJointLimitEnergyKernel(
        DArray<int> is_converged,
        DArray<int> num_constraints,
        DArray<Vec4i> constraint_offset,
        DevArr2D<Real> batch_constraint_force,
        DevArr2D<Real> batch_constraint_energy,
        DevArr2D<Real> batch_Jaref,
        DevArr2D<Real> batch_D,
        DevArr2D<int> batch_unquads,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs || is_converged[env_id])
            return;

        int constraint_start = constraint_offset[env_id][2];
        int cidx = threadIdx.x + constraint_start;
        if(cidx >= num_constraints[env_id])
            return;

        Real Jaref_cidx = batch_Jaref(env_id, cidx);
        Real D_cidx = batch_D(env_id, cidx);
        batch_constraint_force(env_id, cidx) = -D_cidx * Jaref_cidx;
        if(Jaref_cidx > 0)
        {
            batch_constraint_force(env_id, cidx) = 0.f;
            batch_unquads(env_id, cidx) = 1;
        }
        else
        {
            batch_constraint_energy(env_id, cidx) = 0.5f * D_cidx * Jaref_cidx * Jaref_cidx;
        }
    }

    template<typename TDataType>
    __global__ void InertialEnergyKernel(
        DArray<int> is_converged,
        DArray<int> batch_nv,
        DevArr2D<Real> batch_Ma,
        DevArr2D<Real> batch_q_ex_force,
        DevArr2D<Real> batch_qacc,
        DevArr2D<Real> batch_q_ex_acc,
        DArray<Real> batch_energy,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs || is_converged[env_id])
            return;

        int dof_idx = threadIdx.x;
        if(dof_idx >= batch_nv[env_id])
            return;

        Real Ma = batch_Ma(env_id, dof_idx);
        Real q_ex_force = batch_q_ex_force(env_id, dof_idx);
        Real q_acc = batch_qacc(env_id, dof_idx);
        Real q_ex_acc = batch_q_ex_acc(env_id, dof_idx);
        atomicAdd(&batch_energy[env_id], 0.5f * (Ma - q_ex_force) * (q_acc - q_ex_acc));
    }

    template<typename TDataType>
    __global__ void ReduceConstraintEnergyKernel(
        DArray<int> is_converged,
        DArray<int> num_constraints,
        DevArr2D<Real> batch_constraint_energy,
        DArray<Real> batch_energy,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs || is_converged[env_id])
            return;

        __shared__ Real sh_sum[256];
        int tid = threadIdx.x;
        int nc = num_constraints[env_id];

        Real local_sum = 0.f;
        for(int cidx = tid; cidx < nc; cidx += blockDim.x)
            local_sum += batch_constraint_energy(env_id, cidx);

        sh_sum[tid] = local_sum;
        __syncthreads();

        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                sh_sum[tid] += sh_sum[tid + stride];
            __syncthreads();
        }

        if(tid == 0)
            batch_energy[env_id] = sh_sum[0];
    }

    template<typename TDataType>
    __global__ void BuildHessianKernel(
        DArray<int> is_converged,
        DArray<int> batch_nv,
        DArray<int> num_constraints,
        DevMat2D<Real> batch_J,
        DevArr2D<Real> batch_D,
        DevArr2D<int> batch_unquads,
        DevMat2D<Real> batch_qM,
        DevMat2D<Real> batch_H,
        int num_envs)
    {
        const int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        const int nv = batch_nv[env_id];
        const int nc = num_constraints[env_id];

        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.z * blockDim.x + threadIdx.x;

        if(row >= nv || col > row)
            return;

        Real sum = 0.f;
        for(int cidx = 0; cidx < nc; cidx++)
        {
            if(batch_unquads(env_id, cidx) != 0)
                continue;

            const Real d = batch_D(env_id, cidx);
            const Real jr = batch_J(env_id, cidx, row);
            const Real jc = batch_J(env_id, cidx, col);
            sum += jr * d * jc;
        }

        const Real h = batch_qM(env_id, row, col) + sum;
        batch_H.AtBlock(env_id, row, col) = h;
        if(col != row)
            batch_H.AtBlock(env_id, col, row) = h;
    }

    template<typename TDataType>
    __global__ void UpdateGradientKernel(
        DArray<int> is_converged,
        DArray<int> batch_nv,
        DArray<int> num_constraints,
        DevMat2D<Real> batch_J,
        DevArr2D<Real> batch_constraint_force,
        DevArr2D<Real> batch_Ma,
        DevArr2D<Real> batch_q_ex_force,
        DevArr2D<Real> batch_grad,
        int num_envs)
    {
        const int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        const int dof_idx = threadIdx.x;
        const int nv = batch_nv[env_id];
        if(dof_idx >= nv)
            return;

        const int nc = num_constraints[env_id];
        Real jt_f = 0.f;
        for(int cidx = 0; cidx < nc; cidx++)
        {
            const Real j = batch_J(env_id, cidx, dof_idx);
            jt_f += j * batch_constraint_force(env_id, cidx);
        }

        batch_grad(env_id, dof_idx) =
            -batch_Ma(env_id, dof_idx)
            +batch_q_ex_force(env_id, dof_idx)
            +jt_f;
    }

    template<typename TDataType>
    __global__ void ComputeScale(
        DArray<int> batch_nv,
        DevMat2D<Real> batch_qM,
        DArray<Real> batch_scale,
        int num_envs)
    {
        int env_id = threadIdx.x;
        if(env_id >= num_envs)
            return;

        int nv = batch_nv[env_id];
        Real sum_qM_diag = 0.f;
        for(int i = 0; i < nv; i++)
            sum_qM_diag += batch_qM(env_id, i, i);
        batch_scale[env_id] = 1.f / sum_qM_diag;
    }

    template<typename TDataType>
    __global__ void CheckConvergenceKernel(
        DArray<int> is_converged,
        DArray<Real> batch_scale,
        DevArr2D<Real> batch_grad,
        DArray<int> batch_nv,
        DArray<Real> batch_energy_ref,
        DArray<Real> batch_energy,
        int num_envs,
        Real impr,
        Real eps)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        const Real scale = batch_scale[env_id];
        Real grad_square_sum = 0.f;
        for(int i = 0; i < batch_nv[env_id]; i++)
            grad_square_sum += batch_grad(env_id, i) * batch_grad(env_id, i);
        grad_square_sum = scale * sqrt(grad_square_sum);

        Real improvement = scale * (batch_energy_ref[env_id] - batch_energy[env_id]);

        if(grad_square_sum < eps || improvement < impr)
            is_converged[env_id] = 1;
    }

    template<typename TDataType>
    __global__ void ForwardKinematicsKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        DArray2D<Quat<Real>> batch_quat,
        DArray2D<Mat3f> batch_rot,
        DevArr2D<int> parent_idx,
        DevArr2D<Vec3f> joint_axis_ref,
        DevArr2D<Vec3f> joint_anchor_ref,
        DevArr2D<int> joint_type,
        DevArr2D<Real> joint_qpos,
        DevArr2D<Real> joint_qpos_ref,
        DevArr2D<int> joint_qpos_offset,
        DArray2D<Vec3f> batch_pos,
        DevArr2D<Quat<Real>> joint_rel_quat,
        DevArr2D<Vec3f> joint_axis,
        DevArr2D<Vec3f> joint_rel_pos,
        DevArr2D<Vec3f> joint_anchor,
        DevArr2D<Vec3f> global_com_pos,
        DevArr2D<Vec3f> local_com_pos,
        DevArr2D<Quat<Real>> local_com_quat,
        DevArr2D<Mat3f> com_rot,
        int num_groups)
    {
        int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;

        for (int local_bid = 0; local_bid < body_count; ++local_bid)
        {
            const int bid = body_begin + local_bid;
            const int pidx = parent_idx(env_id, bid);
            if(pidx == -1)
            {
                batch_rot(env_id, bid) = batch_quat(env_id, bid).toMatrix3x3();
                global_com_pos(env_id, bid) = batch_rot(env_id, bid) * local_com_pos(env_id, bid) + batch_pos(env_id, bid);

                Quat<Real> quat_tmp = batch_quat(env_id, bid) * local_com_quat(env_id, bid);
                com_rot(env_id, bid) = quat_tmp.toMatrix3x3();
                continue;
            }

            const auto& parent_quat = batch_quat(env_id, pidx);
            const auto& parent_rot = batch_rot(env_id, pidx);
            const auto& local_axis = joint_axis_ref(env_id, bid);
            const auto& local_anchor = joint_anchor_ref(env_id, bid);
            const int jt = joint_type(env_id, bid);
            const int jqpos_start = joint_qpos_offset(env_id, bid);

            Quat<Real> xquat_p = parent_quat * joint_rel_quat(env_id, bid);
            joint_axis(env_id, bid) = RotateVector(local_axis, xquat_p);
            Vec3f xanchor = RotateVector(local_anchor, xquat_p);
            Vec3f xpos = parent_rot * joint_rel_pos(env_id, bid) + batch_pos(env_id, pidx);
            xanchor += xpos;
            joint_anchor(env_id, bid) = xanchor;

            if(jt == 2)     // Slide
            {
                batch_quat(env_id, bid) = xquat_p;
                batch_rot(env_id, bid) = xquat_p.toMatrix3x3();
                batch_pos(env_id, bid) = xpos + (joint_qpos(env_id, jqpos_start) - joint_qpos_ref(env_id, jqpos_start)) * joint_axis(env_id, bid);
            }
            else
            {
                Quat<Real> quat_local;
                if(jt == 1)  // Hinge
                {
                    quat_local = QuatFromAxisAngle(local_axis, joint_qpos(env_id, jqpos_start) - joint_qpos_ref(env_id, jqpos_start));
                }
                else         // Ball
                {
                    Quat<Real> ball_quat = Quat<Real>(
                        joint_qpos(env_id, jqpos_start),
                        joint_qpos(env_id, jqpos_start + 1),
                        joint_qpos(env_id, jqpos_start + 2),
                        joint_qpos(env_id, jqpos_start + 3));
                    ball_quat.normalize();
                    quat_local = ball_quat;
                }

                Quat<Real> xquat_c = xquat_p * quat_local;
                Quat<Real> xquat_c_norm = xquat_c;
                xquat_c_norm.normalize();
                batch_quat(env_id, bid) = xquat_c_norm;
                batch_rot(env_id, bid) = xquat_c.toMatrix3x3();
                xpos = RotateVector(local_anchor, xquat_c);
                batch_pos(env_id, bid) = xanchor - xpos;
            }

            global_com_pos(env_id, bid) = batch_rot(env_id, bid) * local_com_pos(env_id, bid) + batch_pos(env_id, bid);

            Quat<Real> quat_tmp = batch_quat(env_id, bid) * local_com_quat(env_id, bid);
            com_rot(env_id, bid) = quat_tmp.toMatrix3x3();

        }
    }

    template<typename TDataType>
    __global__ void SubtreeComKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<Real> batch_mass,
        DevArr2D<Real> subtree_mass,
        DevArr2D<Vec3f> batch_global_com_pos,
        DevArr2D<int> parent_idx,
        int num_envs,
        int num_groups)
    {
        int group_id = blockDim.x * blockIdx.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        if (env_id >= num_envs)
            return;

        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;

        for (int bidx = body_begin; bidx < body_begin + body_count; ++bidx)
            subtree_com(env_id, bidx) = batch_mass(env_id, bidx) * batch_global_com_pos(env_id, bidx);


        for(int bidx = body_begin + body_count - 1; bidx >= body_begin; --bidx)
        {
            const int pidx = parent_idx(env_id, bidx);
            if(pidx != -1)
                subtree_com(env_id, pidx) += subtree_com(env_id, bidx);
        }

        for(int bidx = body_begin; bidx < body_begin + body_count; ++bidx)
            subtree_com(env_id, bidx) /= subtree_mass(env_id, bidx);
    }

    template<typename TDataType>
    __global__ void ComputeCdofKernel(
        DArray<int> flatten_body_to_env,
        DArray<int> batch_bodies_offset,
        DevArr2D<Real> batch_cdof,
        DevArr2D<int> parent_idx,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Mat3f> batch_rot,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<int> q_offset,
        DevArr2D<int> root_idx,
        DevArr2D<Vec3f> joint_anchor,
        DevArr2D<int> joint_type,
        DevArr2D<Vec3f> joint_axis,
        DevArr2D<int> is_static,
        int num_envs,
        int total_bodies)
    {
        const int global_bid = blockDim.x * blockIdx.x + threadIdx.x;
        if(global_bid >= total_bodies)
            return;

        const int env_id = flatten_body_to_env[global_bid];
        if (env_id >= num_envs)
            return;

        const int bid = global_bid - batch_bodies_offset[env_id];

        const int pidx = parent_idx(env_id, bid);
        const Vec3f& pos = batch_pos(env_id, bid);
        const Mat3f rot = batch_rot(env_id, bid);
        const Vec3f& sub_com = subtree_com(env_id, bid);
        const int q_start = q_offset(env_id, bid);

        if(pidx != -1)
        {
            Vec3f offset = subtree_com(env_id, root_idx(env_id, bid)) - joint_anchor(env_id, bid);
            const int jt = joint_type(env_id, bid);
            const Vec3f& axis = joint_axis(env_id, bid);

            if(jt == 1)
            {
                Vec3f trans_part = cross(axis, offset);
                for(int i = 0; i < 3; i++)
                {
                    batch_cdof(env_id, q_start * 6 + i) = axis[i];
                    batch_cdof(env_id, q_start * 6 + 3 + i) = trans_part[i];
                }
            }
            else if(jt == 2)
            {
                for(int i = 0; i < 3; i++)
                {
                    batch_cdof(env_id, q_start * 6 + i) = 0.f;
                    batch_cdof(env_id, q_start * 6 + 3 + i) = axis[i];
                }
            }
            else
            {
                for(int i = 0; i < 3; i++)
                {
                    Vec3f rot_axis = rot.col(i);
                    Vec3f trans_part = cross(rot_axis, offset);
                    for(int j = 0; j < 3; j++)
                    {
                        batch_cdof(env_id, (q_start + i) * 6 + j) = rot_axis[j];
                        batch_cdof(env_id, (q_start + i) * 6 + 3 + j) = trans_part[j];
                    }
                }
            }
        }
        else
        {
            if(is_static(env_id, bid))
                return;

            for(int i = 0; i < 3; i++)
                batch_cdof(env_id, (q_start + i) * 6 + 3 + i) = 1.f;

            Vec3f offset = sub_com - pos;
            for(int i = 0; i < 3; i++)
            {
                Vec3f rot_axis = rot.col(i);
                Vec3f trans_part = cross(rot_axis, offset);
                for(int j = 0; j < 3; j++)
                {
                    batch_cdof(env_id, (q_start + i + 3) * 6 + 3 + j) = trans_part[j];
                    batch_cdof(env_id, (q_start + i + 3) * 6 + j) = rot_axis[j];
                }
            }
        }
    }

    __device__ void SubtreeComInertia(
        DevArr2D<Real>& com_inertia,
        const Vec3f& body_inertia,
        const Mat3f& rot_mat,
        const Vec3f& offset,
        Real mass,
        int env_id,
        int bid)
    {
        Real rot_mat_0 = rot_mat(0, 0);
        Real rot_mat_1 = rot_mat(0, 1);
        Real rot_mat_2 = rot_mat(0, 2);
        Real rot_mat_3 = rot_mat(1, 0);
        Real rot_mat_4 = rot_mat(1, 1);
        Real rot_mat_5 = rot_mat(1, 2);
        Real rot_mat_6 = rot_mat(2, 0);
        Real rot_mat_7 = rot_mat(2, 1);
        Real rot_mat_8 = rot_mat(2, 2);

        Real tmp_0 = rot_mat_0 * body_inertia.x;
        Real tmp_1 = rot_mat_3 * body_inertia.x;
        Real tmp_2 = rot_mat_6 * body_inertia.x;
        Real tmp_3 = rot_mat_1 * body_inertia.y;
        Real tmp_4 = rot_mat_4 * body_inertia.y;
        Real tmp_5 = rot_mat_7 * body_inertia.y;
        Real tmp_6 = rot_mat_2 * body_inertia.z;
        Real tmp_7 = rot_mat_5 * body_inertia.z;
        Real tmp_8 = rot_mat_8 * body_inertia.z;

        com_inertia(env_id, bid * 10)     = rot_mat_0 * tmp_0 + rot_mat_1 * tmp_3 + rot_mat_2 * tmp_6;
        com_inertia(env_id, bid * 10 + 1) = rot_mat_3 * tmp_1 + rot_mat_4 * tmp_4 + rot_mat_5 * tmp_7;
        com_inertia(env_id, bid * 10 + 2) = rot_mat_6 * tmp_2 + rot_mat_7 * tmp_5 + rot_mat_8 * tmp_8;
        com_inertia(env_id, bid * 10 + 3) = rot_mat_0 * tmp_1 + rot_mat_1 * tmp_4 + rot_mat_2 * tmp_7;
        com_inertia(env_id, bid * 10 + 4) = rot_mat_0 * tmp_2 + rot_mat_1 * tmp_5 + rot_mat_2 * tmp_8;
        com_inertia(env_id, bid * 10 + 5) = rot_mat_3 * tmp_2 + rot_mat_4 * tmp_5 + rot_mat_5 * tmp_8;

        com_inertia(env_id, bid * 10) += mass * (offset.y * offset.y + offset.z * offset.z);
        com_inertia(env_id, bid * 10 + 1) += mass * (offset.x * offset.x + offset.z * offset.z);
        com_inertia(env_id, bid * 10 + 2) += mass * (offset.x * offset.x + offset.y * offset.y);
        com_inertia(env_id, bid * 10 + 3) -= mass * offset.x * offset.y;
        com_inertia(env_id, bid * 10 + 4) -= mass * offset.x * offset.z;
        com_inertia(env_id, bid * 10 + 5) -= mass * offset.y * offset.z;
        com_inertia(env_id, bid * 10 + 6) = mass * offset.x;
        com_inertia(env_id, bid * 10 + 7) = mass * offset.y;
        com_inertia(env_id, bid * 10 + 8) = mass * offset.z;
        com_inertia(env_id, bid * 10 + 9) = mass;
    }

    __global__ void SubtreeInertialKernel(
        DArray<int> flatten_body_to_env,
        DArray<int> batch_bodies_offset,
        DevArr2D<int> root_idx,
        DevArr2D<Vec3f> batch_global_com_pos,
        DevArr2D<Vec3f> subtree_com,
        DevArr2D<Real> subtree_inertia,
        DevArr2D<Vec3f> batch_inertia,
        DevArr2D<Mat3f> batch_com_rot,
        DevArr2D<Real> batch_mass,
        DevArr2D<Real> batch_crb,
        int num_envs,
        int total_bodies)
    {
        const int global_bid = blockDim.x * blockIdx.x + threadIdx.x;
        if(global_bid >= total_bodies)
            return;

        const int env_id = flatten_body_to_env[global_bid];
        if(env_id >= num_envs)
            return;

        const int bid = global_bid - batch_bodies_offset[env_id];

        const int ridx = root_idx(env_id, bid);
        Vec3f offset = batch_global_com_pos(env_id, bid) - subtree_com(env_id, ridx);
        const Vec3f& body_inertia = batch_inertia(env_id, bid);
        const auto& rot_mat = batch_com_rot(env_id, bid);
        const Real mass = batch_mass(env_id, bid);

        SubtreeComInertia(subtree_inertia, body_inertia, rot_mat, offset, mass, env_id, bid);
        for(int i = 0; i < 10; i++)
            batch_crb(env_id, bid * 10 + i) = subtree_inertia(env_id, bid * 10 + i);
    }

    __global__ void AccumulateSubtreeInertialKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        DevArr2D<int> parent_idx,
        DevArr2D<Real> batch_crb,
        DevArr2D<Real> subtree_inertia,
        int num_envs,
        int num_groups)
    {
        const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        if (env_id >= num_envs)
            return;

        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;

        for(int bid = body_begin + body_count - 1; bid >= body_begin; bid--)
        {
            const int pidx = parent_idx(env_id, bid);
            if(pidx != -1)
                for(int i = 0; i < 10; i++)
                    batch_crb(env_id, pidx * 10 + i) += batch_crb(env_id, bid * 10 + i);
        }

        // for(int bid = 0; bid < batch_bodies[env_id]; bid++)
        // {
        //     printf("Env %d, Body %d, Composite Rigid Body Inertia:\n", env_id, bid);
        //     printf("ComInertial: \n");
        //     for(int i = 0; i < 10; i++)
        //         printf("%f\t", subtree_inertia(env_id, bid * 10 + i));
        //     printf("\n");
        //     printf("CRB: \n");
        //     for(int i = 0; i < 10; i++)
        //         printf("%f\t", batch_crb(env_id, bid * 10 + i));
        //     printf("\n");
        // }
    }

    __device__ void InertiaMultiVec(const DevArr2D<Real>& inertia, const Real* vec, Real* res, int env_id, int bid)
    {
        const Real& inertia_0 = inertia(env_id, bid * 10);
        const Real& inertia_1 = inertia(env_id, bid * 10 + 1);
        const Real& inertia_2 = inertia(env_id, bid * 10 + 2);
        const Real& inertia_3 = inertia(env_id, bid * 10 + 3);
        const Real& inertia_4 = inertia(env_id, bid * 10 + 4);
        const Real& inertia_5 = inertia(env_id, bid * 10 + 5);
        const Real& inertia_6 = inertia(env_id, bid * 10 + 6);
        const Real& inertia_7 = inertia(env_id, bid * 10 + 7);
        const Real& inertia_8 = inertia(env_id, bid * 10 + 8);
        const Real& inertia_9 = inertia(env_id, bid * 10 + 9);

        res[0] = inertia_0 * vec[0] + inertia_3 * vec[1] + inertia_4 * vec[2] - inertia_8 * vec[4] + inertia_7 * vec[5];
        res[1] = inertia_3 * vec[0] + inertia_1 * vec[1] + inertia_5 * vec[2] + inertia_8 * vec[3] - inertia_6 * vec[5];
        res[2] = inertia_4 * vec[0] + inertia_5 * vec[1] + inertia_2 * vec[2] - inertia_7 * vec[3] + inertia_6 * vec[4];
        res[3] = inertia_8 * vec[1] - inertia_7 * vec[2] + inertia_9 * vec[3];
        res[4] = inertia_6 * vec[2] - inertia_8 * vec[0] + inertia_9 * vec[4];
        res[5] = inertia_7 * vec[0] - inertia_6 * vec[1] + inertia_9 * vec[5];
    }

    template<typename TDataType>
    __global__ void UpdateGeneralizedInertialMatrixKernel(
        DArray<Pair<int, int>> flatten_q_to_env_body,
        DevMat2D<Real> batch_qM,
        DevArr2D<int> batch_nv_offset,
        DArray<int> batch_nv,
        DevArr2D<int> is_isolated,
        DevArr2D<int> is_static,
        DevArr2D<int> parent_idx,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<Real> batch_cdof,
        DevArr2D<Real> batch_crb,
        DevArr2D<Vec3f> batch_inertia,
        DevArr2D<Real> batch_mass,
        int num_envs,
        int total_nv)
    {
        const int global_qidx = blockIdx.x * blockDim.x + threadIdx.x;
        if(global_qidx >= total_nv)
            return;

        const Pair<int, int> q_info = flatten_q_to_env_body[global_qidx];
        const int env_id = q_info.first;
        if(env_id >= num_envs)
            return;

        const int bid = q_info.second;
        const int nv = batch_nv[env_id];
        const int body_global_q_start = batch_nv_offset(env_id, bid);
        const int q_start = q_offset(env_id, bid);
        const int q_num = q_lengths(env_id, bid);
        const int qidx = q_start + (global_qidx - body_global_q_start);

        if(qidx < q_start || qidx >= q_start + q_num)
            return;

        if(is_static(env_id, bid))
            return;

        const int isolated = is_isolated(env_id, bid);
        if(!isolated)
        {
            Real tmp_dof[6];
            Real Icdof[6];
            for(int i = 0; i < 6; i++)
                tmp_dof[i] = batch_cdof(env_id, qidx * 6 + i);

            InertiaMultiVec(batch_crb, tmp_dof, Icdof, env_id, bid);

            int i = qidx;
            int j = bid;
            while(j != -1)
            {
                const int qidx_j = q_offset(env_id, j);
                for(int k = i; k >= qidx_j; k--)
                {
                    Real val = 0.f;
                    for(int n = 0; n < 6; n++)
                        val += batch_cdof(env_id, k * 6 + n) * Icdof[n];

                    batch_qM(env_id, qidx, k) = val;
                }

                j = parent_idx(env_id, j);
                if(j != -1)
                    i = q_offset(env_id, j) + q_lengths(env_id, j) - 1;
            }
        }
        else
        {
            const int local_qidx = qidx - q_start;
            if(local_qidx < 3)
                batch_qM(env_id, qidx, qidx) = batch_mass(env_id, bid);
            else if(local_qidx < q_num)
            {
                const Vec3f& inertia = batch_inertia(env_id, bid);
                if(local_qidx == 3)
                    batch_qM(env_id, qidx, qidx) = inertia.x;
                else if(local_qidx == 4)
                    batch_qM(env_id, qidx, qidx) = inertia.y;
                else if(local_qidx == 5)
                    batch_qM(env_id, qidx, qidx) = inertia.z;
            }
        }
    }

    // mirror the upper triangle to the lower triangle to ensure symmetry
    __global__ void FillGeneralizedInertialMatrixSymmetryKernel(
        DevMat2D<Real> batch_qM,
        DArray<int> batch_nv,
        int num_envs)
    {
        const int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int nv = batch_nv[env_id];
        for(int row = 0; row < nv; row++)
            for(int col = 0; col < row; col++)
                batch_qM(env_id, col, row) = batch_qM(env_id, row, col);
    }

    template<typename T>
    __device__ void ComputeComVel(const DevArr2D<T>& cdof, const DevArr2D<T>& qvel, DevArr2D<T>& com_vel,
        int env_id, int bid, int q_start, int offset)
    {
        for(int r = 0; r < 6; r++)
        {
            T sum = T(0);
            for(int c = 0; c < 3; c++)
                sum += cdof(env_id, (q_start + offset + c) * 6 + r) * qvel(env_id, q_start + offset + c);

            com_vel(env_id, bid * 6 + r) += sum;
        }
    }

    template<typename T>
    __device__ void ComputeCVelCross(const DevArr2D<T>& cdof, const DevArr2D<T>& com_vel, DevArr2D<T>& cdof_dot,
        int env_id, int bid, int q_start, int offset)
    {
        const Real cvel_0 = com_vel(env_id, bid * 6);
        const Real cvel_1 = com_vel(env_id, bid * 6 + 1);
        const Real cvel_2 = com_vel(env_id, bid * 6 + 2);
        const Real cvel_3 = com_vel(env_id, bid * 6 + 3);
        const Real cvel_4 = com_vel(env_id, bid * 6 + 4);
        const Real cvel_5 = com_vel(env_id, bid * 6 + 5);

        int idx = (q_start + offset) * 6;
        const Real cdof_0 = cdof(env_id, idx);
        const Real cdof_1 = cdof(env_id, idx + 1);
        const Real cdof_2 = cdof(env_id, idx + 2);
        const Real cdof_3 = cdof(env_id, idx + 3);
        const Real cdof_4 = cdof(env_id, idx + 4);
        const Real cdof_5 = cdof(env_id, idx + 5);

        cdof_dot(env_id, idx) = -cvel_2 * cdof_1 + cvel_1 * cdof_2;
        cdof_dot(env_id, idx + 1) = cvel_2 * cdof_0 - cvel_0 * cdof_2;
        cdof_dot(env_id, idx + 2) = -cvel_1 * cdof_0 + cvel_0 * cdof_1;
        cdof_dot(env_id, idx + 3) = -cvel_2 * cdof_4 + cvel_1 * cdof_5 - cvel_5 * cdof_1 + cvel_4 * cdof_2;
        cdof_dot(env_id, idx + 4) = cvel_2 * cdof_3 - cvel_0 * cdof_5 + cvel_5 * cdof_0 - cvel_3 * cdof_2;
        cdof_dot(env_id, idx + 5) = -cvel_1 * cdof_3 + cvel_0 * cdof_4 - cvel_4 * cdof_0 + cvel_3 * cdof_1;
    }

    template<typename TDataType>
    __global__ void ComputeComVelKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        DevArr2D<Real> batch_cdof,
        DevArr2D<Real> batch_qvel,
        DevArr2D<Real> subtree_com_vel,
        DevArr2D<Real> batch_cdof_dot,
        DevArr2D<int> parent_idx,
        DevArr2D<int> q_offset,
        DevArr2D<int> joint_type,
        DevArr2D<int> is_static,
        int num_envs,
        int num_groups)
    {
        const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        if(env_id >= num_envs)
            return;

        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;

        for(int bid = body_begin; bid < body_begin + body_count; bid++)
        {
            const int pidx = parent_idx(env_id, bid);
            const int q_start = q_offset(env_id, bid);
            if(pidx != -1)
            {
                const int jt = joint_type(env_id, bid);
                if(jt < 3)
                {
                    for(int i = 0; i < 6; i++)
                        subtree_com_vel(env_id, bid * 6 + i) = subtree_com_vel(env_id, pidx * 6 + i)
                                                         + batch_qvel(env_id, q_start) * batch_cdof(env_id, q_start * 6 + i);
                    ComputeCVelCross(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start, 0);
                }
                else
                {
                    for(int i = 0; i < 6; i++)
                        subtree_com_vel(env_id, bid * 6 + i) = subtree_com_vel(env_id, pidx * 6 + i);

                    for(int i = 0; i < 3; i++)
                        ComputeCVelCross(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start, i);

                    ComputeComVel(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start, 0);
                }
            }
            else
            {
                if(is_static(env_id, bid))
                    continue;

                ComputeComVel(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start, 0);
                for(int i = 0; i < 3; i++)
                    ComputeCVelCross(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start, 3 + i);
                ComputeComVel(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start, 3);
            }
        }
    }

    template<typename T>
    __device__ void ComputeCACC(const DevArr2D<T>& cdof_dot, const DevArr2D<T>& qvel, DevArr2D<T>& cacc,
        int env_id, int bid, int q_start, int q_length)
    {
        for(int r = 0; r < 6; r++)
        {
            T sum = T(0);
            for(int c = 0; c < q_length; c++)
                sum += cdof_dot(env_id, (q_start + c) * 6 + r) * qvel(env_id, q_start + c);
            cacc(env_id, bid * 6 + r) += sum;
        }
    }

    template<typename T>
    __device__ void ComputeCVelCrossDual(const DevArr2D<T>& com_vel, const Real* Ivel, Real* res, int env_id, int bid)
    {
        const Real cvel_0 = com_vel(env_id, bid * 6);
        const Real cvel_1 = com_vel(env_id, bid * 6 + 1);
        const Real cvel_2 = com_vel(env_id, bid * 6 + 2);
        const Real cvel_3 = com_vel(env_id, bid * 6 + 3);
        const Real cvel_4 = com_vel(env_id, bid * 6 + 4);
        const Real cvel_5 = com_vel(env_id, bid * 6 + 5);

        const Real vec_0 = Ivel[0];
        const Real vec_1 = Ivel[1];
        const Real vec_2 = Ivel[2];
        const Real vec_3 = Ivel[3];
        const Real vec_4 = Ivel[4];
        const Real vec_5 = Ivel[5];

        res[0] = -cvel_2 * vec_1 + cvel_1 * vec_2 - cvel_5 * vec_4 + cvel_4 * vec_5;
        res[1] = cvel_2 * vec_0 - cvel_0 * vec_2 + cvel_5 * vec_3 - cvel_3 * vec_5;
        res[2] = -cvel_1 * vec_0 + cvel_0 * vec_1 - cvel_4 * vec_3 + cvel_3 * vec_4;
        res[3] = -cvel_2 * vec_4 + cvel_1 * vec_5;
        res[4] = cvel_2 * vec_3 - cvel_0 * vec_5;
        res[5] = -cvel_1 * vec_3 + cvel_0 * vec_4;
    }

    template<typename TDataType>
    __global__ void RNECaccKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        const DArray<Vec3f> gravities,
        DevArr2D<Real> batch_cacc,
        DevArr2D<Real> batch_cdof_dot,
        DevArr2D<Real> batch_qvel,
        DevArr2D<int> parent_idx,
        DevArr2D<int> is_static,
        DevArr2D<int> joint_type,
        DevArr2D<int> q_offset,
        int num_envs,
        int num_groups)
    {
        const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        if(env_id >= num_envs)
            return;

        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;
        const Vec3f gravity = gravities[env_id];

        for(int bid = body_begin; bid < body_begin + body_count; bid++)
        {
            const int pidx = parent_idx(env_id, bid);
            const int q_start = q_offset(env_id, bid);

            if(pidx == -1)
            {
                for(int i = 0; i < 3; i++)
                    batch_cacc(env_id, bid * 6 + 3 + i) = -gravity[i];

                if(!is_static(env_id, bid))
                    ComputeCACC(batch_cdof_dot, batch_qvel, batch_cacc, env_id, bid, q_start, 6);
            }
            else
            {
                const int jt = joint_type(env_id, bid);
                for(int i = 0; i < 6; i++)
                    batch_cacc(env_id, bid * 6 + i) = batch_cacc(env_id, pidx * 6 + i);

                if(jt < 3)
                {
                    for(int i = 0; i < 6; i++)
                        batch_cacc(env_id, bid * 6 + i) += batch_qvel(env_id, q_start) * batch_cdof_dot(env_id, q_start * 6 + i);
                }
                else
                {
                    ComputeCACC(batch_cdof_dot, batch_qvel, batch_cacc, env_id, bid, q_start, 3);
                }
            }
        }
    }

    template<typename TDataType>
    __global__ void RNEForceKernel(
        DArray<int> flatten_body_to_env,
        DArray<int> batch_body_offset,
        DevArr2D<Real> batch_cacc,
        DevArr2D<Real> batch_cforce,
        DevArr2D<Real> subtree_inertia,
        DevArr2D<Real> subtree_com_vel,
        int num_envs,
        int total_bodies)
    {
        const int global_bid = blockIdx.x * blockDim.x + threadIdx.x;
        if(global_bid >= total_bodies)
            return;

        const int env_id = flatten_body_to_env[global_bid];
        if(env_id >= num_envs)
            return;

        const int bid = global_bid - batch_body_offset[env_id];

        Real Iacc[6];
        Real Ivel[6];
        Real vec6_buffer[6];

        for(int i = 0; i < 6; i++)
            vec6_buffer[i] = batch_cacc(env_id, bid * 6 + i);
        InertiaMultiVec(subtree_inertia, vec6_buffer, Iacc, env_id, bid);

        for(int i = 0; i < 6; i++)
            vec6_buffer[i] = subtree_com_vel(env_id, bid * 6 + i);
        InertiaMultiVec(subtree_inertia, vec6_buffer, Ivel, env_id, bid);

        ComputeCVelCrossDual(subtree_com_vel, Ivel, vec6_buffer, env_id, bid);
        for(int i = 0; i < 6; i++)
            batch_cforce(env_id, bid * 6 + i) = Iacc[i] + vec6_buffer[i];
    }

    template<typename TDataType>
    __global__ void RNEAccumKernel(
        DevArr2D<Pair<int, int>> batch_groups,
        DArray<int> flatten_group_to_env,
        DevArr2D<int> parent_idx,
        DevArr2D<Real> batch_cforce,
        int num_envs,
        int num_groups)
    {
        const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(group_id >= num_groups)
            return;

        const int env_id = flatten_group_to_env[group_id];
        if(env_id >= num_envs)
            return;

        const int local_gid = group_id - batch_groups.BlockOffset(env_id);
        const Pair<int, int> group = batch_groups(env_id, local_gid);
        const int body_begin = group.first;
        const int body_count = group.second;

        for(int bid = body_begin + body_count - 1; bid >= body_begin; bid--)
        {
            const int pidx = parent_idx(env_id, bid);
            if(pidx == -1)
                continue;

            for(int i = 0; i < 6; i++)
                batch_cforce(env_id, pidx * 6 + i) += batch_cforce(env_id, bid * 6 + i);
        }
    }

    template<typename TDataType>
    __global__ void RNEProjKernel(
        DArray<Pair<int, int>> flatten_q_to_env_body,
        DevArr2D<int> batch_nv_offset,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<Real> batch_cdof,
        DevArr2D<Real> batch_cforce,
        DevArr2D<Real> batch_q_inner_force,
        int num_envs,
        int total_nv)
    {
        const int global_qidx = blockIdx.x * blockDim.x + threadIdx.x;
        if(global_qidx >= total_nv)
            return;

        const Pair<int, int> q_info = flatten_q_to_env_body[global_qidx];
        const int env_id = q_info.first;
        if(env_id >= num_envs)
            return;

        const int bid = q_info.second;
        const int body_global_q_start = batch_nv_offset(env_id, bid);
        const int q_start = q_offset(env_id, bid);
        const int q_num = q_lengths(env_id, bid);
        const int qidx = q_start + (global_qidx - body_global_q_start);

        if(qidx < q_start || qidx >= q_start + q_num)
            return;

        Real sum = 0.f;
        for(int i = 0; i < 6; i++)
            sum += batch_cdof(env_id, qidx * 6 + i) * batch_cforce(env_id, bid * 6 + i);
        batch_q_inner_force(env_id, qidx) = sum;
    }

    template<typename TDataType>
    __global__ void ComputeRNEKernel(
        DArray<int> batch_bodies,
        const DArray<Vec3f> gravities,
        DevArr2D<Real> batch_cacc,
        DevArr2D<Real> batch_cforce,
        DevArr2D<Real> batch_q_inner_force,
        DevArr2D<Real> batch_cdof,
        DevArr2D<Real> batch_cdof_dot,
        DevArr2D<Real> batch_qvel,
        DevArr2D<int> q_offset,
        DevArr2D<Real> subtree_inertia,
        DevArr2D<Real> subtree_com_vel,
        DevArr2D<int> parent_idx,
        DevArr2D<int> is_static,
        DevArr2D<int> joint_type,
        DevArr2D<int> q_lengths,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        const Vec3f gravity = gravities[env_id];
        Real Iacc[6];
        Real Ivel[6];
        Real vec6_buffer[6];

        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int pidx = parent_idx(env_id, bid);
            if(pidx == -1)
            {
                for(int i = 0; i < 3; i++)
                    batch_cacc(env_id, bid * 6 + 3 + i) = -gravity[i];

                if(!is_static(env_id, bid))
                    ComputeCACC(batch_cdof_dot, batch_qvel, batch_cacc, env_id, bid, q_offset(env_id, bid), 6);
            }
            else
            {
                const int jt = joint_type(env_id, bid);
                for(int i = 0; i < 6; i++)
                    batch_cacc(env_id, bid * 6 + i) = batch_cacc(env_id, pidx * 6 + i);

                if(jt < 3)
                {
                    for(int i = 0; i < 6; i++)
                        batch_cacc(env_id, bid * 6 + i) += batch_qvel(env_id, q_offset(env_id, bid)) * batch_cdof_dot(env_id, q_offset(env_id, bid) * 6 + i);
                }
                else
                {
                    ComputeCACC(batch_cdof_dot, batch_qvel, batch_cacc, env_id, bid, q_offset(env_id, bid), 3);
                }
            }

            for(int i = 0; i < 6; i++)
                vec6_buffer[i] = batch_cacc(env_id, bid * 6 + i);
            InertiaMultiVec(subtree_inertia, vec6_buffer, Iacc, env_id, bid);

            for(int i = 0; i < 6; i++)
                vec6_buffer[i] = subtree_com_vel(env_id, bid * 6 + i);
            InertiaMultiVec(subtree_inertia, vec6_buffer, Ivel, env_id, bid);

            ComputeCVelCrossDual(subtree_com_vel, Ivel, vec6_buffer, env_id, bid);
            for(int i = 0; i < 6; i++)
                batch_cforce(env_id, bid * 6 + i) = Iacc[i] + vec6_buffer[i];
        }

        for(int bid = num_bodies - 1; bid >= 0; bid--)
        {
            const int pidx = parent_idx(env_id, bid);
            if(pidx == -1)
                continue;

            for(int i = 0; i < 6; i++)
                batch_cforce(env_id, pidx * 6 + i) += batch_cforce(env_id, bid * 6 + i);
        }

        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int pidx = parent_idx(env_id, bid);
            if(is_static(env_id, bid))
                continue;

            const int jt = joint_type(env_id, bid);
            const int q_start = q_offset(env_id, bid);
            if(pidx == -1)
            {
                for(int i = 0; i < 6; i++)
                {
                    Real sum = 0.f;
                    for(int j = 0; j < 6; j++)
                        sum += batch_cdof(env_id, (q_start + i) * 6 + j) * batch_cforce(env_id, bid * 6 + j);
                    batch_q_inner_force(env_id, q_start + i) = sum;
                }
            }
            else
            {
                if(jt < 3)
                {
                    Real sum = 0.f;
                    for(int i = 0; i < 6; i++)
                        sum += batch_cdof(env_id, q_start * 6 + i) * batch_cforce(env_id, bid * 6 + i);
                    batch_q_inner_force(env_id, q_start) = sum;
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                    {
                        Real sum = 0.f;
                        for(int j = 0; j < 6; j++)
                            sum += batch_cdof(env_id, (q_start + i) * 6 + j) * batch_cforce(env_id, bid * 6 + j);
                        batch_q_inner_force(env_id, q_start + i) = sum;
                    }
                }
            }

            if(env_id == 0)
            {
                printf("env %d, body %d, q_inner_force: ", env_id, bid);
                for(int i = 0; i < q_lengths(env_id, bid); i++)
                    printf("%f ", batch_q_inner_force(env_id, q_start + i));
                printf("\n");
            }
            
        }
    }

    template<typename TDataType>
    __global__ void UpdateJointPoseKernel(
        DArray<int> batch_bodies,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Quat<Real>> batch_quat,
        DArray2D<Mat3f> batch_rot,
        DevArr2D<Real> joint_qpos,
        DevArr2D<int> parent_idx,
        DevArr2D<int> is_static,
        DevArr2D<int> qpos_offset,
        DevArr2D<int> joint_qpos_offset,
        DevArr2D<Real> batch_qpos,
        DevArr2D<int> joint_type,
        int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        int bid = threadIdx.x;
        if(bid >= num_bodies)
            return;

        const int parent = parent_idx(env_id, bid);
        const int is_static_body = is_static(env_id, bid);
        const int qpos_start = qpos_offset(env_id, bid);
        const int joint_qpos_start = joint_qpos_offset(env_id, bid);

        if(parent != -1)
        {
            const int jt = joint_type(env_id, bid);
            if(jt < 3)     // Hinge or Slide
                joint_qpos(env_id, joint_qpos_start) = batch_qpos(env_id, qpos_start);
            else           // Ball
            {
                for(int i = 0; i < 4; i++)
                    joint_qpos(env_id, joint_qpos_start + i) = batch_qpos(env_id, qpos_start + i);
            }
        }
        else if(!is_static_body)
        {
            for(int i = 0; i < 3; i++)
                batch_pos(env_id, bid)[i] = batch_qpos(env_id, qpos_start + i);

            batch_quat(env_id, bid).x = batch_qpos(env_id, qpos_start + 3);
            batch_quat(env_id, bid).y = batch_qpos(env_id, qpos_start + 4);
            batch_quat(env_id, bid).z = batch_qpos(env_id, qpos_start + 5);
            batch_quat(env_id, bid).w = batch_qpos(env_id, qpos_start + 6);
            batch_rot(env_id, bid) = batch_quat(env_id, bid).toMatrix3x3();
        }

        printf("Env %d, Body %d, Position: (%f, %f, %f), joint_qpos: %f\n",
            env_id, bid, batch_pos(env_id, bid).x, batch_pos(env_id, bid).y, batch_pos(env_id, bid).z,
            joint_qpos(env_id, joint_qpos_start));
    }

    __global__ void PrintTestInfos(
        DArray<int> batch_bodies,
        DevArr2D<int> q_offset,
        DevArr2D<int> q_lengths,
        DevArr2D<Real> batch_cdof,
        DevArr2D<int> is_static,
        DevArr2D<int> is_isolated,
        DArray2D<int> shape_type,
        DevArr2D<int> parent_idx,
        DevArr2D<int> joint_type,
        int num_envs, int target_env_id)
    {
        int env_id = threadIdx.x;

        if(env_id != target_env_id)
            return;
        const int num_bodies = batch_bodies[env_id];
        for(int bid = 0; bid < num_bodies; bid++)
        {
            printf("Env %d, Body %d, is_static: %d, is_isolated: %d, shape_type: %d, parent_idx: %d, joint_type: %d, q_offset: %d, q_length: %d\n",
                env_id, bid, is_static(env_id, bid), is_isolated(env_id, bid), shape_type(env_id, bid),
                parent_idx(env_id, bid), joint_type(env_id, bid), q_offset(env_id, bid), q_lengths(env_id, bid));

            printf("!!cdof:\n");
            for(int i = 0; i < q_lengths(env_id, bid); i++)
            {
                for(int j = 0; j < 6; j++)
                    printf("  dof %d: %f ", i, batch_cdof(env_id, (q_offset(env_id, bid) + i) * 6 + j));
                printf("\n\n");
            }
        }
    }

    __global__ void BatchAlphaInertiaEnergyKernel(
        DevArr2D<Real> batch_Ma,
        DevArr2D<Real> batch_Ma_increm,
        DevArr2D<Real> batch_q_ex_force,
        DevArr2D<Real> batch_qacc,
        DevArr2D<Real> batch_dx,
        DevArr2D<Real> batch_q_ex_acc,
        DevArr2D<Real> batch_alpha_energies,
        DArray<Real> alphas,
        DArray<int> batch_nv, DArray<int> is_converged, int num_envs, int num_alphas)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        int nv = batch_nv[env_id];
        int alpha_idx = threadIdx.x / nv;
        int dof_idx = threadIdx.x % nv;

        if(alpha_idx >= num_alphas)
            return;

        const Real alpha = alphas[alpha_idx];


        Real Ma = batch_Ma(env_id, dof_idx) + alpha * batch_Ma_increm(env_id, dof_idx);
        Real q_ex_force = batch_q_ex_force(env_id, dof_idx);
        Real q_acc = batch_qacc(env_id, dof_idx) + alpha * batch_dx(env_id, dof_idx);
        Real q_ex_acc = batch_q_ex_acc(env_id, dof_idx);
        atomicAdd(&batch_alpha_energies(env_id, alpha_idx), 0.5f * (Ma - q_ex_force) * (q_acc - q_ex_acc));
    }

    __global__ void BatchAlphaAnchorEnergyKernel(
        DevArr2D<Real> batch_Jaref,
        DevArr2D<Real> batch_Jaref_increm,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_alpha_energies,
        DArray<Real> alphas,
        DArray<Vec4i> num_each_constraint,
        DArray<int> is_converged, int num_envs, int num_alphas)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        int num_anchor = num_each_constraint[env_id][0];
        if(num_anchor == 0)
            return;
        int alpha_idx = threadIdx.x / num_anchor;
        if(alpha_idx >= num_alphas)
            return;

        
        int cidx = threadIdx.x % num_anchor;

        const Real D = batch_D(env_id, cidx);
        const Real Jaref = batch_Jaref(env_id, cidx) + alphas[alpha_idx] * batch_Jaref_increm(env_id, cidx);
        atomicAdd(&batch_alpha_energies(env_id, alpha_idx), 0.5f * D * Jaref * Jaref);

    }

    __global__ void BatchAlphaFrictionEnergyKernel(
        BatchFrictionLossConstraints friction_loss_constraints,
        DevArr2D<Real> batch_Jaref,
        DevArr2D<Real> batch_Jaref_increm,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_alpha_energies,
        DArray<Real> alphas,
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DArray<int> is_converged, int num_envs, int num_alphas)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;
        int num_friction = num_each_constraint[env_id][1];
        if(num_friction == 0)
            return;
        int alpha_idx = threadIdx.x / num_friction;

        if(alpha_idx >= num_alphas)
            return;

        
        int cidx = threadIdx.x % num_friction;

        int c_offset = constraint_offset[env_id][1];
        const int ridx = c_offset + cidx;
        Real D = batch_D(env_id, ridx);
        Real Jaref = batch_Jaref(env_id, ridx) + alphas[alpha_idx] * batch_Jaref_increm(env_id, ridx);
        Real dof_frictionloss = friction_loss_constraints.dof_frictionloss(env_id, cidx);

        Real R_dof_fl = dof_frictionloss / D;

        if(Jaref <= -R_dof_fl)
            atomicAdd(&batch_alpha_energies(env_id, alpha_idx), -0.5f * R_dof_fl * dof_frictionloss - dof_frictionloss * Jaref);
        else if(Jaref >= R_dof_fl)
            atomicAdd(&batch_alpha_energies(env_id, alpha_idx), -0.5f * R_dof_fl * dof_frictionloss + dof_frictionloss * Jaref);
        else
            atomicAdd(&batch_alpha_energies(env_id, alpha_idx), 0.5f * D * Jaref * Jaref);
    }

    __global__ void BatchAlphaContactAndJointLimitEnergyKernel(
        DevArr2D<Real> batch_Jaref,
        DevArr2D<Real> batch_Jaref_increm,
        DevArr2D<Real> batch_D,
        DevArr2D<Real> batch_alpha_energies,
        DArray<Real> alphas,
        DArray<Vec4i> num_each_constraint,
        DArray<Vec4i> constraint_offset,
        DArray<int> is_converged, int num_envs, int num_alphas)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        int num_contact = num_each_constraint[env_id][2];
        int num_joint_limit = num_each_constraint[env_id][3];
        int nc = num_contact + num_joint_limit;
        if(nc == 0)
            return;
        int alpha_idx = threadIdx.x / nc;

        if(alpha_idx >= num_alphas)
            return;

        
        int cidx = constraint_offset[env_id][2] + threadIdx.x % nc;

        Real Jaref = batch_Jaref(env_id, cidx) + alphas[alpha_idx] * batch_Jaref_increm(env_id, cidx);
        Real D = batch_D(env_id, cidx);
        if(Jaref < 0)
            atomicAdd(&batch_alpha_energies(env_id, alpha_idx), 0.5f * D * Jaref * Jaref);

    }

    __global__ void ChooseAlphaKernel(
        DevArr2D<Real> batch_alpha_energies,
        DArray<Real> alphas,
        DArray<Real> alphas_cands,
        DArray<int> is_converged,
        int num_envs, int num_alphas)
    {
        int env_id = threadIdx.x;
        if(env_id >= num_envs)
            return;
        if(is_converged[env_id])
            return;

        Real min_energy = batch_alpha_energies(env_id, 0);
        int min_idx = 0;
        for(int i = 1; i < num_alphas; i++)
        {
            const Real& energy = batch_alpha_energies(env_id, i);
            if(energy < min_energy)
            {
                min_energy = energy;
                min_idx = i;
            }
        }

        alphas[env_id] = alphas_cands[min_idx];
        if(alphas[env_id] == 0.f)
            is_converged[env_id] = 1;
    }

    // __global__ void 
}

namespace dyno
{
    template<typename TDataType>
    void MujocoSolver<TDataType>::Init()
    {
        spdlog::info("[MujocoSolver Solver] Starting initialization.");

        const int max_joint_qpos = 128;

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        const int num_envs = env_infos->num_envs;
        const int num_max_constraints = env_infos->max_constraints;
        rigid_body_system->max_bodies = GetMaxValue(rigid_body_system->batch_bodies, num_envs);
        const int max_bodies = rigid_body_system->max_bodies;
        const int max_nv = max_bodies * 6;


        std::vector<int> num_bodies_host(num_envs);
        cudaMemcpy(num_bodies_host.data(), rigid_body_system->batch_bodies.begin(), num_envs * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> num_bodies6_host(num_envs);
        std::transform(num_bodies_host.begin(), num_bodies_host.end(), num_bodies6_host.begin(), [](int num){ return num * 6; });
        std::vector<int> num_bodies10_host(num_envs);
        std::transform(num_bodies_host.begin(), num_bodies_host.end(), num_bodies10_host.begin(), [](int num){ return num * 10; });

        // =============================  Num Bodies  =============================
        rigid_body_system->is_isolated.BuildFromSizes(num_bodies_host);
        rigid_body_system->q_lengths.BuildFromSizes(num_bodies_host);
        rigid_body_system->q_offset.BuildFromSizes(num_bodies_host);
        rigid_body_system->qpos_offset.BuildFromSizes(num_bodies_host);

        rigid_body_system->root_idx.BuildFromSizes(num_bodies_host);
        rigid_body_system->subtree_mass.BuildFromSizes(num_bodies_host);
        rigid_body_system->subtree_com.BuildFromSizes(num_bodies_host);
        rigid_body_system->batch_inertia.BuildFromSizes(num_bodies_host);
        rigid_body_system->batch_weight_inv.BuildFromSizes(num_bodies_host);
        rigid_body_system->joint_axis.BuildFromSizes(num_bodies_host);
        rigid_body_system->joint_anchor.BuildFromSizes(num_bodies_host);
        rigid_body_system->batch_global_com_pos.BuildFromSizes(num_bodies_host);
        rigid_body_system->batch_com_rot.BuildFromSizes(num_bodies_host);

        // ==========================  Num Bodies * 6   ===========================
        rigid_body_system->subtree_com_vel.BuildFromSizes(num_bodies6_host);
        rigid_body_system->batch_cacc.BuildFromSizes(num_bodies6_host);
        rigid_body_system->batch_cforce.BuildFromSizes(num_bodies6_host);


        // ==========================  Num Bodies * 10  ===========================
        rigid_body_system->subtree_inertia.BuildFromSizes(num_bodies10_host);
        rigid_body_system->batch_crb.BuildFromSizes(num_bodies10_host);

        
        // ==========================  Env Scalar  ===========================
        INIT_DYNO_ARRAY(rigid_body_system->batch_nv, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->batch_energy, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->batch_energy_ref, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->batch_scale, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->is_converged, num_envs);
        INIT_DYNO_ARRAY(rigid_body_system->sys_alpha, num_envs);

        
        // Calculate the DoFs and establish index
        DArray<int> num_qpos(num_envs);
        DArray<int> num_groups(num_envs);

        rigid_body_system->batch_nv_offset.BuildFromSizes(num_bodies_host);

        DofCountAndBuildIndexKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies, rigid_body_system->q_offset, rigid_body_system->q_lengths, 
            rigid_body_system->qpos_offset, rigid_body_system->parent_idx, rigid_body_system->is_static,
            rigid_body_system->is_isolated, rigid_body_system->joint_type,
            rigid_body_system->batch_nv, num_envs, num_qpos, num_groups);
        cudaDeviceSynchronize();
        
        // single thread build batch_nv_offset
        BuildGlobalDofOffsetKernel<<<1,1>>>(
        rigid_body_system->batch_bodies,
        rigid_body_system->q_lengths,
        rigid_body_system->batch_nv_offset,
        rigid_body_system->batch_nv,
        num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->batch_groups.BuildFromSizes(num_groups);

        FillGroupKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies, rigid_body_system->parent_idx, rigid_body_system->batch_groups, num_envs);
        cudaDeviceSynchronize();

        

        // 暂时用thrust来reduce出total_bodies和total_nv, 后续改成cpu上统计
        int total_bodies_, total_groups_, total_nv_;

        total_groups_ = rigid_body_system->batch_groups.TotalSize();

        total_bodies_ = thrust::reduce(
            thrust::device,
            rigid_body_system->batch_bodies.begin(),
            rigid_body_system->batch_bodies.begin() + num_envs,
            0
        );

        total_nv_ = thrust::reduce(
            thrust::device,
            rigid_body_system->batch_nv.begin(),
            rigid_body_system->batch_nv.begin() + num_envs,
            0
        );

        INIT_DYNO_ARRAY(rigid_body_system->flatten_group_to_env, total_groups_);
        INIT_DYNO_ARRAY(rigid_body_system->flatten_body_to_env, total_bodies_);
        INIT_DYNO_ARRAY(rigid_body_system->flatten_q_to_env_body, total_nv_);
        INIT_DYNO_ARRAY(rigid_body_system->flatten_constraint_to_env, num_envs * num_max_constraints);


        FillFlattenMappingInfoKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies, rigid_body_system->batch_body_offset, 
            rigid_body_system->q_lengths,rigid_body_system->batch_nv_offset, rigid_body_system->batch_groups, 
            rigid_body_system->flatten_group_to_env, rigid_body_system->flatten_body_to_env, 
            rigid_body_system->flatten_q_to_env_body, num_envs);

        // 2. malloc the solver states based on the DoF count
        rigid_body_system->batch_qpos.BuildFromSizes(num_qpos);
        

        // ==========================  Num Nv  ===========================
        std::vector<int> batch_nv_host(num_envs);
        cudaMemcpy(batch_nv_host.data(), rigid_body_system->batch_nv.begin(), num_envs * sizeof(int), cudaMemcpyDeviceToHost);
        
        rigid_body_system->batch_qacc.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_q_ex_acc.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_dx.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_qvel.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_q_inner_force.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_q_ex_force.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_grad.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_grad_cpy.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_dof_weight_inv.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_Ma.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_Ma_line_search.BuildFromSizes(batch_nv_host);
        rigid_body_system->batch_q_chain.BuildFromSizes(batch_nv_host);

        // ========================  Num Nv * 6  ==========================
        std::vector<int> num_dofs6_host(num_envs);
        std::transform(batch_nv_host.begin(), batch_nv_host.end(), num_dofs6_host.begin(), [](int num){ return num * 6; });
        rigid_body_system->batch_cdof.BuildFromSizes(num_dofs6_host);
        rigid_body_system->batch_cdof_dot.BuildFromSizes(num_dofs6_host);

        // ====================  Num Max Constraints  ====================
        std::vector<int> max_constraints_host(num_envs, num_max_constraints);
        rigid_body_system->batch_aref.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_imp.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_Jaref.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_Jaref_line_search.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_constraint_energy.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_unquads.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_dA.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_D.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_constraint_vel.BuildFromSizes(max_constraints_host);
        rigid_body_system->batch_constraint_force.BuildFromSizes(max_constraints_host);

        // ===============  Dense Matrices of size Nv * Nv  ==============
        rigid_body_system->batch_H.BuildFromSquares(rigid_body_system->batch_nv);
        rigid_body_system->batch_qM.BuildFromSquares(rigid_body_system->batch_nv);
        rigid_body_system->batch_qM_L.BuildFromSquares(rigid_body_system->batch_nv);

        // ===============  Dense Matrices of size Nc * Nv  ==============
        rigid_body_system->batch_J.BuildFromShapes(max_constraints_host, batch_nv_host);
        

        INIT_DYNO_ARRAY(rigid_body_system->collision_constraints.collision_nums, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.body_idxs, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.depth, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.normal, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.point, num_envs, 1024);
        INIT_DYNO_ARRAY2D(rigid_body_system->collision_constraints.mu, num_envs, 1024);

        spdlog::info("[MujocoSolver Solver] Allocated solver state arrays based on DoF counts.");
        // Initialize mass matrix for isolated bodies
        InitInertiaKernel<<<32, 512>>>(rigid_body_system->batch_bodies, rigid_body_system->batch_inertia,
            rigid_body_system->batch_mass, rigid_body_system->spheres, rigid_body_system->boxes,
            rigid_body_system->capsules, rigid_body_system->shape_type, rigid_body_system->shape_idx, num_envs);
        cudaDeviceSynchronize();
        // 3. Initialize the qpos
        InitQposKernel<<<32, 512>>>(rigid_body_system->batch_bodies, rigid_body_system->batch_qpos,
            rigid_body_system->qpos_offset, rigid_body_system->batch_pos, rigid_body_system->batch_quat,
            rigid_body_system->joint_qpos, rigid_body_system->joint_qpos_offset, rigid_body_system->parent_idx,
            rigid_body_system->joint_type, rigid_body_system->is_static, num_envs);
        cudaDeviceSynchronize();
        // 4. Build root index and calculate subtree mass
        BuildRootIndexKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->root_idx,
            rigid_body_system->subtree_mass,
            rigid_body_system->batch_mass,
            rigid_body_system->parent_idx,
            num_envs);
        cudaDeviceSynchronize();
        CalculateSubtreeMassKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->subtree_mass,
            rigid_body_system->parent_idx,
            num_envs);
        cudaDeviceSynchronize();

        spdlog::info("[MujocoSolver Solver] Initialization complete. Number of environments: {}", env_infos->num_envs);

        // Init batch cholesky solver
        cudaStream_t stream = nullptr;
        cuSafeCall(cudaStreamCreate(&stream));
        cholesky_solver = std::make_shared<BatchedCholeskySolver<typename TDataType::Real>>();
        cholesky_solver->Initialize(stream);

        // Init batch line search
        int alpha_nums = 10;
        std::vector<Real> alphas_host(alpha_nums);
        for(int i = 0; i < alpha_nums; i++)
            alphas_host[i] = (1.f / (alpha_nums - 1)) * i;
        rigid_body_system->alpha_cands.assign(alphas_host);
        rigid_body_system->batch_alpha_energies.BuildFromSizes(std::vector<int>(num_envs, alpha_nums));


        spdlog::info("[MujocoSolver Solver] Finished initialization.");
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::Step()
    {
        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        // 1. Reset forces and accelerations
        rigid_body_system->batch_q_inner_force.Reset();
        rigid_body_system->batch_q_ex_force.Reset();
        rigid_body_system->batch_q_ex_acc.Reset();
        rigid_body_system->is_converged.reset();
        // Update forward kinematics and subtree com
        ForwardKinematics();

        MakeConstraints();

        // q_ex_force = -q_inner_force
        SumBatchArray<<<32, 128>>>(rigid_body_system->batch_q_ex_force, rigid_body_system->batch_q_inner_force,
            rigid_body_system->batch_q_ex_force, false, DArray<Real>(), DArray<int>());
        cudaDeviceSynchronize();

        // Solve qM * q_ex_acc = q_ex_force by Cholesky factorization instead of explicitly forming qM^{-1}.        
        rigid_body_system->batch_qM_L.Assign(rigid_body_system->batch_qM);
        auto& qM_L= rigid_body_system->batch_qM_L;
        cholesky_solver->Factorize(qM_L.Begin(), rigid_body_system->is_converged.begin(),rigid_body_system->batch_nv.begin(), 
            qM_L.Offsets().Begin(), env_infos->num_envs, CholeskyMethod::PaddedTiled);
            
        DevArr2D<Real> q_ex_force_bak;
        q_ex_force_bak.Assign(rigid_body_system->batch_q_ex_force);
        cholesky_solver->Solve(qM_L.Begin(), q_ex_force_bak.Begin(),
            rigid_body_system->is_converged.begin(),
            rigid_body_system->batch_nv.begin(), qM_L.Offsets().Begin(), 
            rigid_body_system->batch_q_ex_acc.Offsets().Begin(),
            env_infos->num_envs, CholeskyMethod::PaddedTiled);

        // PrintVector<<<1, 1>>>(q_ex_force_bak, 0);
        // cudaDeviceSynchronize();

        rigid_body_system->batch_q_ex_acc.Assign(q_ex_force_bak);
        // =========================== Batch Cholesky Solver Usage ===========================

        rigid_body_system->batch_qacc.Assign(rigid_body_system->batch_q_ex_acc);

        NewtonSolver();

        TimeIntegration();
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


        thrust::fill(thrust::device, rigid_body_system->sys_alpha.begin(), rigid_body_system->sys_alpha.begin() + num_envs, 1.f);
        ComputeScale<TDataType><<<1, num_envs>>>(
            rigid_body_system->batch_nv,
            rigid_body_system->batch_qM,
            rigid_body_system->batch_scale,
            num_envs);

        // rigid_body_system->sys_alpha.reset();

        int iter = 0;
        while(iter < 10)
        {
            BatchLineSearch();

            // Update qacc      qacc += α * dx
            SumBatchArray<<<32, 128>>>(rigid_body_system->batch_qacc, rigid_body_system->batch_dx,
                rigid_body_system->batch_qacc, true, rigid_body_system->sys_alpha, rigid_body_system->is_converged);
            cudaDeviceSynchronize();

            // Update Ma        Ma += α * qM * dx
            BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_dx, 
                rigid_body_system->batch_Ma, rigid_body_system->batch_nv, rigid_body_system->batch_nv, true, 
                rigid_body_system->sys_alpha, rigid_body_system->is_converged);
            cudaDeviceSynchronize();
            // Update Jaref     Jaref += α * J * dx
            BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_dx, 
                rigid_body_system->batch_Jaref, rigid_body_system->num_constraints, rigid_body_system->batch_nv, 
                true, rigid_body_system->sys_alpha, rigid_body_system->is_converged);
            cudaDeviceSynchronize();
            // spdlog::info("qacc in newton");
            // PrintVector<<<1, 1>>>(rigid_body_system->batch_qacc, 0);
            // cuSynchronize();
            // spdlog::info("Ma in newton");
            // PrintVector<<<1, 1>>>(rigid_body_system->batch_Ma, 0);
            // cuSynchronize();
            // spdlog::info("Jaref in newton");
            // PrintVector<<<1, 1>>>(rigid_body_system->batch_Jaref, 0);
            // cuSynchronize();

            rigid_body_system->batch_energy_ref.assign(rigid_body_system->batch_energy);
            ComputeEnergy();
            BuildHessian();
            UpdateGradient();
            SolveSystem();


            CheckConvergenceKernel<TDataType><<<32, 128>>>(
                rigid_body_system->is_converged,
                rigid_body_system->batch_scale,
                rigid_body_system->batch_grad,
                rigid_body_system->batch_nv,
                rigid_body_system->batch_energy_ref,
                rigid_body_system->batch_energy,
                num_envs,
                1e-8f,
                1e-8f);
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

        UpdateGeneralizedVelKernel<<<32, 512>>>(
            rigid_body_system->batch_nv,
            rigid_body_system->batch_qvel,
            rigid_body_system->batch_qacc,
            env_infos->timesteps,
            env_infos->num_envs);
        cudaDeviceSynchronize();

        TimeIntegrationKernel<<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->parent_idx,
            rigid_body_system->is_static,
            rigid_body_system->qpos_offset,
            rigid_body_system->q_offset,
            rigid_body_system->batch_qpos,
            rigid_body_system->batch_qvel,
            rigid_body_system->joint_type,
            env_infos->timesteps,
            env_infos->num_envs);
        cudaDeviceSynchronize();

        spdlog::info("Qvel:");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qvel, 0);
        cudaDeviceSynchronize();
        spdlog::info("QPOS:");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qpos, 0);
        cudaDeviceSynchronize();

        UpdateJointPoseKernel<TDataType><<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->batch_pos,
            rigid_body_system->batch_quat,
            rigid_body_system->batch_rot,
            rigid_body_system->joint_qpos,
            rigid_body_system->parent_idx,
            rigid_body_system->is_static,
            rigid_body_system->qpos_offset,
            rigid_body_system->joint_qpos_offset,
            rigid_body_system->batch_qpos,
            rigid_body_system->joint_type,
            env_infos->num_envs);
        cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ForwardKinematics()
    {
        spdlog::info("[MujocoSolver Solver] Start forward kinematics.");

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
        const int num_groups = rigid_body_system->batch_groups.TotalSize();

        const int threads = 128;
        const int blocks_group = (num_groups + threads - 1) / threads;
        
        ForwardKinematicsKernel<TDataType><<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            rigid_body_system->batch_quat,
            rigid_body_system->batch_rot,
            rigid_body_system->parent_idx,
            rigid_body_system->joint_axis_ref,
            rigid_body_system->joint_anchor_ref,
            rigid_body_system->joint_type,
            rigid_body_system->joint_qpos,
            rigid_body_system->joint_qpos_ref,
            rigid_body_system->joint_qpos_offset,
            rigid_body_system->batch_pos,
            rigid_body_system->joint_rel_quat,
            rigid_body_system->joint_axis,
            rigid_body_system->joint_rel_pos,
            rigid_body_system->joint_anchor,
            rigid_body_system->batch_global_com_pos,
            rigid_body_system->batch_local_com_pos,
            rigid_body_system->batch_local_com_quat,
            rigid_body_system->batch_com_rot,
            num_groups);
            
        cudaDeviceSynchronize();

        SubtreeComKernel<TDataType><<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            rigid_body_system->subtree_com,
            rigid_body_system->batch_mass,
            rigid_body_system->subtree_mass,
            rigid_body_system->batch_global_com_pos,
            rigid_body_system->parent_idx,
            num_envs,
            num_groups);
        cudaDeviceSynchronize();

        // rigid_body_system->batch_cdof.reset();
        const int total_bodies_ = rigid_body_system->flatten_body_to_env.size();
        const int blocks_body = (total_bodies_ + threads - 1) / threads;

        ComputeCdofKernel<TDataType><<<blocks_body, threads>>>(
            rigid_body_system->flatten_body_to_env,
            rigid_body_system->batch_body_offset,
            rigid_body_system->batch_cdof,
            rigid_body_system->parent_idx,
            rigid_body_system->batch_pos,
            rigid_body_system->batch_rot,
            rigid_body_system->subtree_com,
            rigid_body_system->q_offset,
            rigid_body_system->root_idx,
            rigid_body_system->joint_anchor,
            rigid_body_system->joint_type,
            rigid_body_system->joint_axis,
            rigid_body_system->is_static,
            num_envs,
            total_bodies_);
        cudaDeviceSynchronize();

        // spdlog::info("Env 0");
        // PrintTestInfos<<<1, num_envs>>>(
        //     rigid_body_system->batch_bodies,
        //     rigid_body_system->q_offset,
        //     rigid_body_system->q_lengths,
        //     rigid_body_system->batch_cdof,
        //     rigid_body_system->is_static,
        //     rigid_body_system->is_isolated,
        //     rigid_body_system->shape_type,
        //     rigid_body_system->parent_idx,
        //     rigid_body_system->joint_type,
        //     num_envs, 0);
        // cudaDeviceSynchronize();
        // spdlog::info("Env 1");
        // PrintTestInfos<<<1, num_envs>>>(
        //     rigid_body_system->batch_bodies,
        //     rigid_body_system->q_offset,
        //     rigid_body_system->q_lengths,
        //     rigid_body_system->batch_cdof,
        //     rigid_body_system->is_static,
        //     rigid_body_system->is_isolated,
        //     rigid_body_system->shape_type,
        //     rigid_body_system->parent_idx,
        //     rigid_body_system->joint_type,
        //     num_envs, 1);
        // cudaDeviceSynchronize();

        // Crb
        rigid_body_system->batch_crb.Reset();
        // 1. Calculate the global inertia matrix of each rigid body when the center of mass of the corresponding kinematic tree is taken as the reference point.
        SubtreeInertialKernel<<<blocks_body, threads>>>(
            rigid_body_system->flatten_body_to_env,
            rigid_body_system->batch_body_offset,
            rigid_body_system->root_idx,
            rigid_body_system->batch_global_com_pos,
            rigid_body_system->subtree_com,
            rigid_body_system->subtree_inertia,
            rigid_body_system->batch_inertia,
            rigid_body_system->batch_com_rot,
            rigid_body_system->batch_mass,
            rigid_body_system->batch_crb,
            num_envs,
            total_bodies_);
        // 2. Calculate the global inertia matrix of each sub-tree.
        cudaDeviceSynchronize();

        AccumulateSubtreeInertialKernel<<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            rigid_body_system->parent_idx,
            rigid_body_system->batch_crb,
            rigid_body_system->subtree_inertia,
            num_envs,
            num_groups);
        cudaDeviceSynchronize();


        // 3. Construct the system inertia matrix in the generalized coordinate system.
        const int total_nv_ = rigid_body_system->flatten_q_to_env_body.size();

        rigid_body_system->batch_qM.Reset();
        const int blocks_nv = (total_nv_ + threads - 1) / threads;
        UpdateGeneralizedInertialMatrixKernel<TDataType><<<blocks_nv, threads>>>(
            rigid_body_system->flatten_q_to_env_body,
            rigid_body_system->batch_qM,
            rigid_body_system->batch_nv_offset,
            rigid_body_system->batch_nv,
            rigid_body_system->is_isolated,
            rigid_body_system->is_static,
            rigid_body_system->parent_idx,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->batch_cdof,
            rigid_body_system->batch_crb,
            rigid_body_system->batch_inertia,
            rigid_body_system->batch_mass,
            num_envs,
            total_nv_);
        cudaDeviceSynchronize();
        // mirror the upper triangle to the lower triangle
        FillGeneralizedInertialMatrixSymmetryKernel<<<32, 128>>>(
            rigid_body_system->batch_qM,
            rigid_body_system->batch_nv,
            num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->subtree_com_vel.Reset();
        // rigid_body_system->batch_cdof_dot.reset();
        ComputeComVelKernel<TDataType><<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            rigid_body_system->batch_cdof,
            rigid_body_system->batch_qvel,
            rigid_body_system->subtree_com_vel,
            rigid_body_system->batch_cdof_dot,
            rigid_body_system->parent_idx,
            rigid_body_system->q_offset,
            rigid_body_system->joint_type,
            rigid_body_system->is_static,
            num_envs,
            num_groups);
        cudaDeviceSynchronize();
        // 4. Recursive Newton-Euler in split stages.
        rigid_body_system->batch_cacc.Reset();
        rigid_body_system->batch_cforce.Reset();

        // 4.1 Forward recursion: propagate spatial accelerations along each kinematic tree.
        RNECaccKernel<TDataType><<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            env_infos->gravities,
            rigid_body_system->batch_cacc,
            rigid_body_system->batch_cdof_dot,
            rigid_body_system->batch_qvel,
            rigid_body_system->parent_idx,
            rigid_body_system->is_static,
            rigid_body_system->joint_type,
            rigid_body_system->q_offset,
            num_envs,
            num_groups);
        cudaDeviceSynchronize();

        // 4.2 Body-wise Newton-Euler: compute each body's spatial force from inertia, acceleration and velocity.
        RNEForceKernel<TDataType><<<blocks_body, threads>>>(
            rigid_body_system->flatten_body_to_env,
            rigid_body_system->batch_body_offset,
            rigid_body_system->batch_cacc,
            rigid_body_system->batch_cforce,
            rigid_body_system->subtree_inertia,
            rigid_body_system->subtree_com_vel,
            num_envs,
            total_bodies_);
        cudaDeviceSynchronize();

        // 4.3 Backward recursion: accumulate child reaction forces back to the parent body.
        RNEAccumKernel<TDataType><<<blocks_group, threads>>>(
            rigid_body_system->batch_groups,
            rigid_body_system->flatten_group_to_env,
            rigid_body_system->parent_idx,
            rigid_body_system->batch_cforce,
            num_envs,
            num_groups);
        cudaDeviceSynchronize();

        // 4.4 Project spatial forces to generalized joint forces.
        rigid_body_system->batch_q_inner_force.Reset();
        RNEProjKernel<TDataType><<<blocks_nv, threads>>>(
            rigid_body_system->flatten_q_to_env_body,
            rigid_body_system->batch_nv_offset,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->batch_cdof,
            rigid_body_system->batch_cforce,
            rigid_body_system->batch_q_inner_force,
            num_envs,
            total_nv_);
        cudaDeviceSynchronize();

        // ComputeRNEKernel<TDataType><<<32, 512>>>(
        //     rigid_body_system->batch_bodies,
        //     env_infos->gravities,
        //     rigid_body_system->batch_cacc,
        //     rigid_body_system->batch_cforce,
        //     rigid_body_system->batch_q_inner_force,
        //     rigid_body_system->batch_cdof,
        //     rigid_body_system->batch_cdof_dot,
        //     rigid_body_system->batch_qvel,
        //     rigid_body_system->q_offset,
        //     rigid_body_system->subtree_inertia,
        //     rigid_body_system->subtree_com_vel,
        //     rigid_body_system->parent_idx,
        //     rigid_body_system->is_static,
        //     rigid_body_system->joint_type,
        //     rigid_body_system->q_lengths,
        //     num_envs);

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

        CollisonDetectionKernel<TDataType><<<32, 512>>>(
            rigid_body_system->collision_constraints,
            rigid_body_system->batch_bodies,
            rigid_body_system->is_static,
            rigid_body_system->batch_pos,
            rigid_body_system->batch_rot,
            rigid_body_system->shape_type,
            rigid_body_system->shape_idx,
            rigid_body_system->boxes,
            num_envs);
        cudaDeviceSynchronize();
        spdlog::info("Collision detection done.");
        
        UpdateAnchorConstarints<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->anchor_constraints,
            rigid_body_system->batch_rot,
            rigid_body_system->batch_pos,
            num_envs);
        cudaDeviceSynchronize();
        spdlog::info("Anchor constraints updated.");
        
        UpdateJointLimitConstraints<TDataType><<<32, 512>>>(
            rigid_body_system->joint_limit_constraints,
            rigid_body_system->joint_type,
            rigid_body_system->joint_qpos_offset,
            rigid_body_system->joint_qpos,
            num_envs);
        cudaDeviceSynchronize();
        spdlog::info("Joint limit constraints updated.");

        CountConstraintNums<TDataType><<<1, num_envs>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->joint_limit_constraints,
            rigid_body_system->collision_constraints,
            rigid_body_system->constraint_offset,
            rigid_body_system->num_constraints,
            num_envs);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::MakeJacobian()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_J.Reset();
        ContactConstraintJacobianKernel<TDataType><<<32, 512>>>(
            rigid_body_system->collision_constraints,
            rigid_body_system->batch_nv,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_J,
            rigid_body_system->root_idx,
            rigid_body_system->subtree_com,
            rigid_body_system->batch_cdof,
            rigid_body_system->is_static,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->parent_idx,
            num_envs);
        cudaDeviceSynchronize();

        // Anchor constraints
        AnchorConstraintJacobianKernel<TDataType><<<32, 512>>>(
            rigid_body_system->anchor_constraints,
            rigid_body_system->num_each_constraint,
            rigid_body_system->batch_nv,
            rigid_body_system->batch_J,
            rigid_body_system->root_idx,
            rigid_body_system->subtree_com,
            rigid_body_system->batch_cdof,
            rigid_body_system->is_static,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->parent_idx,
            num_envs);
        cudaDeviceSynchronize();

        // Friction loss constraints
        FrictionLossJacobianKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_nv,
            rigid_body_system->batch_J,
            rigid_body_system->friction_loss_constraints,
            num_envs);
        cudaDeviceSynchronize();

        // Joint limit constraints
        JointLimitJacobianKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->joint_limit_constraints,
            rigid_body_system->q_offset,
            rigid_body_system->joint_type,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_nv,
            rigid_body_system->batch_J,
            num_envs);
        cudaDeviceSynchronize();


        // spdlog::info("Jacobian: ");
        // PrintJacobian<TDataType><<<1, 1>>>(
        //     rigid_body_system->batch_J,
        //     rigid_body_system->num_constraints,
        //     rigid_body_system->batch_nv,
        //     0);
        // cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeAref()
    {
        // constraint_vel = J * qvel
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_qvel, rigid_body_system->batch_constraint_vel,
            rigid_body_system->num_constraints, rigid_body_system->batch_nv, false, DArray<Real>(), DArray<int>());
        cudaDeviceSynchronize();

        ComputeAnchorAref<TDataType><<<32, 512>>>(
            rigid_body_system->anchor_constraints,
            rigid_body_system->num_each_constraint,
            rigid_body_system->batch_constraint_vel,
            rigid_body_system->batch_aref,
            rigid_body_system->batch_imp,
            num_envs);
        ComputeFrictionLossAref<TDataType><<<32, 512>>>(
            rigid_body_system->friction_loss_constraints,
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_constraint_vel,
            rigid_body_system->batch_imp,
            rigid_body_system->batch_aref,
            num_envs);
        ComputeJointLimitAref<TDataType><<<32, 512>>>(
            rigid_body_system->joint_limit_constraints,
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_constraint_vel,
            rigid_body_system->batch_imp,
            rigid_body_system->batch_aref,
            num_envs);
        ComputeContactAref<TDataType><<<32, 512>>>(
            rigid_body_system->collision_constraints,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_constraint_vel,
            rigid_body_system->contact_weights,
            rigid_body_system->batch_imp,
            rigid_body_system->batch_aref,
            num_envs);
        cudaDeviceSynchronize();

        // printf("Aref:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_aref, 0);
        // cudaDeviceSynchronize();

        // printf("Imp:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_imp, 0);
        // cudaDeviceSynchronize();

        // Compute constraint residuals Jaref
        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_qacc,
            rigid_body_system->batch_Ma, rigid_body_system->batch_nv, rigid_body_system->batch_nv, false, DArray<Real>(), DArray<int>());
        cudaDeviceSynchronize();
        printf("Ma!!!:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_Ma, 0);
        cudaDeviceSynchronize();

        printf("qacc!!!:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qacc, 0);
        cudaDeviceSynchronize();

        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_qacc,
            rigid_body_system->batch_Jaref, rigid_body_system->num_constraints, rigid_body_system->batch_nv, false, DArray<Real>(), DArray<int>());
        cudaDeviceSynchronize();

        SumBatchArray<<<32, 128>>>(rigid_body_system->batch_Jaref, rigid_body_system->batch_aref, 
            rigid_body_system->batch_Jaref, false, DArray<Real>(), DArray<int>());
        cudaDeviceSynchronize();

        // printf("Jaref:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_Jaref, 0);
        // cudaDeviceSynchronize();


    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeRD()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
    
        ComputeDiagJMinvJTForBodies<TDataType><<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->is_static,
            rigid_body_system->batch_nv,
            rigid_body_system->batch_global_com_pos,
            rigid_body_system->root_idx,
            rigid_body_system->subtree_com,
            rigid_body_system->batch_cdof,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->parent_idx,
            rigid_body_system->batch_qM_L,
            rigid_body_system->batch_weight_inv,
            num_envs);
        cudaDeviceSynchronize();

        ComputeDiagJMinvJTForJoints<TDataType><<<32, 512>>>(
            rigid_body_system->batch_bodies,
            rigid_body_system->batch_nv,
            rigid_body_system->q_offset,
            rigid_body_system->q_lengths,
            rigid_body_system->joint_type,
            rigid_body_system->batch_qM_L,
            rigid_body_system->batch_dof_weight_inv,
            num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->batch_dA.Reset();

        ComputeAnchor_dAKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->anchor_constraints,
            rigid_body_system->batch_weight_inv,
            rigid_body_system->batch_dA,
            num_envs);
        ComputeFrictionLoss_dAKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->friction_loss_constraints,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_dof_weight_inv,
            rigid_body_system->batch_dA,
            num_envs);
        ComputeJointLimit_dAKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_each_constraint,
            rigid_body_system->joint_limit_constraints,
            rigid_body_system->q_offset,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_dof_weight_inv,
            rigid_body_system->batch_dA,
            num_envs);
        ComputeContact_dAKernel<TDataType><<<32, 512>>>(
            rigid_body_system->collision_constraints,
            rigid_body_system->batch_weight_inv,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_dA,
            num_envs);
        cudaDeviceSynchronize();

        ComputeRKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_constraints,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_D,
            rigid_body_system->batch_imp,
            rigid_body_system->batch_dA,
            rigid_body_system->collision_constraints,
            num_envs);
        cudaDeviceSynchronize();

        // printf("R:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_D, 0);
        // cudaDeviceSynchronize();

        ComputeDKernel<TDataType><<<32, 512>>>(
            rigid_body_system->num_constraints,
            rigid_body_system->batch_D,
            num_envs);
        cudaDeviceSynchronize();

        // printf("D:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_D, 0);
        // cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::ComputeEnergy()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_energy.reset();
        rigid_body_system->batch_constraint_energy.Reset();
        rigid_body_system->batch_unquads.Reset();

        AnchorEnergyKernel<TDataType><<<num_envs, 512>>>(
            rigid_body_system->is_converged,
            rigid_body_system->num_each_constraint,
            rigid_body_system->batch_constraint_force,
            rigid_body_system->batch_constraint_energy,
            rigid_body_system->batch_D,
            rigid_body_system->batch_Jaref,
            num_envs);
        FrictionLossEnergyKernel<TDataType><<<num_envs, 512>>>(
            rigid_body_system->is_converged,
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_constraint_force,
            rigid_body_system->batch_constraint_energy,
            rigid_body_system->batch_D,
            rigid_body_system->batch_Jaref,
            rigid_body_system->friction_loss_constraints,
            rigid_body_system->batch_unquads,
            num_envs);
        ContactAndJointLimitEnergyKernel<TDataType><<<num_envs, 512>>>(
            rigid_body_system->is_converged,
            rigid_body_system->num_constraints,
            rigid_body_system->constraint_offset,
            rigid_body_system->batch_constraint_force,
            rigid_body_system->batch_constraint_energy,
            rigid_body_system->batch_Jaref,
            rigid_body_system->batch_D,
            rigid_body_system->batch_unquads,
            num_envs);
        cudaDeviceSynchronize();

        ReduceConstraintEnergyKernel<TDataType><<<num_envs, 256>>>(
            rigid_body_system->is_converged,
            rigid_body_system->num_constraints,
            rigid_body_system->batch_constraint_energy,
            rigid_body_system->batch_energy,
            num_envs);
        cudaDeviceSynchronize();

        // spdlog::info("Constraint force:");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_constraint_force, 0);
        // cudaDeviceSynchronize();

        spdlog::info("energies constraints: ");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_energy, 1);
        cudaDeviceSynchronize();

        InertialEnergyKernel<TDataType><<<num_envs, 512>>>(
            rigid_body_system->is_converged,
            rigid_body_system->batch_nv,
            rigid_body_system->batch_Ma,
            rigid_body_system->batch_q_ex_force,
            rigid_body_system->batch_qacc,
            rigid_body_system->batch_q_ex_acc,
            rigid_body_system->batch_energy,
            num_envs);
        cudaDeviceSynchronize();
        spdlog::info("energies total: ");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_energy, 1);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::BuildHessian()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_H.Reset();

        const int max_nv = rigid_body_system->max_bodies * 6;
        dim3 block(16, 16, 1);
        dim3 grid(num_envs,
            (max_nv + block.y - 1) / block.y,
            (max_nv + block.x - 1) / block.x);

        BuildHessianKernel<TDataType><<<grid, block>>>(
            rigid_body_system->is_converged,
            rigid_body_system->batch_nv,
            rigid_body_system->num_constraints,
            rigid_body_system->batch_J,
            rigid_body_system->batch_D,
            rigid_body_system->batch_unquads,
            rigid_body_system->batch_qM,
            rigid_body_system->batch_H,
            num_envs);
        cudaDeviceSynchronize();

        // printf("Hessian:\n");
        // PrintVector<<<1, 1>>>(rigid_body_system->batch_H, 0, 6 * 6);
        // cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::UpdateGradient()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_grad.Reset();

        UpdateGradientKernel<TDataType><<<num_envs, 512>>>(
            rigid_body_system->is_converged,
            rigid_body_system->batch_nv,
            rigid_body_system->num_constraints,
            rigid_body_system->batch_J,
            rigid_body_system->batch_constraint_force,
            rigid_body_system->batch_Ma,
            rigid_body_system->batch_q_ex_force,
            rigid_body_system->batch_grad,
            num_envs);
        cudaDeviceSynchronize();

        printf("Gradient:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_grad, 0);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::SolveSystem()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;
        rigid_body_system->batch_dx.Reset();

        auto& H = rigid_body_system->batch_H;
        auto& grad = rigid_body_system->batch_grad_cpy;
        auto& x = rigid_body_system->batch_dx; // reuse qacc as solution
        auto& is_converged = rigid_body_system->is_converged;

        grad.Assign(rigid_body_system->batch_grad);
        cholesky_solver->Factorize(H.Begin(), is_converged.begin(), rigid_body_system->batch_nv.begin(), 
            H.Offsets().Begin(), env_infos->num_envs, CholeskyMethod::PaddedTiled);
            
        cholesky_solver->Solve(H.Begin(), grad.Begin(), is_converged.begin(),
            rigid_body_system->batch_nv.begin(), H.Offsets().Begin(), 
            grad.Offsets().Begin(),
            env_infos->num_envs, CholeskyMethod::PaddedTiled);

        x.Assign(grad);


        printf("dx (solution):\n");
        PrintVector<<<1, 1>>>(x, 0);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::BatchLineSearch()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->sys_alpha.reset();
        rigid_body_system->batch_alpha_energies.Reset();
        // 要更新的变量：qacc, Ma, Jaref
        // qacc += α * dx
        // Ma += α * qM * dx
        // Jaref += α * J * dx
        // Update qacc      qacc += α * dx

        // Update Ma        qM * dx
        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_dx, rigid_body_system->batch_Ma_line_search,
            rigid_body_system->batch_nv, rigid_body_system->batch_nv, false, DArray<Real>(), rigid_body_system->is_converged);
        cudaDeviceSynchronize();
        // Update Jaref     J * dx
        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_J, rigid_body_system->batch_dx, rigid_body_system->batch_Jaref_line_search,
            rigid_body_system->num_constraints, rigid_body_system->batch_nv, false, DArray<Real>(), rigid_body_system->is_converged);
        cudaDeviceSynchronize();

        // 1. 多个environment并行评估不同α下的inertia能量
        BatchAlphaInertiaEnergyKernel<<<num_envs, 512>>>(
            rigid_body_system->batch_Ma,
            rigid_body_system->batch_Ma_line_search,
            rigid_body_system->batch_q_ex_force,
            rigid_body_system->batch_qacc,
            rigid_body_system->batch_dx,
            rigid_body_system->batch_q_ex_acc,
            rigid_body_system->batch_alpha_energies,
            rigid_body_system->alpha_cands,
            rigid_body_system->batch_nv, rigid_body_system->is_converged, 
            env_infos->num_envs, rigid_body_system->alpha_cands.size());
        cudaDeviceSynchronize();

        // 2. 多个environment并行评估不同α下的Constraint能量
        BatchAlphaAnchorEnergyKernel<<<num_envs, 512>>>(
            rigid_body_system->batch_Jaref,
            rigid_body_system->batch_Jaref_line_search,
            rigid_body_system->batch_D,
            rigid_body_system->batch_alpha_energies,
            rigid_body_system->alpha_cands,
            rigid_body_system->num_each_constraint,
            rigid_body_system->is_converged,
            env_infos->num_envs, rigid_body_system->alpha_cands.size());
        BatchAlphaFrictionEnergyKernel<<<num_envs, 512>>>(
            rigid_body_system->friction_loss_constraints,
            rigid_body_system->batch_Jaref,
            rigid_body_system->batch_Jaref_line_search,
            rigid_body_system->batch_D,
            rigid_body_system->batch_alpha_energies,
            rigid_body_system->alpha_cands,
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->is_converged,
            env_infos->num_envs, rigid_body_system->alpha_cands.size());
        BatchAlphaContactAndJointLimitEnergyKernel<<<num_envs, 512>>>(
            rigid_body_system->batch_Jaref,
            rigid_body_system->batch_Jaref_line_search,
            rigid_body_system->batch_D,
            rigid_body_system->batch_alpha_energies,
            rigid_body_system->alpha_cands,
            rigid_body_system->num_each_constraint,
            rigid_body_system->constraint_offset,
            rigid_body_system->is_converged,
            env_infos->num_envs, rigid_body_system->alpha_cands.size());
        cudaDeviceSynchronize();

        ChooseAlphaKernel<<<1, num_envs>>>(
            rigid_body_system->batch_alpha_energies,
            rigid_body_system->sys_alpha,
            rigid_body_system->alpha_cands,
            rigid_body_system->is_converged,
            env_infos->num_envs, rigid_body_system->alpha_cands.size());
        cudaDeviceSynchronize();
    }

    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}
