#pragma once

#include "Field/VehicleInfo.h"
#include "MujocoSolver.h"
#include "../../Utils/utils.h"
#include "Algorithm.h"
#include <cstdio>


namespace dyno
{
namespace device_kernel
{

    // 查找Offset数组中target应该落在哪个区间
    template<typename TArray, typename T>
    __device__ inline int BinarySearchLEArray(const TArray& arr, int n, T target)
    {
        int left = 0;
        int right = n - 1;
        int ans = -1;

        while (left <= right)
        {
            const int mid = left + ((right - left) >> 1);
            const auto value = arr[mid];

            if (value <= target)
            {
                ans = mid;
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }

        return ans;
    }

    __device__ void SubtreeComInertiaTemp(DArray2D<Real>& com_inertia, const Vec3f& body_inertia, const Mat3f& rot_mat, const Vec3f& offset, Real mass, int env_id, int bid)
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

        printf("Body[%d]: tmp: %f %f %f %f %f %f %f %f %f\n", bid, tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8);

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

    template<typename TDataType>
    __global__ void CountGroupsKernel(
        DArray<int> batch_bodies,
        DArray2D<int> parent_idx,
        DevArr<int> group_sizes,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (env_id >= num_envs)
            return;

        const int num_bodies = batch_bodies[env_id];
        int group_count = 0;

        for (int bid = 0; bid < num_bodies; ++bid)
        {
            if (parent_idx(env_id, bid) == -1)
                ++group_count;
        }

        group_sizes[env_id] = group_count;
    }

    template<typename TDataType>
    __global__ void BuildGroupsKernel(
        DArray<int> batch_bodies,
        DArray2D<int> parent_idx,
        DevBlockVector<Pair<int, int>> groups,
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

    template<typename TDataType>
    __global__ void PrintGroupsKernel(
        DevBlockVector<Pair<int, int>> groups,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (env_id >= num_envs) return;

        if (groups.Sizes()[env_id] <= 0) return;

        const auto g = groups.BlockPtr(env_id)[0];
        printf("[device-readback] env %d group 0: %d, %d\n", env_id, g.first, g.second);
    }
    
    template<typename TDataType>
    __global__ void ForwardKinematicsKernel(
        DevBlockVector<Pair<int, int>> groups,
        DArray2D<int> parent_idx_arr,
        DArray2D<Quat<typename TDataType::Real>> batch_quat,
        DArray2D<Mat3f> batch_rot,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Vec3f> joint_axis_ref,
        DArray2D<Vec3f> joint_anchor_ref,
        DArray2D<int> joint_type_arr,
        DArray2D<typename TDataType::Real> joint_qpos,
        DArray2D<typename TDataType::Real> joint_qpos_ref,
        DArray2D<int> joint_qpos_offset,
        DArray2D<Vec3f> joint_rel_pos,
        DArray2D<Quat<typename TDataType::Real>> joint_rel_quat,
        DArray2D<Vec3f> joint_axis,
        DArray2D<Vec3f> joint_anchor,
        int num_envs,
        int num_groups)
    {
        using Real = typename TDataType::Real;

        int group_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (group_id >= num_groups)
            return;

        int env_id = BinarySearchLEArray(groups.Offsets(), num_envs, group_id);

        if (env_id >= num_envs)
            return;

        const int local_gid = group_id - groups.Offsets()[env_id];
        const auto group = groups[env_id][local_gid];

        const int group_begin = group.first;
        const int group_count = group.second;

        for(int local = 0; local < group_count; ++local)
        {
            const int bid = group_begin + local;
            const int parent_idx = parent_idx_arr(env_id, bid);

            if (parent_idx == -1)
            {
                batch_rot(env_id, bid) = batch_quat(env_id, bid).toMatrix3x3();
            }
            else {
                const auto& parent_quat = batch_quat(env_id, parent_idx);
                const auto& parent_rot = batch_rot(env_id, parent_idx);
                const auto& local_axis = joint_axis_ref(env_id, bid);
                const auto& local_anchor = joint_anchor_ref(env_id, bid);
                const int& joint_type = joint_type_arr(env_id, bid);
                const auto& joint_qpos_start = joint_qpos_offset(env_id, bid);
                

                Quat<Real> xquat_p = parent_quat * joint_rel_quat(env_id, bid);
                joint_axis(env_id, bid) = RotateVector(local_axis, xquat_p);
                Vec3f xanchor = RotateVector(local_anchor, xquat_p);
                Vec3f xpos = parent_rot * joint_rel_pos(env_id, bid) + batch_pos(env_id, parent_idx);
                xanchor += xpos;
                joint_anchor(env_id, bid) = xanchor;


                if(joint_type == 2)     // Slide
                {
                    batch_quat(env_id, bid) = xquat_p;
                    batch_rot(env_id, bid) = xquat_p.toMatrix3x3();
                    batch_pos(env_id, bid) = xpos + (joint_qpos(env_id, joint_qpos_start) - joint_qpos_ref(env_id, joint_qpos_start)) * joint_axis(env_id, bid);
                }
                else
                {
                    Quat<Real> quat_local;
                    if(joint_type == 1)     // Hinge
                        quat_local = QuatFromAxisAngle(local_axis, joint_qpos(env_id, joint_qpos_start) - joint_qpos_ref(env_id, joint_qpos_start));
                    else if (joint_type == 3)   // Ball
                    {
                        Quat<Real> ball_quat = Quat<Real>(
                            joint_qpos(env_id, joint_qpos_start),
                            joint_qpos(env_id, joint_qpos_start + 1),
                            joint_qpos(env_id, joint_qpos_start + 2),
                            joint_qpos(env_id, joint_qpos_start + 3));
                        ball_quat.normalize();
                        quat_local = ball_quat;
                    }

                    Quat<Real> xquat_c = xquat_p * quat_local;
                    Quat<Real> xquat_c_bak = xquat_c;
                    xquat_c_bak.normalize();
                    batch_quat(env_id, bid) = xquat_c_bak;
                    batch_rot(env_id, bid) = xquat_c.toMatrix3x3();
                    xpos = RotateVector(local_anchor, xquat_c);
                    batch_pos(env_id, bid) = xanchor - xpos;
                }
            }
        }
    }

    template<typename TDataType>
    __global__ void SubtreeComGroupKernel(
        DevBlockVector<Pair<int, int>> groups,
        DArray2D<int> parent_idx,
        DArray2D<Vec3f> subtree_com,
        DArray2D<typename TDataType::Real> batch_mass,
        DArray2D<typename TDataType::Real> subtree_mass,
        DArray2D<Vec3f> batch_pos,
        int num_envs,
        int num_groups)
    {
        int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        int env_id = BinarySearchLEArray(groups.Offsets(), num_envs, group_id);
        if (env_id < 0 || env_id >= num_envs)
            return;

        const int group_offset = groups.Offsets()[env_id];
        const int group_size = groups.Sizes()[env_id];
        const int local_gid = group_id - group_offset;
        if (local_gid < 0 || local_gid >= group_size)
            return;

        const auto group = groups[env_id][local_gid];
        const int group_begin = group.first;
        const int group_count = group.second;
        const int group_end = group_begin + group_count;

        for (int local = 0; local < group_count; ++local)
        {
            const int bidx = group_begin + local;
            subtree_com(env_id, bidx) = batch_mass(env_id, bidx) * batch_pos(env_id, bidx);
        }

        for (int local = group_count - 1; local >= 0; --local)
        {
            const int bidx = group_begin + local;
            const int pid = parent_idx(env_id, bidx);
            if (pid >= group_begin && pid < group_end)
            {
                subtree_com(env_id, pid) += subtree_com(env_id, bidx);
            }
        }

        for (int local = 0; local < group_count; ++local)
        {
            const int bidx = group_begin + local;
            subtree_com(env_id, bidx) /= subtree_mass(env_id, bidx);
        }
    }

    template<typename TDataType>
    __global__ void SubtreeInertialKernel(
        DArray<int> batch_bodies,
        DArray<int> batch_body_offset,
        DArray2D<int> root_idx,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Vec3f> subtree_com,
        DArray2D<Vec3f> batch_inertia,
        DArray2D<Mat3f> batch_rot,
        DArray2D<typename TDataType::Real> batch_mass,
        DArray2D<typename TDataType::Real> subtree_inertia,
        DArray2D<typename TDataType::Real> batch_crb,
        int num_envs,
        int total_bodies)
    {
        using Real = typename TDataType::Real;

        const int global_bid = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_bid >= total_bodies)
            return;

        const int env_id = BinarySearchLEArray(batch_body_offset, num_envs, global_bid);
        if (env_id < 0 || env_id >= num_envs)
            return;

        const int env_body_begin = batch_body_offset[env_id];
        const int env_body_count = batch_bodies[env_id];
        if (global_bid >= env_body_begin + env_body_count)
            return;

        const int bid = global_bid - env_body_begin;
        const int rid = root_idx(env_id, bid);
        const Vec3f offset = batch_pos(env_id, bid) - subtree_com(env_id, rid);
        const Vec3f& body_inertia = batch_inertia(env_id, bid);
        const Mat3f rot_mat = batch_rot(env_id, bid);
        const Real mass = batch_mass(env_id, bid);

        SubtreeComInertiaTemp(subtree_inertia, body_inertia, rot_mat, offset, mass, env_id, bid);

        for (int i = 0; i < 10; ++i)
            batch_crb(env_id, bid * 10 + i) = subtree_inertia(env_id, bid * 10 + i);
    }


    // Flatten bodies across all environments. If a q-parallel kernel is needed later,
    // it is better to introduce an explicit q_to_body mapping.
    template<typename TDataType>
    __global__ void ComputeCdofKernel(
        DArray<int> batch_bodies,
        DArray<int> env_body_offsets,
        DArray2D<int> parent_idx,
        DArray2D<int> root_idx,
        DArray2D<int> is_static,
        DArray2D<int> joint_type,
        DArray2D<int> q_offset,
        DArray2D<typename TDataType::Real> batch_cdof,
        DArray2D<Vec3f> batch_pos,
        DArray2D<Mat3f> batch_rot,
        DArray2D<Vec3f> subtree_com,
        DArray2D<Vec3f> joint_anchor,
        DArray2D<Vec3f> joint_axis,
        int num_envs,
        int total_bodies)
    {
        using Real = typename TDataType::Real;

        const int global_bid = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_bid >= total_bodies)
            return;

        const int env_id = BinarySearchLEArray(env_body_offsets, num_envs, global_bid);
        if (env_id < 0 || env_id >= num_envs)
            return;

        const int env_body_begin = env_body_offsets[env_id];
        const int env_self_bodies = batch_bodies[env_id];
        if (global_bid >= env_body_begin + env_self_bodies)
            return;

        const int bid = global_bid - env_body_begin;
        const int pid = parent_idx(env_id, bid);
        const Vec3f& pos = batch_pos(env_id, bid);
        const Mat3f rot = batch_rot(env_id, bid);
        const Vec3f& body_subtree_com = subtree_com(env_id, bid);
        const int q_start = q_offset(env_id, bid);

        if (pid != -1)
        {
            const Vec3f offset = subtree_com(env_id, root_idx(env_id, bid)) - joint_anchor(env_id, bid);
            const int jtype = joint_type(env_id, bid);
            const Vec3f& axis = joint_axis(env_id, bid);

            if (jtype == 1)     // Hinge
            {
                const Vec3f trans_part = cross(axis, offset);
                for (int i = 0; i < 3; ++i)
                {
                    batch_cdof(env_id, q_start * 6 + i) = axis[i];
                    batch_cdof(env_id, q_start * 6 + 3 + i) = trans_part[i];
                }
            }
            else if (jtype == 2)    // Slide
            {
                for (int i = 0; i < 3; ++i)
                {
                    batch_cdof(env_id, q_start * 6 + i) = Real(0);
                    batch_cdof(env_id, q_start * 6 + 3 + i) = axis[i];
                }
            }
            else                    // Ball
            {
                for (int i = 0; i < 3; ++i)
                {
                    const Vec3f rot_axis = rot.col(i);
                    const Vec3f trans_part = cross(rot_axis, offset);
                    for (int j = 0; j < 3; ++j)
                    {
                        batch_cdof(env_id, (q_start + i) * 6 + j) = rot_axis[j];
                        batch_cdof(env_id, (q_start + i) * 6 + 3 + j) = trans_part[j];
                    }
                }
            }
        }
        else
        {
            if (is_static(env_id, bid))
                return;

            for (int i = 0; i < 3; ++i)
                batch_cdof(env_id, (q_start + i) * 6 + 3 + i) = Real(1);

            const Vec3f offset = body_subtree_com - pos;
            for (int i = 0; i < 3; ++i)
            {
                const Vec3f rot_axis = rot.col(i);
                const Vec3f trans_part = cross(rot_axis, offset);
                for (int j = 0; j < 3; ++j)
                {
                    batch_cdof(env_id, (q_start + i + 3) * 6 + 3 + j) = trans_part[j];
                    batch_cdof(env_id, (q_start + i + 3) * 6 + j) = rot_axis[j];
                }
            }
        }
    }

    template<typename TDataType>
    __global__ void AccumulateSubtreeInertialKernel(
        DevBlockVector<Pair<int, int>> groups,
        DArray2D<int> parent_idx,
        DArray2D<typename TDataType::Real> batch_crb,
        int num_envs,
        int num_groups)
    {
        int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        int env_id = BinarySearchLEArray(groups.Offsets(), num_envs, group_id);
        if (env_id < 0 || env_id >= num_envs)
            return;

        const int group_offset = groups.Offsets()[env_id];
        const int group_size = groups.Sizes()[env_id];
        const int local_gid = group_id - group_offset;
        if (local_gid < 0 || local_gid >= group_size)
            return;

        const auto group = groups[env_id][local_gid];
        const int group_begin = group.first;
        const int group_count = group.second;
        const int group_end = group_begin + group_count;

        for (int local = group_count - 1; local >= 0; --local)
        {
            const int bid = group_begin + local;
            const int pid = parent_idx(env_id, bid);
            if (pid >= group_begin && pid < group_end)
            {
                for (int i = 0; i < 10; ++i)
                    batch_crb(env_id, pid * 10 + i) += batch_crb(env_id, bid * 10 + i);
            }
        }
    }

    template<typename TArray>
    __device__ inline int FindEnvFromGlobalNvNaive(
        const TArray& batch_nv,
        int num_envs,
        int global_nvid,
        int& local_nvid)
    {
        local_nvid = global_nvid;
        for (int env_id = 0; env_id < num_envs; ++env_id)
        {
            const int nv = batch_nv[env_id];
            if (local_nvid < nv)
                return env_id;
            local_nvid -= nv;
        }
        return -1;
    }

    template<typename TDataType>
    __device__ inline int FindBodyFromLocalNvNaive(
        const DArray<int>& batch_bodies,
        const DArray2D<int>& q_offset,
        const DArray2D<int>& q_lengths,
        int env_id,
        int local_nvid,
        int& local_dof)
    {
        const int num_bodies = batch_bodies[env_id];
        for (int bid = 0; bid < num_bodies; ++bid)
        {
            const int q_begin = q_offset(env_id, bid);
            const int q_num = q_lengths(env_id, bid);
            if (local_nvid >= q_begin && local_nvid < q_begin + q_num)
            {
                local_dof = local_nvid - q_begin;
                return bid;
            }
        }
        local_dof = -1;
        return -1;
    }

    template<typename Real>
    __device__ inline void InertiaMultiVecTemp(
        const DArray2D<Real>& inertia,
        const Real* vec,
        Real* res,
        int env_id,
        int bid)
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

    template<typename T>
    __device__ inline void ComputeComVelTemp(
        const DArray2D<T>& cdof,
        const DArray2D<T>& qvel,
        DArray2D<T>& com_vel,
        int env_id,
        int bid,
        int q_start,
        int offset)
    {
        for (int r = 0; r < 6; r++)
        {
            T sum = T(0);
            for (int c = 0; c < 3; c++)
                sum += cdof(env_id, (q_start + offset + c) * 6 + r) * qvel(env_id, q_start + offset + c);
            com_vel(env_id, bid * 6 + r) += sum;
        }
    }

    template<typename T>
    __device__ inline void ComputeCVelCrossTemp(
        const DArray2D<T>& cdof,
        const DArray2D<T>& com_vel,
        DArray2D<T>& cdof_dot,
        int env_id,
        int bid,
        int q_start,
        int offset)
    {
        const int idx = (q_start + offset) * 6;

        const T& cvel_0 = com_vel(env_id, bid * 6);
        const T& cvel_1 = com_vel(env_id, bid * 6 + 1);
        const T& cvel_2 = com_vel(env_id, bid * 6 + 2);
        const T& cvel_3 = com_vel(env_id, bid * 6 + 3);
        const T& cvel_4 = com_vel(env_id, bid * 6 + 4);
        const T& cvel_5 = com_vel(env_id, bid * 6 + 5);

        const T& cdof_0 = cdof(env_id, idx);
        const T& cdof_1 = cdof(env_id, idx + 1);
        const T& cdof_2 = cdof(env_id, idx + 2);
        const T& cdof_3 = cdof(env_id, idx + 3);
        const T& cdof_4 = cdof(env_id, idx + 4);
        const T& cdof_5 = cdof(env_id, idx + 5);

        cdof_dot(env_id, idx)     = -cvel_2 * cdof_1 + cvel_1 * cdof_2;
        cdof_dot(env_id, idx + 1) =  cvel_2 * cdof_0 - cvel_0 * cdof_2;
        cdof_dot(env_id, idx + 2) = -cvel_1 * cdof_0 + cvel_0 * cdof_1;
        cdof_dot(env_id, idx + 3) = -cvel_2 * cdof_4 + cvel_1 * cdof_5 - cvel_5 * cdof_1 + cvel_4 * cdof_2;
        cdof_dot(env_id, idx + 4) =  cvel_2 * cdof_3 - cvel_0 * cdof_5 + cvel_5 * cdof_0 - cvel_3 * cdof_2;
        cdof_dot(env_id, idx + 5) = -cvel_1 * cdof_3 + cvel_0 * cdof_4 - cvel_4 * cdof_0 + cvel_3 * cdof_1;
    }

    template<typename TDataType>
    __global__ void ComputeComVelKernel(
        DevBlockVector<Pair<int, int>> groups,
        DArray2D<int> parent_idx,
        DArray2D<int> q_offset,
        DArray2D<int> joint_type,
        DArray2D<int> is_static,
        DArray2D<typename TDataType::Real> batch_cdof,
        DArray2D<typename TDataType::Real> batch_qvel,
        DArray2D<typename TDataType::Real> subtree_com_vel,
        DArray2D<typename TDataType::Real> batch_cdof_dot,
        int num_envs,
        int num_groups)
    {
        int group_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (group_id >= num_groups)
            return;

        int env_id = BinarySearchLEArray(groups.Offsets(), num_envs, group_id);
        if (env_id < 0 || env_id >= num_envs)
            return;

        const int group_offset = groups.Offsets()[env_id];
        const int group_size = groups.Sizes()[env_id];
        const int local_gid = group_id - group_offset;
        if (local_gid < 0 || local_gid >= group_size)
            return;

        const auto group = groups[env_id][local_gid];
        const int group_begin = group.first;
        const int group_count = group.second;

        for (int local = 0; local < group_count; ++local)
        {
            const int bid = group_begin + local;
            const int pid = parent_idx(env_id, bid);
            const int q_start_local = q_offset(env_id, bid);

            if (pid != -1)
            {
                const int jtype = joint_type(env_id, bid);
                if (jtype < 3)      // Hinge or Slide
                {
                    for (int i = 0; i < 6; i++)
                        subtree_com_vel(env_id, bid * 6 + i) =
                            subtree_com_vel(env_id, pid * 6 + i) +
                            batch_qvel(env_id, q_start_local) * batch_cdof(env_id, q_start_local * 6 + i);

                    ComputeCVelCrossTemp(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start_local, 0);
                }
                else                // Ball
                {
                    for (int i = 0; i < 6; i++)
                        subtree_com_vel(env_id, bid * 6 + i) = subtree_com_vel(env_id, pid * 6 + i);

                    for (int i = 0; i < 3; i++)
                        ComputeCVelCrossTemp(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start_local, i);

                    ComputeComVelTemp(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start_local, 0);
                }
            }
            else
            {
                if (is_static(env_id, bid))
                    continue;

                ComputeComVelTemp(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start_local, 0);

                for (int i = 0; i < 3; i++)
                    ComputeCVelCrossTemp(batch_cdof, subtree_com_vel, batch_cdof_dot, env_id, bid, q_start_local, 3 + i);

                ComputeComVelTemp(batch_cdof, batch_qvel, subtree_com_vel, env_id, bid, q_start_local, 3);
            }
        }
    }

    template<typename TDataType>
    __global__ void UpdateGeneralizedInertialMatrixKernel(
        DArray<int> batch_bodies,
        DArray<int> batch_nv,
        DArray2D<int> is_static,
        DArray2D<int> parent_idx,
        DArray2D<int> q_offset,
        DArray2D<int> q_lengths,
        DArray2D<typename TDataType::Real> batch_cdof,
        DArray2D<typename TDataType::Real> batch_crb,
        DArray2D<typename TDataType::Real> batch_qM,
        int num_envs,
        int total_nv)
    {
        using Real = typename TDataType::Real;

        const int global_nvid = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_nvid >= total_nv)
            return;

        int local_nvid = -1;
        const int env_id = FindEnvFromGlobalNvNaive(batch_nv, num_envs, global_nvid, local_nvid);
        if (env_id < 0 || env_id >= num_envs)
            return;

        int local_dof = -1;
        const int bid = FindBodyFromLocalNvNaive<TDataType>(
            batch_bodies, q_offset, q_lengths, env_id, local_nvid, local_dof);
        if (bid < 0)
            return;

        const int nv = batch_nv[env_id];
        const int q_start = q_offset(env_id, bid);
        const int q_num = q_lengths(env_id, bid);
        if (local_nvid < q_start || local_nvid >= q_start + q_num)
            return;

        const int static_flag = is_static(env_id, bid);
        if (static_flag)
            return;

        Real tmp_dof[6];
        Real Icdof[6];
        for (int i = 0; i < 6; ++i)
            tmp_dof[i] = batch_cdof(env_id, local_nvid * 6 + i);

        InertiaMultiVecTemp(batch_crb, tmp_dof, Icdof, env_id, bid);

        int upper_q = local_nvid;
        int cur_body = bid;
        // follow q chain update
        while (cur_body != -1)
        {
            const int q_begin = q_offset(env_id, cur_body);
            for (int i1 = upper_q; i1 >= q_begin; --i1)
            {
                Real val = Real(0);
                for (int n = 0; n < 6; ++n)
                    val += batch_cdof(env_id, i1 * 6 + n) * Icdof[n];

                // q-parallel version: write lower triangle only and mirror later.
                batch_qM(env_id, local_nvid * nv + i1) = val;
            }

            cur_body = parent_idx(env_id, cur_body);
            if (cur_body != -1)
                upper_q = q_offset(env_id, cur_body) + q_lengths(env_id, cur_body) - 1;
        }

    }

    template<typename TDataType>
    __global__ void SymmetrizeGeneralizedInertialMatrixKernel(
        DArray<int> batch_nv,
        DArray2D<typename TDataType::Real> batch_qM,
        int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (env_id >= num_envs)
            return;

        const int nv = batch_nv[env_id];
        for (int i = 0; i < nv; ++i)
        {
            for (int j = 0; j < i; ++j)
                batch_qM(env_id, j * nv + i) = batch_qM(env_id, i * nv + j);
        }
    }

}
namespace host_interface
{

    template<typename TDataType>
    void BuildGroupsTemp(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        // Step 1: temporary init, only to make Sizes()/Offsets()/Data() valid
        rigid_body_system.groups.BuildFromSizes(std::vector<int>(num_envs, 1));

        const int threads = 128;
        const int blocks = (num_envs + threads - 1) / threads;
        // Step 2: count groups on GPU, write counts into groups.Sizes()
        device_kernel::CountGroupsKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.batch_bodies,
            rigid_body_system.parent_idx,
            rigid_body_system.groups.Sizes(),
            num_envs);
        cudaDeviceSynchronize();

        // Step 3: download temporary sizes
        HostArr<int> real_group_sizes(num_envs);
        real_group_sizes.Assign(rigid_body_system.groups.Sizes());

        // Step 4: rebuild groups with exact block sizes
        rigid_body_system.groups.BuildFromSizes(real_group_sizes);

        // Step 5: fill actual <begin, count> pairs
        device_kernel::BuildGroupsKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.batch_bodies,
            rigid_body_system.parent_idx,
            rigid_body_system.groups,
            num_envs);
        cudaDeviceSynchronize();

    }

    template<typename TDataType>
    void ForwardKinematicsHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        BuildGroupsTemp(rigid_body_system, num_envs);

        int num_groups = rigid_body_system.groups.TotalSize();

        // printf("num_groups: %d\n", num_groups);

        const int threads = 128;
        const int blocks = (num_groups + threads - 1) / threads;
        
        device_kernel::ForwardKinematicsKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.groups,
            rigid_body_system.parent_idx,
            rigid_body_system.batch_quat,
            rigid_body_system.batch_rot,
            rigid_body_system.batch_pos,
            rigid_body_system.joint_axis_ref,
            rigid_body_system.joint_anchor_ref,
            rigid_body_system.joint_type,
            rigid_body_system.joint_qpos,
            rigid_body_system.joint_qpos_ref,
            rigid_body_system.joint_qpos_offset,
            rigid_body_system.joint_rel_pos,
            rigid_body_system.joint_rel_quat,
            rigid_body_system.joint_axis,
            rigid_body_system.joint_anchor,
            num_envs,
            num_groups);
    
    }

    template<typename TDataType>
    void SubtreeComHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        const int num_groups = rigid_body_system.groups.TotalSize();
        if (num_groups <= 0)
            return;

        const int threads = 128;
        const int blocks = (num_groups + threads - 1) / threads;

        device_kernel::SubtreeComGroupKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.groups,
            rigid_body_system.parent_idx,
            rigid_body_system.subtree_com,
            rigid_body_system.batch_mass,
            rigid_body_system.subtree_mass,
            rigid_body_system.batch_pos,
            num_envs,
            num_groups);
    }

    template<typename TDataType>
    void ComputeCdofHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        CArray<int> batch_bodies_host(num_envs);
        batch_bodies_host.assign(rigid_body_system.batch_bodies);

        int total_bodies = 0;
        for (int env_id = 0; env_id < num_envs; ++env_id)
        {
            total_bodies += batch_bodies_host[env_id];
        }

        if (total_bodies <= 0)
            return;


        const int threads = 128;
        const int blocks = (total_bodies + threads - 1) / threads;
        device_kernel::ComputeCdofKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.batch_bodies,
            rigid_body_system.batch_body_offset,
            rigid_body_system.parent_idx,
            rigid_body_system.root_idx,
            rigid_body_system.is_static,
            rigid_body_system.joint_type,
            rigid_body_system.q_offset,
            rigid_body_system.batch_cdof,
            rigid_body_system.batch_pos,
            rigid_body_system.batch_rot,
            rigid_body_system.subtree_com,
            rigid_body_system.joint_anchor,
            rigid_body_system.joint_axis,
            num_envs,
            total_bodies);
    }

    template<typename TDataType>
    void SubtreeInertial(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        CArray<int> batch_bodies_host(num_envs);
        batch_bodies_host.assign(rigid_body_system.batch_bodies);

        int total_bodies = 0;
        for (int env_id = 0; env_id < num_envs; ++env_id)
            total_bodies += batch_bodies_host[env_id];

        if (total_bodies <= 0)
            return;

        const int threads = 128;
        const int blocks = (total_bodies + threads - 1) / threads;
        device_kernel::SubtreeInertialKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.batch_bodies,
            rigid_body_system.batch_body_offset,
            rigid_body_system.root_idx,
            rigid_body_system.batch_pos,
            rigid_body_system.subtree_com,
            rigid_body_system.batch_inertia,
            rigid_body_system.batch_rot,
            rigid_body_system.batch_mass,
            rigid_body_system.subtree_inertia,
            rigid_body_system.batch_crb,
            num_envs,
            total_bodies);
    }

    template<typename TDataType>
    void AccumulateSubtreeInertialHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        const int num_groups = rigid_body_system.groups.TotalSize();
        if (num_groups <= 0)
            return;

        const int threads = 128;
        const int blocks = (num_groups + threads - 1) / threads;
        device_kernel::AccumulateSubtreeInertialKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.groups,
            rigid_body_system.parent_idx,
            rigid_body_system.batch_crb,
            num_envs,
            num_groups);
    }

    template<typename TDataType>
    void UpdateGeneralizedInertialMatrixHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        CArray<int> batch_nv_host(num_envs);
        batch_nv_host.assign(rigid_body_system.batch_nv);

        int total_nv = 0;
        for (int env_id = 0; env_id < num_envs; ++env_id)
            total_nv += batch_nv_host[env_id];

        if (total_nv <= 0)
            return;

        const int threads = 128;
        const int blocks = (total_nv + threads - 1) / threads;
        device_kernel::UpdateGeneralizedInertialMatrixKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.batch_bodies,
            rigid_body_system.batch_nv,
            rigid_body_system.is_static,
            rigid_body_system.parent_idx,
            rigid_body_system.q_offset,
            rigid_body_system.q_lengths,
            rigid_body_system.batch_cdof,
            rigid_body_system.batch_crb,
            rigid_body_system.batch_qM,
            num_envs,
            total_nv);

        const int env_threads = 128;
        const int env_blocks = (num_envs + env_threads - 1) / env_threads;
        device_kernel::SymmetrizeGeneralizedInertialMatrixKernel<TDataType><<<env_blocks, env_threads>>>(
            rigid_body_system.batch_nv,
            rigid_body_system.batch_qM,
            num_envs);
    }

    template<typename TDataType>
    void ComputeComVelHost(RigidBody<TDataType>& rigid_body_system, int num_envs)
    {
        const int num_groups = rigid_body_system.groups.TotalSize();
        if (num_groups <= 0)
            return;

        const int threads = 128;
        const int blocks = (num_groups + threads - 1) / threads;
        device_kernel::ComputeComVelKernel<TDataType><<<blocks, threads>>>(
            rigid_body_system.groups,
            rigid_body_system.parent_idx,
            rigid_body_system.q_offset,
            rigid_body_system.joint_type,
            rigid_body_system.is_static,
            rigid_body_system.batch_cdof,
            rigid_body_system.batch_qvel,
            rigid_body_system.subtree_com_vel,
            rigid_body_system.batch_cdof_dot,
            num_envs,
            num_groups);
    }


}


}
