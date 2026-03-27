#include "MujocoSolver.h"
#include <spdlog/spdlog.h>
#include <thrust/device_ptr.h>
#include "../../Utils/utils.h"
#include "Algorithm.h"
#include <Eigen/Dense>

#define NV_TMP 256

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
                    qrot = QuatFromAxisAngle(axis, angle);
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
            else
            {
                const int& joint_type = rigid_body_system.joint_type(env_id, bid);
                if(joint_type < 3)  // Hinge or Slide
                    qpos(env_id, qpos_start) += qvel(env_id, q_start) * dt;
                else
                {
                    Vec3f w = Vec3f(qvel(env_id, q_start), qvel(env_id, q_start + 1), qvel(env_id, q_start + 2));
                    Quat<Real> quat = Quat<Real>(qpos(env_id, qpos_start), qpos(env_id, qpos_start + 1), qpos(env_id, qpos_start + 2), qpos(env_id, qpos_start + 3));
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
                    qpos(env_id, qpos_start) = quat_new.x;
                    qpos(env_id, qpos_start + 1) = quat_new.y;
                    qpos(env_id, qpos_start + 2) = quat_new.z;
                    qpos(env_id, qpos_start + 3) = quat_new.w;
                }

            }  
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
    __global__ void InitInertiaKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        auto& inertia = rigid_body_system.batch_inertia;
        const auto& shape_type = rigid_body_system.shape_type;
        const auto& shape_idx = rigid_body_system.shape_idx;
        const auto& mass = rigid_body_system.batch_mass;
        const auto& spheres = rigid_body_system.spheres;
        const auto& boxes = rigid_body_system.boxes;
        const auto& capsules = rigid_body_system.capsules;

        for(int bid = 0; bid < env_self_bodies; bid++) {
            Real body_mass = mass(env_id, bid);

            switch(shape_type(env_id, bid)) {
                case 0: {
                    Real radius = spheres(env_id, shape_idx(env_id, bid)).radius;
                    Real I = (2.0f / 5.0f) * body_mass * radius * radius;
                    inertia(env_id, bid).x = I;
                    inertia(env_id, bid).y = I;
                    inertia(env_id, bid).z = I;
                    break;
                }

                case 1: {
                    Vec3f halfSize = boxes(env_id, shape_idx(env_id, bid)).halfLength;
                    inertia(env_id, bid).x = (body_mass / 3.f) * (halfSize.y * halfSize.y + halfSize.z * halfSize.z);
                    inertia(env_id, bid).y = (body_mass / 3.f) * (halfSize.x * halfSize.x + halfSize.z * halfSize.z);
                    inertia(env_id, bid).z = (body_mass / 3.f) * (halfSize.x * halfSize.x + halfSize.y * halfSize.y);
                    break;
                }

                case 2: {
                    Real radius = capsules(env_id, shape_idx(env_id, bid)).radius;
                    Real halfLength = capsules(env_id, shape_idx(env_id, bid)).halfLength;
                    Real sphere_mass = 4.f * body_mass * radius / (4.f * radius + 6.f * halfLength);
                    Real cylinder_mass = body_mass - sphere_mass;
                    Real sphere_inertia = 2.f / 5.f * sphere_mass * radius * radius;

                    inertia(env_id, bid).x = cylinder_mass * (3.f * radius * radius + 4.f * halfLength * halfLength) / 12.f
                                                + sphere_inertia + sphere_mass * halfLength * (3.f * radius + 4.f * halfLength) / 4.f;
                    inertia(env_id, bid).y = inertia(env_id, bid).x;
                    inertia(env_id, bid).z = cylinder_mass * radius * radius / 2.f + sphere_inertia;
                    break;
                }
                default: ;
            }
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
        auto& quat_world = rigid_body_system.batch_quat;
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
                const auto& parent_quat = quat_world(env_id, parent_idx);
                const auto& parent_rot = rot_world(env_id, parent_idx);
                const auto& local_axis = rigid_body_system.joint_axis_ref(env_id, bid);
                const auto& local_anchor = rigid_body_system.joint_anchor_ref(env_id, bid);
                const int& joint_type = rigid_body_system.joint_type(env_id, bid);
                const auto& joint_qpos = rigid_body_system.joint_qpos;
                const auto& joint_qpos0 = rigid_body_system.joint_qpos_ref;
                const auto& joint_qpos_start = rigid_body_system.joint_qpos_offset(env_id, bid);
                auto& pos = rigid_body_system.batch_pos;
                

                Quat<Real> xquat_p = parent_quat * rigid_body_system.joint_rel_quat(env_id, bid);
                rigid_body_system.joint_axis(env_id, bid) = RotateVector(local_axis, xquat_p);
                Vec3f xanchor = RotateVector(local_anchor, xquat_p);
                Vec3f xpos = parent_rot * rigid_body_system.joint_rel_pos(env_id, bid) + pos(env_id, parent_idx);
                xanchor += xpos;
                rigid_body_system.joint_anchor(env_id, bid) = xanchor;


                if(joint_type == 2)     // Slide
                {
                    quat_world(env_id, bid) = xquat_p;
                    rot_world(env_id, bid) = xquat_p.toMatrix3x3();
                    pos(env_id, bid) = xpos + (joint_qpos(env_id, joint_qpos_start) - joint_qpos0(env_id, joint_qpos_start)) * rigid_body_system.joint_axis(env_id, bid);
                }
                else
                {
                    Quat<Real> quat_local;
                    if(joint_type == 1)     // Hinge
                        quat_local = QuatFromAxisAngle(local_axis, joint_qpos(env_id, joint_qpos_start) - joint_qpos0(env_id, joint_qpos_start));
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
                    quat_world(env_id, bid) = xquat_c_bak;
                    rot_world(env_id, bid) = xquat_c.toMatrix3x3();
                    xpos = RotateVector(local_anchor, xquat_c);
                    pos(env_id, bid) = xanchor - xpos;

                    printf("body: %d\n", bid);
                    printf("quat_local: %f, %f, %f, %f\n", quat_local.x, quat_local.y, quat_local.z, quat_local.w);
                    printf("xquat_c: %f, %f, %f, %f\n", xquat_c.x, xquat_c.y, xquat_c.z, xquat_c.w);
                    printf("local_anchor: %f, %f, %f\n", local_anchor.x, local_anchor.y, local_anchor.z);
                    printf("xanchor: %f, %f, %f\n", xanchor.x, xanchor.y, xanchor.z);
                    printf("xpos: %f, %f, %f\n", xpos.x, xpos.y, xpos.z);

                    printf("parent quat: %f, %f, %f, %f\n", parent_quat.x, parent_quat.y, parent_quat.z, parent_quat.w);
                    printf("joint_rel_quat: %f, %f, %f, %f\n", rigid_body_system.joint_rel_quat(env_id, bid).x, rigid_body_system.joint_rel_quat(env_id, bid).y, rigid_body_system.joint_rel_quat(env_id, bid).z, rigid_body_system.joint_rel_quat(env_id, bid).w);
                }
            }
            printf("body %d\n", bid);
            printf("global frame pos: %f, %f, %f\n", rigid_body_system.batch_pos(env_id, bid).x, rigid_body_system.batch_pos(env_id, bid).y, rigid_body_system.batch_pos(env_id, bid).z);
            printf("global frame quat: %f, %f, %f, %f\n", rigid_body_system.batch_quat(env_id, bid).x, rigid_body_system.batch_quat(env_id, bid).y, rigid_body_system.batch_quat(env_id, bid).z, rigid_body_system.batch_quat(env_id, bid).w);
            printf("global axis: %f, %f, %f\n", rigid_body_system.joint_axis(env_id, bid).x, rigid_body_system.joint_axis(env_id, bid).y, rigid_body_system.joint_axis(env_id, bid).z);
            printf("global anchor: %f, %f, %f\n\n", rigid_body_system.joint_anchor(env_id, bid).x, rigid_body_system.joint_anchor(env_id, bid).y, rigid_body_system.joint_anchor(env_id, bid).z);
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
        const auto& mass = rigid_body_system.batch_mass;
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
            Vec3f offset = rigid_body_system.subtree_com(env_id, rigid_body_system.root_idx(env_id, bid)) - rigid_body_system.joint_anchor(env_id, bid);
            const int& joint_type = rigid_body_system.joint_type(env_id, bid);

            const Vec3f& joint_axis = rigid_body_system.joint_axis(env_id, bid);

            if(joint_type == 1)     // Hinge
            {
                Vec3f trans_part = cross(joint_axis, offset);
                for(int i = 0; i < 3; i++)
                {
                    cdof(env_id, q_start * 6 + i) = joint_axis[i];
                    cdof(env_id, q_start * 6 + 3 + i) = trans_part[i];
                }
            }
            else if (joint_type == 2)   // Slide
            {
                for(int i = 0; i < 3; i++)
                {
                    cdof(env_id, q_start * 6 + i) = 0.f;
                    cdof(env_id, q_start * 6 + 3 + i) = joint_axis[i];
                }
            }
            else                    // Ball
            {
                for(int i = 0; i < 3; i++)
                {
                    Vec3f rot_axis = rot.col(i);
                    Vec3f trans_part = cross(rot_axis, offset);
                    for(int j = 0; j < 3; j++)
                    {
                        cdof(env_id, (q_start + i) * 6 + j) = rot_axis[j];
                        cdof(env_id, (q_start + i) * 6 + 3 + j) = trans_part[j];
                    }
                }
            }
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

    __device__ void SubtreeComInertia(DArray2D<Real>& com_inertia, const Vec3f& body_inertia, const Mat3f& rot_mat, const Vec3f& offset, Real mass, int env_id, int bid)
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
    __global__ void SubtreeInertialKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];
        int bid = threadIdx.x;
        if(bid >= env_self_bodies)
            return;

        const int& root_idx = rigid_body_system.root_idx(env_id, bid);
        Vec3f offset = rigid_body_system.batch_pos(env_id, bid) - rigid_body_system.subtree_com(env_id, root_idx);
        auto& subtree_inertia = rigid_body_system.subtree_inertia;
        const Vec3f& body_inertia = rigid_body_system.batch_inertia(env_id, bid);
        const auto& rot_mat = rigid_body_system.batch_rot(env_id, bid);
        const Real& mass = rigid_body_system.batch_mass(env_id, bid);

        SubtreeComInertia(subtree_inertia, body_inertia, rot_mat, offset, mass, env_id, bid);

        for(int i = 0; i < 10; i++)
            rigid_body_system.batch_crb(env_id, bid * 10 + i) = subtree_inertia(env_id, bid * 10 + i);
    }

    template<typename TDataType>
    __global__ void AccumulateSubtreeInertialKernel(RigidBody<TDataType> rigid_body_system, int num_env)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_env)
            return;

        for(int bid = rigid_body_system.batch_bodies[env_id] - 1; bid >= 0; bid--)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            if(parent_idx != -1)
                for(int i = 0; i < 10; i++)
                    rigid_body_system.batch_crb(env_id, parent_idx * 10 + i) += rigid_body_system.batch_crb(env_id, bid * 10 + i);

        }


        for(int bid = 0; bid < rigid_body_system.batch_bodies[env_id]; bid++)
        {
            printf("Env %d, Body %d, Composite Rigid Body Inertia:\n", env_id, bid);

            printf("ComInertial: \n");
            for(int i = 0; i < 10; i++)
                printf("%f\t", rigid_body_system.subtree_inertia(env_id, bid * 10 + i));
            printf("\n");

            printf("CRB: \n");
            for(int i = 0; i < 10; i++)
                printf("%f\t", rigid_body_system.batch_crb(env_id, bid * 10 + i));
            printf("\n");
            
        }
    }

    __device__ void InertiaMultiVec(const DArray2D<Real>& inertia, const Real* vec, Real* res, int env_id, int bid)
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
    __global__ void UpdateGeneralizedInertialMatrixKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        int env_self_bodies = rigid_body_system.batch_bodies[env_id];    
        auto& batch_qM = rigid_body_system.batch_qM;
        const auto& inertia = rigid_body_system.batch_inertia;
        const auto& mass = rigid_body_system.batch_mass;
        const int nv = rigid_body_system.batch_nv[env_id];

        for(int bid = 0; bid < env_self_bodies; bid++)
        {
            const int& is_isolated = rigid_body_system.is_isolated(env_id, bid);
            const int is_static = rigid_body_system.is_static(env_id, bid);
            if(is_static)
                continue;

            const int parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int q_start = rigid_body_system.q_offset(env_id, bid);
            const int q_num = rigid_body_system.q_lengths(env_id, bid);

            if(!is_isolated)
            {
                const auto& cdof = rigid_body_system.batch_cdof;
                auto& q_chain = rigid_body_system.batch_q_chain;
                Real tmp_dof[6];
                Real Icdof[6];
                for(int qidx = q_start; qidx < q_start + q_num; qidx++)
                {
                    for(int i = 0; i < 6; i++)  // tmp_dof ← cdof[q_index]
                        tmp_dof[i] = cdof(env_id, qidx * 6 + i);

                    InertiaMultiVec(rigid_body_system.batch_crb, tmp_dof, Icdof, env_id, bid); // Icdof ← inertia_multi_vec(tmp_crb, tmp_dof)
                    
                    int i = qidx;
                    int j = bid;
                    int q_chain_length = 0;
                    while(j != -1)
                    {
                        int qidx_j = rigid_body_system.q_offset(env_id, j);
                        for(int k = i; k >= qidx_j; k--)
                        {
                            q_chain(env_id, q_chain_length) = k;
                            q_chain_length++;
                        }

                        j = rigid_body_system.parent_idx(env_id, j);
                        if(j != -1)
                            i = rigid_body_system.q_offset(env_id, j) + rigid_body_system.q_lengths(env_id, j) - 1;
                    }
                    
                    for(int m = 0; m < q_chain_length; m++)
                    {
                        int i1 = q_chain(env_id, m);
                        Real val = 0.f;
                        for(int n = 0; n < 6; n++)
                            val += cdof(env_id, i1 * 6 + n) * Icdof[n];

                        batch_qM(env_id, qidx * nv + i1) = val;
                        batch_qM(env_id, i1 * nv + qidx) = val;   // qM is symmetric
                    }
                }
            }
            else
            {
                for(int i = 0; i < 6; i++)
                    for(int j = 0; j < 6; j++)
                        batch_qM(env_id, (q_start + i) * nv + (q_start + j)) = 0.f;
                
                // Trick, cube
                batch_qM(env_id, (q_start + 0) * nv + (q_start + 0)) = mass(env_id, bid);
                batch_qM(env_id, (q_start + 1) * nv + (q_start + 1)) = mass(env_id, bid);
                batch_qM(env_id, (q_start + 2) * nv + (q_start + 2)) = mass(env_id, bid);

                batch_qM(env_id, (q_start + 3) * nv + (q_start + 3)) = inertia(env_id, bid).x;
                batch_qM(env_id, (q_start + 4) * nv + (q_start + 4)) = inertia(env_id, bid).y;
                batch_qM(env_id, (q_start + 5) * nv + (q_start + 5)) = inertia(env_id, bid).z;

            }
        }
        
        printf("\nEnv %d, qM:\n", env_id);
        for(int i = 0; i < nv; i++)
        {
            for(int j = 0; j < nv; j++)
                printf("%f ", batch_qM(env_id, i * nv + j));
            printf("\n");
        }


        
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
    __global__ void UpdateAnchorConstarints(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        if(anchor_idx >= rigid_body_system.num_each_constraint[env_id][0])
            return;
        
        auto& anchor_constraints = rigid_body_system.anchor_constraints;
        const int& body_A_idx = anchor_constraints.body_idxs(env_id, anchor_idx).first;
        const int& body_B_idx = anchor_constraints.body_idxs(env_id, anchor_idx).second;

        const Mat3f& rot_A = rigid_body_system.batch_rot(env_id, body_A_idx);
        const Mat3f& rot_B = rigid_body_system.batch_rot(env_id, body_B_idx);

        const Vec3f& local_anchor_A = anchor_constraints.anchor_A_local(env_id, anchor_idx);
        const Vec3f& local_anchor_B = anchor_constraints.anchor_B_local(env_id, anchor_idx);
        const Vec3f& pos_A = rigid_body_system.batch_pos(env_id, body_A_idx);
        const Vec3f& pos_B = rigid_body_system.batch_pos(env_id, body_B_idx);
        Vec3f& global_anchor_A = anchor_constraints.anchor_A_world(env_id, anchor_idx);
        Vec3f& global_anchor_B = anchor_constraints.anchor_B_world(env_id, anchor_idx);
        Vec3f& pos_err = anchor_constraints.anchor_error(env_id, anchor_idx);

        global_anchor_A = rot_A * local_anchor_A + pos_A;
        global_anchor_B = rot_B * local_anchor_B + pos_B;
        pos_err = global_anchor_A - global_anchor_B;
    }

    template<typename TDataType>
    __global__ void UpdateJointLimitConstraints(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int jl_idx = threadIdx.x;
        auto& joint_limits = rigid_body_system.joint_limit_constraints;
        if(jl_idx >= joint_limits.ref_nums[env_id])
            return;

        const int& bid = joint_limits.joint_idx(env_id, jl_idx);
        const int& joint_type = rigid_body_system.joint_type(env_id, bid);
        const int& is_upper = joint_limits.is_upper(env_id, jl_idx);
        const int& qpos_idx = rigid_body_system.qpos_offset(env_id, bid);
        const auto& qpos = rigid_body_system.batch_qpos;
        const Real& limit = joint_limits.limit(env_id, jl_idx);
        auto& limit_err = joint_limits.limit_error(env_id, jl_idx);
        auto& is_active = joint_limits.is_active(env_id, jl_idx);
        auto& limit_extern = joint_limits.limit_extern(env_id, jl_idx);
        is_active = 0;

        if(joint_type < 3)      // Hinge or Slide
        {
            if(is_upper)
            {
                Real dist = qpos(env_id, qpos_idx) - limit;
                if(dist > 0.f)
                {
                    limit_err = dist;
                    is_active = 1;
                }
            }
            else
            {
                Real dist = qpos(env_id, qpos_idx) - limit;
                if(dist < 0.f)
                {
                    limit_err = dist;
                    is_active = 1;
                }
            }
        }
        else                    // Ball, upper only
        {
            Quat<Real> quat = Quat<Real>(qpos(env_id, qpos_idx), qpos(env_id, qpos_idx + 1), qpos(env_id, qpos_idx + 2), qpos(env_id, qpos_idx + 3));
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

        
    }

    template<typename TDataType>
    __global__ void CountConstraintNums(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockDim.x * blockIdx.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        auto& num_each_constraint = rigid_body_system.num_each_constraint;
        
        const auto& anchors = rigid_body_system.anchor_constraints;
        const auto& friction_losses = rigid_body_system.friction_loss_constraints;
        auto& joint_limits = rigid_body_system.joint_limit_constraints;
        auto& collisions = rigid_body_system.collision_constraints;


        // count joint limit constraints
        int active_jl_num = 0;
        for(int i = 0; i < joint_limits.ref_nums[env_id]; i++)
        {
            if(joint_limits.is_active(env_id, i))
            {
                joint_limits.active_mapping(env_id, active_jl_num) = i;
                active_jl_num++;
            }
        }

        num_each_constraint[env_id][2] = active_jl_num;
        num_each_constraint[env_id][3] = collisions.collision_nums[env_id] * 4;
        
        auto& offsets = rigid_body_system.constraint_offset[env_id];
        auto& num_constraints = rigid_body_system.num_constraints[env_id];

        num_constraints = num_each_constraint[env_id][0] + num_each_constraint[env_id][1];
        for(int i = 2; i < 4; i++)
        {
            offsets[i] = offsets[i - 1] + num_each_constraint[env_id][i - 1];
            num_constraints += num_each_constraint[env_id][i];
        }


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
    __device__ void ComputeJac(Real* dst_jac, const Vec3f& c_point, const RigidBody<TDataType>& rigid_body_system, int env_id, int bid, int cidx)
    {
        const int root_idx = rigid_body_system.root_idx(env_id, bid);
        Vec3f offset = c_point - rigid_body_system.subtree_com(env_id, root_idx);

        const auto& cdof = rigid_body_system.batch_cdof;
        // const int jac_offset = cidx * 6;
        const int nv = rigid_body_system.batch_nv[env_id];

        // Always clear the local Jacobian buffer first. ComputeJac only writes
        // a kinematic chain subset of columns, so untouched columns must be zero.
        for(int i = 0; i < 6 * nv; i++)
            dst_jac[i] = 0.f;

        
        if(rigid_body_system.is_static(env_id, bid))
            return;
            
        int j = bid;
        while(j != -1)
        {
            int q_start = rigid_body_system.q_offset(env_id, j);
            int q_num = rigid_body_system.q_lengths(env_id, j);

            for(int k = q_num - 1; k >= 0; k--)
            {
                int q_idx = q_start + k;
                Vec3f cdof_angular = Vec3f(cdof(env_id, q_idx * 6), cdof(env_id, q_idx * 6 + 1), cdof(env_id, q_idx * 6 + 2));
                Vec3f d = cross(cdof_angular, offset);
                for(int i = 0; i < 6; i++)
                    dst_jac[i * nv + q_idx] = cdof(env_id, q_idx * 6 + i);
                for(int i = 3; i < 6; i++)
                    dst_jac[i * nv + q_idx] += d[i - 3];
            }
            j = rigid_body_system.parent_idx(env_id, j);
        }
    }

    template<typename TDataType>
    __global__ void ContactConstraintJacobianKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
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


        Real jacA[6 * NV_TMP];
        Real jacB[6 * NV_TMP];

        ComputeJac(jacA, c_point, rigid_body_system, env_id, a_idx, cidx);
        if(b_idx != -1)
            ComputeJac(jacB, c_point, rigid_body_system, env_id, b_idx, cidx);
        else
        {
            for(int i = 0; i < 6 * num_nv; i++)
                jacB[i] = 0.f;
        }

        int test_idx = 0;
        if(cidx == test_idx)
        {
            printf("collision point: (%f, %f, %f)\n", c_point.x, c_point.y, c_point.z);
            for(int i = 0; i < 6; i++)
            {
                printf("JacA: ");
                for(int j = 0; j < num_nv; j++)
                    printf("%f\t", jacA[i * num_nv + j]);
                printf("\n");
            }
        }
       

        // compute relative linear jacobian: J = JacA_linear - JacB_linear
        for(int j = 0; j < 3; j++)
            for(int k = 0; k < num_nv; k++)
                jacA[j * num_nv + k] = jacA[(j + 3) * num_nv + k] - jacB[(j + 3) * num_nv + k];     // now, jacA top 3-rows js jacp

        // convert to contact frame

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < num_nv; j++)
            {
                Real sum = 0;
                for(int k = 0; k < 3; k++)
                    sum += c_basis(k, i) * jacA[k * num_nv + j];
                jacB[i * num_nv + j] = sum;
            }

        if(cidx == test_idx)
        {
            for(int i = 0; i < 3; i++)
            {
                printf("JacDif: ");
                for(int j = 0; j < num_nv; j++)
                    printf("%f\t", jacB[i * num_nv + j]);
                printf("\n");
            }
        }

        const Real mu = collisions.mu(env_id, cidx);
        const int row0 = rigid_body_system.constraint_offset[env_id][3] + 4 * cidx;
        for(int i = 0; i < num_nv; i++)
        {
            const Real j0j = jacB[i];
            const Real j1j = jacB[num_nv + i];
            const Real j2j = jacB[2 * num_nv + i];

            MatrixAt(J, env_id, row0 + 0, 0, i, Vec2i(1, num_nv)) = j0j + mu * j1j;
            MatrixAt(J, env_id, row0 + 1, 0, i, Vec2i(1, num_nv)) = j0j - mu * j1j;
            MatrixAt(J, env_id, row0 + 2, 0, i, Vec2i(1, num_nv)) = j0j + mu * j2j;
            MatrixAt(J, env_id, row0 + 3, 0, i, Vec2i(1, num_nv)) = j0j - mu * j2j;
        }
        
    }

    template<typename TDataType>
    __global__ void AnchorConstraintJacobianKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;
        const auto& batch_anchor = rigid_body_system.anchor_constraints;
        const int& anchor_nums = batch_anchor.anchor_nums[env_id];
        if(cidx >= anchor_nums)
            return;

        const int& c_start = rigid_body_system.constraint_offset[env_id][0];
        const int& a_idx = batch_anchor.body_idxs(env_id, cidx).first;
        const int& b_idx = batch_anchor.body_idxs(env_id, cidx).second;
        const Vec3f& anchor_A_global = batch_anchor.anchor_A_world(env_id, cidx);
        const Vec3f& anchor_B_global = batch_anchor.anchor_B_world(env_id, cidx);
        const int& nv = rigid_body_system.batch_nv[env_id];
        
        auto& J = rigid_body_system.batch_J;

        Real jacA[6 * NV_TMP];
        Real jacB[6 * NV_TMP];
        ComputeJac(jacA, anchor_A_global, rigid_body_system, env_id, a_idx, cidx);
        ComputeJac(jacB, anchor_B_global, rigid_body_system, env_id, b_idx, cidx);

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < nv; j++)
            {
                int row = c_start + cidx * 3 + i;
                J(env_id, row * nv + j) = jacA[(i + 3) * nv + j] - jacB[(i + 3) * nv + j];
            }
    }

    template<typename TDataType>
    __global__ void FrictionLossJacobianKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;

        const int& constraint_num = rigid_body_system.num_each_constraint[env_id][1];
        if(cidx >= constraint_num)
            return;
        
        const int& constraint_start = rigid_body_system.constraint_offset[env_id][1];
        const int& nv = rigid_body_system.batch_nv[env_id];
        auto& batch_J = rigid_body_system.batch_J;

        const int& nv_idx = rigid_body_system.friction_loss_constraints.dof_frictionloss(env_id, cidx);
        
        batch_J(env_id, (constraint_start + cidx) * nv + nv_idx) = 1.f;


    }

    template<typename TDataType>
    __global__ void JointLimitJacobianKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int cidx = threadIdx.x;

        const int& constraint_num = rigid_body_system.num_each_constraint[env_id][2];
        if(cidx >= constraint_num)
            return;
        
        const auto& constraints = rigid_body_system.joint_limit_constraints;
        const int& jl_cid = constraints.active_mapping(env_id, cidx);
        const int& bid = constraints.joint_idx(env_id, jl_cid);
        const int& q_start = rigid_body_system.q_offset(env_id, bid);
        const int& joint_type = rigid_body_system.joint_type(env_id, bid);
        const Real& pos_err = constraints.limit_error(env_id, jl_cid);
        const int row_idx = rigid_body_system.constraint_offset[env_id][2] + cidx;
        const int& nv = rigid_body_system.batch_nv[env_id];

        auto& J = rigid_body_system.batch_J;

        if(joint_type < 3)     // Hinge or Slide
            J(env_id, row_idx * nv + q_start) = pos_err < 0.f ? 1.f : -1.f;
        else                    // Ball
        {
            const Vec3f& axis = constraints.limit_extern(env_id, jl_cid);
            for(int i = 0; i < 3; i++)
                J(env_id, row_idx * nv + q_start + i) = axis[i];
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

    __device__ Vec4f ComputeKBIP(Real error, Real dmax, Real dmin, Real time_const, 
        Real damp_ratio, Real midpoint, Real width, Real power)
    {
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
    __global__ void ComputeAnchorAref(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
    
        int anchor_idx = threadIdx.x;
        const auto& anchor_constraints = rigid_body_system.anchor_constraints;
        if(anchor_idx >= rigid_body_system.num_each_constraint[env_id][0])
            return;

        const auto& dmax = anchor_constraints.dmax(env_id, anchor_idx);
        const auto& dmin = anchor_constraints.dmin(env_id, anchor_idx);
        const auto& time_const = anchor_constraints.time_const(env_id, anchor_idx);
        const auto& damp_ratio = anchor_constraints.damp_ratio(env_id, anchor_idx);
        const auto& midpoint = anchor_constraints.midpoint(env_id, anchor_idx);
        const auto& width = anchor_constraints.width(env_id, anchor_idx);
        const auto& power = anchor_constraints.power(env_id, anchor_idx);
        const auto& constraint_vels = rigid_body_system.batch_constraint_vel;

        auto& aref = rigid_body_system.batch_aref;
        auto& imp = rigid_body_system.batch_imp;

        Real pos_err_norm = anchor_constraints.anchor_error(env_id, anchor_idx).norm();
        Vec4f KBIP = ComputeKBIP(pos_err_norm, dmax, dmin, time_const, damp_ratio, midpoint, width, power);
        Real K = KBIP[0];
        Real B = KBIP[1];
        Real I = KBIP[2];
        for(int i = 0; i < 3; i++)
        {
            imp(env_id, anchor_idx * 3 + i) = I;
            aref(env_id, anchor_idx * 3 + i) = -B * constraint_vels(env_id, anchor_idx * 3 + i) - K * I * pos_err_norm;
        }
    }

    template<typename TDataType>
    __global__ void ComputeFrictionLossAref(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int fidx = threadIdx.x;
        const auto& friction_loss_constraints = rigid_body_system.friction_loss_constraints;
        if(fidx >= rigid_body_system.num_each_constraint[env_id][1])
            return;

        const int& c_offset = rigid_body_system.constraint_offset[env_id][1];
        const auto& dmax = friction_loss_constraints.dmax(env_id, fidx);
        const auto& dmin = friction_loss_constraints.dmin(env_id, fidx);
        const auto& time_const = friction_loss_constraints.time_const(env_id, fidx);
        const auto& damp_ratio = friction_loss_constraints.damp_ratio(env_id, fidx);
        const auto& midpoint = friction_loss_constraints.midpoint(env_id, fidx);
        const auto& width = friction_loss_constraints.width(env_id, fidx);
        const auto& power = friction_loss_constraints.power(env_id, fidx);

        auto& imp = rigid_body_system.batch_imp;
        auto& aref = rigid_body_system.batch_aref;
        const auto& constraint_vels = rigid_body_system.batch_constraint_vel;
        Vec4f KBIP = ComputeKBIP(0.f, dmax, dmin, time_const, damp_ratio, midpoint, width, power);
        imp(env_id, c_offset + fidx) = KBIP[2];
        aref(env_id, c_offset + fidx) = -KBIP[1] * constraint_vels(env_id, c_offset + fidx);
    }

    template<typename TDataType>
    __global__ void ComputeJointLimitAref(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
    
        int cidx = threadIdx.x;
        const auto& joint_limit_constraints = rigid_body_system.joint_limit_constraints;
        if(cidx >= rigid_body_system.num_each_constraint[env_id][2])
            return;

        const int& c_offset = rigid_body_system.constraint_offset[env_id][2];
        const int& jl_cidx = joint_limit_constraints.active_mapping(env_id, cidx);

        const auto& dmax = joint_limit_constraints.dmax(env_id, jl_cidx);
        const auto& dmin = joint_limit_constraints.dmin(env_id, jl_cidx);
        const auto& time_const = joint_limit_constraints.time_const(env_id, jl_cidx);
        const auto& damp_ratio = joint_limit_constraints.damp_ratio(env_id, jl_cidx);
        const auto& midpoint = joint_limit_constraints.midpoint(env_id, jl_cidx);
        const auto& width = joint_limit_constraints.width(env_id, jl_cidx);
        const auto& power = joint_limit_constraints.power(env_id, jl_cidx);
        const auto& constraint_vels = rigid_body_system.batch_constraint_vel;
        const auto& pos_err = joint_limit_constraints.limit_error(env_id, jl_cidx);

        auto& imp = rigid_body_system.batch_imp;
        auto& aref = rigid_body_system.batch_aref;

        Real pos_err_abs = abs(pos_err);

        Vec4f KBIP = ComputeKBIP(pos_err_abs, dmax, dmin, time_const, damp_ratio, midpoint, width, power);

        Real K = KBIP[0];
        Real B = KBIP[1];
        Real I = KBIP[2];
        imp(env_id, c_offset + cidx) = I;
        aref(env_id, c_offset + cidx) = -B * constraint_vels(env_id, c_offset + cidx) + K * I * pos_err_abs;
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
        const auto& body_idxs = collision_constraints.body_idxs(env_id, contact_idx);

        const int a_idx = body_idxs.first;
        const int b_idx = body_idxs.second;

        const Real& wa = rigid_body_system.contact_weights(env_id, a_idx);
        Real wb = b_idx == -1 ? 0.f : rigid_body_system.contact_weights(env_id, b_idx);

        Real time_const, damp_ratio, dmax, dmin, midpoint, width, power;
        ComputeContactParas(collision_constraints, body_idxs, env_id, wa, wb, time_const, damp_ratio, dmax, dmin, midpoint, width, power);

        Vec4f KBIP = ComputeKBIP(depth, dmax, dmin, time_const, damp_ratio, midpoint, width, power);

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
    __global__ void ComputeDiagJMinvJTForBodies(RigidBody<TDataType> rigid_body_system, int num_envs)
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

        Real jac[6 * NV_TMP];
        Real j_tmp[6 * NV_TMP];
        const auto& pos = rigid_body_system.batch_pos(env_id, bid);
        ComputeJac(j_tmp, pos, rigid_body_system, env_id, bid, bid);
        RotateJacobianRow(j_tmp, jac, nv);

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
                Real sum = jac[r * nv + i];
                for(int k = 0; k < i; k++)
                    sum -= MatrixAt(L, env_id, i, k, Vec2i(nv, nv)) * j_tmp[r * nv + k];

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                j_tmp[r * nv + i] = sum / lii;
            }

            for(int i = nv - 1; i >= 0; i--)
            {
                Real sum = j_tmp[r * nv + i];
                for(int k = i + 1; k < nv; k++)
                    sum -= MatrixAt(L, env_id, k, i, Vec2i(nv, nv)) * j_tmp[r * nv + k];

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
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

        rigid_body_system.batch_weight_inv(env_id, bid) = (a00 + a11 + a22) / 3.f;
        
        
    }

    template<typename TDataType>
    __global__ void ComputeDiagJMinvJTForJoints(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int bid = threadIdx.x;
        int num_bodies = rigid_body_system.batch_bodies[env_id];
        if(bid >= num_bodies)
            return;

        const int nv = rigid_body_system.batch_nv[env_id];
        const int q_start = rigid_body_system.q_offset(env_id, bid);
        const int q_num = rigid_body_system.q_lengths(env_id, bid);
        auto& dof_weight_inv = rigid_body_system.batch_dof_weight_inv;

        const int& joint_type = rigid_body_system.joint_type(env_id, bid);
        if(joint_type == 0)     // free
            return;

        const auto& L = rigid_body_system.batch_qM_inv;
        
        if(joint_type < 3)      // Hinge or Slide
        {

            Real x[NV_TMP];
            for(int i = 0; i < nv; i++)
                x[i] = (i == q_start) ? 1.f : 0.f;

            for(int i = 0; i < nv; i++)
            {
                Real sum = x[i];
                for(int k = 0; k < i; k++)
                    sum -= MatrixAt(L, env_id, i, k, Vec2i(nv, nv)) * x[k];

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                x[i] = sum / lii;
            }

            for(int i = nv - 1; i >= 0; i--)
            {
                Real sum = x[i];
                for(int k = i + 1; k < nv; k++)
                    sum -= MatrixAt(L, env_id, k, i, Vec2i(nv, nv)) * x[k];

                const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                x[i] = sum / lii;
            }

            dof_weight_inv(env_id, q_start) = x[q_start];
        }
        else                    // Ball
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
                        sum -= MatrixAt(L, env_id, i, k, Vec2i(nv, nv)) * x[k];

                    const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
                    x[i] = sum / lii;
                }

                for(int i = nv - 1; i >= 0; i--)
                {
                    Real sum = x[i];
                    for(int k = i + 1; k < nv; k++)
                        sum -= MatrixAt(L, env_id, k, i, Vec2i(nv, nv)) * x[k];

                    const Real lii = MatrixAt(L, env_id, i, i, Vec2i(nv, nv));
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
    __global__ void ComputeAnchor_dAKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int anchor_idx = threadIdx.x;
        if(anchor_idx >= rigid_body_system.num_each_constraint[env_id][0])
            return;

        const auto& anchor_constraints = rigid_body_system.anchor_constraints;

        const int& body_A_idx = anchor_constraints.body_idxs(env_id, anchor_idx).first;
        const int& body_B_idx = anchor_constraints.body_idxs(env_id, anchor_idx).second;
        Real w = rigid_body_system.batch_weight_inv(env_id, body_A_idx) + rigid_body_system.batch_weight_inv(env_id, body_B_idx);

        auto& dA = rigid_body_system.batch_dA;
        for(int i = 0; i < 3; i++)
            dA(env_id, anchor_idx * 3 + i) = w;
    }

    template<typename TDataType>
    __global__ void ComputeFrictionLoss_dAKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        int fidx = threadIdx.x;
        if(fidx >= rigid_body_system.num_each_constraint[env_id][1])
            return;

        const auto& friction_loss_constraints = rigid_body_system.friction_loss_constraints;

        const int& dof_idx = friction_loss_constraints.dof_frictionloss(env_id, fidx);
        const int& c_offset = rigid_body_system.constraint_offset[env_id][1];
        Real w = rigid_body_system.batch_dof_weight_inv(env_id, dof_idx);
        rigid_body_system.batch_dA(env_id, c_offset + fidx) = w;
    }

    template<typename TDataType>
    __global__ void ComputeJointLimit_dAKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        int jidx = threadIdx.x;
        if(jidx >= rigid_body_system.num_each_constraint[env_id][2])
            return;

        const auto& joint_limit_constraints = rigid_body_system.joint_limit_constraints;
        auto& dA = rigid_body_system.batch_dA;

        const int& jl_cidx = joint_limit_constraints.active_mapping(env_id, jidx);
        const int& body_idx = joint_limit_constraints.joint_idx(env_id, jl_cidx);
        const int& q_start = rigid_body_system.q_offset(env_id, body_idx);
        const int& c_offset = rigid_body_system.constraint_offset[env_id][2];
        
        dA(env_id, c_offset + jidx) = rigid_body_system.batch_dof_weight_inv(env_id, q_start);
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

        const int collision_offset = rigid_body_system.constraint_offset[env_id][3];
        auto& R = rigid_body_system.batch_D;
        const auto& imp = rigid_body_system.batch_imp;
        const auto& dA = rigid_body_system.batch_dA;

        
        R(env_id, cidx) = (1.f - imp(env_id, cidx)) * dA(env_id, cidx) / imp(env_id, cidx);
        if(cidx >= collision_offset)
        {
            const Real mu = rigid_body_system.collision_constraints.mu(env_id, (cidx - collision_offset) / 4);
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
    __global__ void AnchorEnergyKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int anchor_num = rigid_body_system.num_each_constraint[env_id][0];

        int cidx = threadIdx.x;
        if(cidx >= anchor_num)
            return ;

        auto& constraint_force = rigid_body_system.batch_constraint_force;
        auto& energy = rigid_body_system.batch_constraint_energy;
        const auto& D = rigid_body_system.batch_D(env_id, cidx);
        const Real& Jaref = rigid_body_system.batch_Jaref(env_id, cidx);

        constraint_force(env_id, cidx) = - D * Jaref;
        energy(env_id, cidx) = 0.5f * D * Jaref * Jaref;
    }

    template<typename TDataType>
    __global__ void FrictionLossEnergyKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(rigid_body_system.is_converged[env_id])
            return;

        const int& c_num = rigid_body_system.num_each_constraint[env_id][1];
        int cidx = threadIdx.x;
        if(cidx >= c_num)
            return;

        const int& c_offset = rigid_body_system.constraint_offset[env_id][1];
        auto& constraint_force = rigid_body_system.batch_constraint_force(env_id, c_offset + cidx);
        auto& energy = rigid_body_system.batch_constraint_energy(env_id, c_offset + cidx);
        const auto& D = rigid_body_system.batch_D(env_id, c_offset + cidx);
        const Real& Jaref = rigid_body_system.batch_Jaref(env_id, c_offset + cidx);
        const Real& dof_frictionloss = rigid_body_system.friction_loss_constraints.dof_frictionloss(env_id, cidx);
        auto& unquads = rigid_body_system.batch_unquads(env_id, c_offset + cidx);

        constraint_force = - D * Jaref;

        Real R_dof_fl = dof_frictionloss / D;
        
        if(Jaref <= -R_dof_fl)
        {
            energy = -0.5f * R_dof_fl * dof_frictionloss - dof_frictionloss * Jaref;
            constraint_force = dof_frictionloss;
            unquads = 1;
        }
        else if(Jaref >= R_dof_fl)
        {
            energy = -0.5f * R_dof_fl * dof_frictionloss + dof_frictionloss * Jaref;
            constraint_force = -dof_frictionloss;
            unquads = 1;
        }
        else
            energy = 0.5f * D * Jaref * Jaref;
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
        if(Jaref_cidx > 0)
        {
            constraint_force(env_id, cidx) = 0.f;
            rigid_body_system.batch_unquads(env_id, cidx) = 1;
        }
        else
            constraint_energy(env_id, cidx) = 0.5f * D_cidx * Jaref_cidx * Jaref_cidx;
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
        printf("Env %d, DOF %d, JTf = %f\n", env_id, dof_idx, jt_f);

        rigid_body_system.batch_grad(env_id, dof_idx) =
            - rigid_body_system.batch_Ma(env_id, dof_idx)
            + rigid_body_system.batch_q_ex_force(env_id, dof_idx)
            + jt_f;
        printf("Env %d, DOF %d, JTf = %f, Ma = %f, q_ex_force = %f\n", env_id, dof_idx, jt_f, rigid_body_system.batch_Ma(env_id, dof_idx), rigid_body_system.batch_q_ex_force(env_id, dof_idx));
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

    template<typename T>
    __device__ void ComputeComVel(const DArray2D<T>& cdof, const DArray2D<T>& qvel, DArray2D<T>& com_vel,
        int env_id, int bid, int q_start, int offset)
    {
        // com_vel = Mat(cdof)^T * qvel, [3 * 6]^T x [3 * 1]
        
        for(int r = 0; r < 6; r++)
        {
            T sum = T(0);
            for(int c = 0; c < 3; c++)
                sum += cdof(env_id, (q_start + offset + c) * 6 + r) * qvel(env_id, q_start + offset + c);

            com_vel(env_id, bid * 6 + r) += sum; 
        }
    }

    template<typename T>
    __device__ void ComputeCVelCross(const DArray2D<T>& cdof, const DArray2D<T>& com_vel, DArray2D<T>& cdof_dot,
        int env_id, int bid, int q_start, int offset)
    {
        const Real& cvel_0 = com_vel(env_id, bid * 6);
        const Real& cvel_1 = com_vel(env_id, bid * 6 + 1);
        const Real& cvel_2 = com_vel(env_id, bid * 6 + 2);
        const Real& cvel_3 = com_vel(env_id, bid * 6 + 3);
        const Real& cvel_4 = com_vel(env_id, bid * 6 + 4);
        const Real& cvel_5 = com_vel(env_id, bid * 6 + 5);

        int idx = (q_start + offset) * 6;
        const Real& cdof_0 = cdof(env_id, idx);
        const Real& cdof_1 = cdof(env_id, idx + 1);
        const Real& cdof_2 = cdof(env_id, idx + 2);
        const Real& cdof_3 = cdof(env_id, idx + 3);
        const Real& cdof_4 = cdof(env_id, idx + 4);
        const Real& cdof_5 = cdof(env_id, idx + 5);

        cdof_dot(env_id, idx) = -cvel_2 * cdof_1 + cvel_1 * cdof_2;
        cdof_dot(env_id, idx + 1) = cvel_2 * cdof_0 - cvel_0 * cdof_2;
        cdof_dot(env_id, idx + 2) = -cvel_1 * cdof_0 + cvel_0 * cdof_1;
        cdof_dot(env_id, idx + 3) = -cvel_2 * cdof_4 + cvel_1 * cdof_5 - cvel_5 * cdof_1 + cvel_4 * cdof_2;
        cdof_dot(env_id, idx + 4) = cvel_2 * cdof_3 - cvel_0 * cdof_5 + cvel_5 * cdof_0 - cvel_3 * cdof_2;
        cdof_dot(env_id, idx + 5) = -cvel_1 * cdof_3 + cvel_0 * cdof_4 - cvel_4 * cdof_0 + cvel_3 * cdof_1;
    }

    template<typename TDataType>
    __global__ void ComputeComVelKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int& num_bodies = rigid_body_system.batch_bodies[env_id];
        const auto& cdof = rigid_body_system.batch_cdof;
        const auto& qvel = rigid_body_system.batch_qvel;
        auto& com_vel = rigid_body_system.subtree_com_vel;
        auto& cdof_dot = rigid_body_system.batch_cdof_dot;

        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int& q_start = rigid_body_system.q_offset(env_id, bid);
            if(parent_idx != -1)
            {
                const int& parent_q_start = rigid_body_system.q_offset(env_id, parent_idx);
                const int& joint_type = rigid_body_system.joint_type(env_id, bid);
                if(joint_type < 3)      // Hinge or Slide
                {
                    for(int i = 0; i < 6; i++)
                        com_vel(env_id, bid * 6 + i) = com_vel(env_id, parent_idx * 6 + i) 
                                                     + qvel(env_id, q_start) * cdof(env_id, q_start * 6 + i); 
                    
                    ComputeCVelCross(cdof, com_vel, cdof_dot, env_id, bid, q_start, 0);
                }
                else                    // Ball
                {
                    for(int i = 0; i < 6; i++)
                        com_vel(env_id, bid * 6 + i) = com_vel(env_id, parent_idx * 6 + i);
                    
                    for(int i = 0; i < 3; i++)
                        ComputeCVelCross(cdof, com_vel, cdof_dot, env_id, bid, q_start, i);
                    
                    ComputeComVel(cdof, qvel, com_vel, env_id, bid, q_start, 0);
                }
            }
            else
            {
                const int& is_static = rigid_body_system.is_static(env_id, bid);
                if(is_static)
                    continue;
                
                // convert linear vel to world frame 
                ComputeComVel(cdof, qvel, com_vel, env_id, bid, q_start, 0);
                
                // compute cdof_dot
                for(int i = 0; i < 3; i++)
                    ComputeCVelCross(cdof, com_vel, cdof_dot, env_id, bid, q_start, 3 + i);

                // convert angular vel to world frame
                ComputeComVel(cdof, qvel, com_vel, env_id, bid, q_start, 3);
            }

            printf("comvel body %d: %f, %f, %f, %f, %f, %f\n", bid,
                com_vel(env_id, bid * 6), com_vel(env_id, bid * 6 + 1), com_vel(env_id, bid * 6 + 2),
                com_vel(env_id, bid * 6 + 3), com_vel(env_id, bid * 6 + 4), com_vel(env_id, bid * 6 + 5));
        }

    }
    
    template<typename T>
    __device__ void ComputeCACC(const DArray2D<T>& cdof_dot, const DArray2D<T>& qvel, DArray2D<T>& cacc,
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
    __device__ void ComputeCVelCrossDual(const DArray2D<T>& com_vel, const Real* Ivel, Real* res, int env_id, int bid)
    {
        const Real& cvel_0 = com_vel(env_id, bid * 6);
        const Real& cvel_1 = com_vel(env_id, bid * 6 + 1);
        const Real& cvel_2 = com_vel(env_id, bid * 6 + 2);
        const Real& cvel_3 = com_vel(env_id, bid * 6 + 3);
        const Real& cvel_4 = com_vel(env_id, bid * 6 + 4);
        const Real& cvel_5 = com_vel(env_id, bid * 6 + 5);

        const Real& vec_0 = Ivel[0];
        const Real& vec_1 = Ivel[1];
        const Real& vec_2 = Ivel[2];
        const Real& vec_3 = Ivel[3];
        const Real& vec_4 = Ivel[4];
        const Real& vec_5 = Ivel[5];

        res[0] = -cvel_2 * vec_1 + cvel_1 * vec_2 - cvel_5 * vec_4 + cvel_4 * vec_5;
        res[1] = cvel_2 * vec_0 - cvel_0 * vec_2 + cvel_5 * vec_3 - cvel_3 * vec_5;
        res[2] = -cvel_1 * vec_0 + cvel_0 * vec_1 - cvel_4 * vec_3 + cvel_3 * vec_4;
        res[3] = -cvel_2 * vec_4 + cvel_1 * vec_5;
        res[4] = cvel_2 * vec_3 - cvel_0 * vec_5;
        res[5] = -cvel_1 * vec_3 + cvel_0 * vec_4;
    }

    template<typename TDataType>
    __global__ void ComputeRNEKernel(RigidBody<TDataType> rigid_body_system, const DArray<Vec3f> gravities, int num_envs)
    {
        int env_id = blockIdx.x * blockDim.x + threadIdx.x;
        if(env_id >= num_envs)
            return;

        const int& num_bodies = rigid_body_system.batch_bodies[env_id];
        const Vec3f& gravity = gravities[env_id];
        
        auto& cacc = rigid_body_system.batch_cacc;
        auto& cforce = rigid_body_system.batch_cforce;
        auto& q_inner_force = rigid_body_system.batch_q_inner_force;
        const auto& cdof = rigid_body_system.batch_cdof;
        const auto& cdof_dot = rigid_body_system.batch_cdof_dot;
        const auto& qvel = rigid_body_system.batch_qvel;
        const auto& q_start = rigid_body_system.q_offset;
        const auto& com_inertia = rigid_body_system.subtree_inertia;
        const auto& com_vel = rigid_body_system.subtree_com_vel;

        Real Iacc[6];
        Real Ivel[6];
        Real vec6_buffer[6];

        // Construct joint space offset acceleration cacc
        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            if(parent_idx == -1)
            {
                for(int i = 0; i < 3; i++)
                    cacc(env_id, bid * 6 + 3 + i ) = -gravity[i];
                if(!rigid_body_system.is_static(env_id, bid))
                    ComputeCACC(cdof_dot, qvel, cacc, env_id, bid, q_start(env_id, bid), 6);
                
            }
            else
            {
                const int& joint_type = rigid_body_system.joint_type(env_id, bid);
                for(int i = 0; i < 6; i++)
                    cacc(env_id, bid * 6 + i) = cacc(env_id, parent_idx * 6 + i);
                if(joint_type < 3)     // Hinge or Slide
                {
                    for(int i = 0; i < 6; i++)
                        cacc(env_id, bid * 6 + i) += qvel(env_id, q_start(env_id, bid)) * cdof_dot(env_id, q_start(env_id, bid) * 6 + i);
                }
                else                   // Ball
                    ComputeCACC(cdof_dot, qvel, cacc, env_id, bid, q_start(env_id, bid), 3);
            }
            
            for(int i = 0; i < 6; i++)
                vec6_buffer[i] = cacc(env_id, bid * 6 + i); // vec6_buffer = cacc_tmp
            InertiaMultiVec(com_inertia, vec6_buffer, Iacc, env_id, bid);
            for(int i = 0; i < 6; i++)
                vec6_buffer[i] = com_vel(env_id, bid * 6 + i); // vec6_buffer = com_vel
            InertiaMultiVec(com_inertia, vec6_buffer, Ivel, env_id, bid);
            
            ComputeCVelCrossDual(com_vel, Ivel, vec6_buffer, env_id, bid); // vec6_buffer = vIv

            for(int i = 0; i < 6;i++)
                cforce(env_id, bid * 6 + i) = Iacc[i] + vec6_buffer[i];
        
        }

        // Accumulate cforce from children to parent
        for(int bid = num_bodies - 1; bid >= 0; bid--)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            if(parent_idx == -1)
                continue;

            for(int i = 0; i < 6; i++)
                cforce(env_id, parent_idx * 6 + i) += cforce(env_id, bid * 6 + i);
        }
        for(int bid = 0; bid < num_bodies; bid++)
            printf("cforce body %d: %f, %f, %f, %f, %f, %f\n", bid,
                cforce(env_id, bid * 6), cforce(env_id, bid * 6 + 1), cforce(env_id, bid * 6 + 2),
                cforce(env_id, bid * 6 + 3), cforce(env_id, bid * 6 + 4), cforce(env_id, bid * 6 + 5));

        // compute q_inner_force 
        for(int bid = 0; bid < num_bodies; bid++)
        {
            const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
            const int& is_static = rigid_body_system.is_static(env_id, bid);
            const int& joint_type = rigid_body_system.joint_type(env_id, bid);
            const int& q_start = rigid_body_system.q_offset(env_id, bid);

            if(is_static)
                continue;

            if(parent_idx == -1)
            {
                for(int i = 0; i < 6; i++)
                {
                    Real sum = 0.f;
                    for(int j = 0; j < 6; j++)
                        sum += cdof(env_id, (q_start + i) * 6 + j) * cforce(env_id, bid * 6 + j);

                    q_inner_force(env_id, q_start + i) = sum;
                }
            }
            else
            {
                if(joint_type < 3)      // Hinge or Slide
                {
                    Real sum = 0.f;
                    for(int i = 0; i < 6; i++)
                        sum += cdof(env_id, q_start * 6 + i) * cforce(env_id, bid * 6 + i);

                    q_inner_force(env_id, q_start) = sum;
                }
                else                    // Ball
                {
                    for(int i = 0; i < 3; i++)
                    {
                        Real sum = 0.f;
                        for(int j = 0; j < 6; j++)
                            sum += cdof(env_id, (q_start + i) * 6 + j) * cforce(env_id, bid * 6 + j);

                        q_inner_force(env_id, q_start + i) = sum;
                    }
                }
            }

            printf("env %d, body %d, q_inner_force: ", env_id, bid);
            for(int i = 0; i < rigid_body_system.q_lengths(env_id, bid); i++)
                printf("%f ", q_inner_force(env_id, q_start + i));
            printf("\n");
        }


    }

    template<typename TDataType>
    __global__ void UpdateJointPoseKernel(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if (env_id >= num_envs)
            return;

        const int& num_bodies = rigid_body_system.batch_bodies[env_id];
        int bid = threadIdx.x;
        if(bid >= num_bodies)
            return;

        auto& pos = rigid_body_system.batch_pos;
        auto& quat = rigid_body_system.batch_quat;
        auto& rot_mat = rigid_body_system.batch_rot;
        auto& joint_qpos = rigid_body_system.joint_qpos;

        const int& parent_idx = rigid_body_system.parent_idx(env_id, bid);
        const int& is_static = rigid_body_system.is_static(env_id, bid);
        const int& q_start = rigid_body_system.q_offset(env_id, bid);
        const int& qpos_start = rigid_body_system.qpos_offset(env_id, bid);
        const int& joint_qpos_start = rigid_body_system.joint_qpos_offset(env_id, bid);
        const auto& qpos = rigid_body_system.batch_qpos;

        if(parent_idx != -1)
        {
            const int& joint_type = rigid_body_system.joint_type(env_id, bid);
            if(joint_type < 3)     // Hinge or Slide
                joint_qpos(env_id, joint_qpos_start) = qpos(env_id, qpos_start);
            else                   // Ball
            {
                for(int i = 0; i < 4; i++)
                    joint_qpos(env_id, joint_qpos_start + i) = qpos(env_id, qpos_start + i);
            }
        }
        else if(!is_static)
        {
            for(int i = 0; i < 3; i++)
                pos(env_id, bid)[i] = qpos(env_id, qpos_start + i);

            quat(env_id, bid).x = qpos(env_id, qpos_start + 3);
            quat(env_id, bid).y = qpos(env_id, qpos_start + 4);
            quat(env_id, bid).z = qpos(env_id, qpos_start + 5);
            quat(env_id, bid).w = qpos(env_id, qpos_start + 6);
            rot_mat(env_id, bid) = quat(env_id, bid).toMatrix3x3();
        }

        printf("Env %d, Body %d, Position: (%f, %f, %f), joint_qpos: %f\n", 
                env_id, bid, pos(env_id, bid).x, pos(env_id, bid).y, pos(env_id, bid).z, joint_qpos(env_id, joint_qpos_start));
    }


    template<typename TDataType>
    __global__ void PrintTestInfos(RigidBody<TDataType> rigid_body_system, int num_envs)
    {
        int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;

        int tid = threadIdx.x;
        if(tid > 0)
            return;

        const int& num_bodies = rigid_body_system.batch_bodies[env_id];
        const auto& q_start = rigid_body_system.q_offset;
        const auto& q_length = rigid_body_system.q_lengths;
        const auto& cdof = rigid_body_system.batch_cdof;
        const auto& is_static = rigid_body_system.is_static;
        const auto& is_isolated = rigid_body_system.is_isolated;
        const auto& shape_type = rigid_body_system.shape_type;
        const auto& parent_idx = rigid_body_system.parent_idx;
        const auto& joint_type = rigid_body_system.joint_type;
        

        for(int bid = 0; bid < num_bodies; bid++)
        {
            printf("Env %d, Body %d, is_static: %d, is_isolated: %d, shape_type: %d, parent_idx: %d, joint_type: %d, q_offset: %d, q_length: %d\n",
                env_id, bid, is_static(env_id, bid), is_isolated(env_id, bid), shape_type(env_id, bid),
                parent_idx(env_id, bid), joint_type(env_id, bid), q_start(env_id, bid), q_length(env_id, bid));

            printf("!!cdof:\n");
            for(int i = 0; i < q_length(env_id, bid); i++)
            {
                for(int j = 0; j < 6; j++)
                    printf("  dof %d: %f ", i, cdof(env_id, (q_start(env_id, bid) + i) * 6 + j));
                printf("\n\n");
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

        const int max_joint_qpos = 128;

        const auto& env_infos = this->env_infos;
        const auto& rigid_body_system = this->rigid_body;

        const int num_envs = env_infos->num_envs;
        const int num_max_constraints = env_infos->max_constraints;
        rigid_body_system->max_bodies = GetMaxValue(rigid_body_system->batch_bodies, num_envs);
        const int max_bodies = rigid_body_system->max_bodies;
        const int max_nv = max_bodies * 6;

        CArray<int> env_num_bodies(num_envs);
        env_num_bodies.assign(rigid_body_system->batch_bodies);

        auto& collision_constraints = this->rigid_body->collision_constraints;
        CArray2D<Real> time_const_host(num_envs, max_bodies);
        CArray2D<Real> damp_ratio_host(num_envs, max_bodies);
        CArray2D<Real> dmax_host(num_envs, max_bodies);
        CArray2D<Real> dmin_host(num_envs, max_bodies);
        CArray2D<Real> width_host(num_envs, max_bodies);
        CArray2D<Real> midpoint_host(num_envs, max_bodies);
        CArray2D<int> power_host(num_envs, max_bodies);
        CArray2D<Real> contact_weights_host(num_envs, max_bodies);
        for(int i = 0; i < max_bodies; i++)
        {
            time_const_host(0, i) = 0.02f;
            damp_ratio_host(0, i) = 1.f;
            dmax_host(0, i) = 0.95f;
            dmin_host(0, i) = 0.9f;
            width_host(0, i) = 0.001f;
            midpoint_host(0, i) = 0.5f;
            power_host(0, i) = 2;
            contact_weights_host(0, i) = 1.f;
        }
        collision_constraints.time_const.assign(time_const_host);
        collision_constraints.damp_ratio.assign(damp_ratio_host);
        collision_constraints.dmax.assign(dmax_host);
        collision_constraints.dmin.assign(dmin_host);
        collision_constraints.width.assign(width_host);
        collision_constraints.midpoint.assign(midpoint_host);
        collision_constraints.power.assign(power_host);
        rigid_body_system->contact_weights.assign(contact_weights_host);
        

        INIT_DYNO_ARRAY(rigid_body_system->batch_nv, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->is_isolated, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->q_lengths, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->q_offset, num_envs, max_bodies);

        INIT_DYNO_ARRAY2D(rigid_body_system->qpos_lengths, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->qpos_offset, num_envs, max_bodies);

        INIT_DYNO_ARRAY2D(rigid_body_system->root_idx, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_mass, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_com, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_inertia, num_envs, max_bodies * 10);
        INIT_DYNO_ARRAY2D(rigid_body_system->subtree_com_vel, num_envs, max_bodies * 6);


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
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_inertia, num_envs, max_bodies);
        INIT_DYNO_ARRAY(rigid_body_system->batch_scale, num_envs);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdof, num_envs, max_nv * 6);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cdof_dot, num_envs, max_nv * 6);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_q_chain, num_envs, max_nv * max_bodies);

        INIT_DYNO_ARRAY2D(rigid_body_system->batch_qpos, num_envs, max_bodies * 7);
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

        INIT_DYNO_ARRAY(rigid_body_system->anchor_constraints.anchor_nums, num_envs);


        INIT_DYNO_ARRAY2D(rigid_body_system->joint_axis, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->joint_anchor, num_envs, max_bodies);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cacc, num_envs, max_bodies * 6);
        INIT_DYNO_ARRAY2D(rigid_body_system->batch_cforce, num_envs, max_bodies * 6);


        spdlog::info("[MujocoSolver Solver] Allocated solver state arrays based on DoF counts.");
        // Initialize mass matrix for isolated bodies
        InitInertiaKernel<<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // 3. Initialize the qpos
        InitQposKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // 4. Build root index and calculate subtree mass
        BuildRootIndexKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        CalculateSubtreeMassKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // TODO: init constraint data

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
            env_infos->num_envs,
            rigid_body_system->is_converged);
        cudaDeviceSynchronize();

        rigid_body_system->batch_qacc.assign(rigid_body_system->batch_q_ex_acc);

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

        UpdateJointPoseKernel<TDataType><<<32, 512>>>(*rigid_body_system, env_infos->num_envs);
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

        // rigid_body_system->batch_cdof.reset();
        ComputeCdofKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        spdlog::info("INIT TEST");
        PrintTestInfos<TDataType><<<8, 1>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Crb 
        rigid_body_system->batch_crb.reset();
        // 1. Calculate the global inertia matrix of each rigid body when the center of mass of the corresponding kinematic tree is taken as the reference point.
        SubtreeInertialKernel<<<32, 512>>>(*rigid_body_system, num_envs);
        // 2. Calculate the global inertia matrix of each sub-tree.
        cudaDeviceSynchronize();
        AccumulateSubtreeInertialKernel<<<32, 128>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();


        // 3. Construct the system inertia matrix in the generalized coordinate system.
        rigid_body_system->batch_qM.reset();
        auto& q_chain = rigid_body_system->batch_q_chain;
        cudaMemset((void*)q_chain.begin(), -1, q_chain.pitch() * q_chain.ny());
        UpdateGeneralizedInertialMatrixKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->subtree_com_vel.reset();
        // rigid_body_system->batch_cdof_dot.reset();
        ComputeComVelKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        // Compute RNE
        rigid_body_system->batch_cacc.reset();
        rigid_body_system->batch_cforce.reset();
        ComputeRNEKernel<TDataType><<<32, 512>>>(*rigid_body_system, env_infos->gravities, num_envs);
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

        CollisonDetectionKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        UpdateAnchorConstarints<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        UpdateJointLimitConstraints<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();


        CountConstraintNums<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
    }

    template<typename TDataType>
    void MujocoSolver<TDataType>::MakeJacobian()
    {
        auto& env_infos = this->env_infos;
        auto& rigid_body_system = this->rigid_body;
        const int num_envs = env_infos->num_envs;

        rigid_body_system->batch_J.reset();
        ContactConstraintJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Anchor constraints
        AnchorConstraintJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Friction loss constraints
        FrictionLossJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        // Joint limit constraints
        JointLimitJacobianKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
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

        ComputeAnchorAref<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        ComputeFrictionLossAref<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        ComputeJointLimitAref<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        ComputeContactAref<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();
        
        printf("Aref:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_aref, 0, 16);
        cudaDeviceSynchronize();

        // Compute constraint residuals Jaref
        BatchDenseMatrixVectorMul<<<32, 512>>>(rigid_body_system->batch_qM, rigid_body_system->batch_qacc,
            rigid_body_system->batch_Ma, rigid_body_system->batch_nv, rigid_body_system->batch_nv, num_envs);
        cudaDeviceSynchronize();
        printf("Ma!!!:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_Ma, 0, 3);
        cudaDeviceSynchronize();

        printf("qacc!!!:\n");
        PrintVector<<<1, 1>>>(rigid_body_system->batch_qacc, 0, 3);
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
    
        ComputeDiagJMinvJTForBodies<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        ComputeDiagJMinvJTForJoints<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        cudaDeviceSynchronize();

        rigid_body_system->batch_dA.reset();

        ComputeAnchor_dAKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        ComputeFrictionLoss_dAKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
        ComputeJointLimit_dAKernel<TDataType><<<32, 512>>>(*rigid_body_system, num_envs);
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
        rigid_body_system->batch_unquads.reset();

        AnchorEnergyKernel<TDataType><<<num_envs, 512>>>(*rigid_body_system, num_envs);
        FrictionLossEnergyKernel<TDataType><<<num_envs, 512>>>(*rigid_body_system, num_envs);
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

        
        BatchCholeskySolveVarSizeKernel<<<num_envs, 1>>>(H, grad, x, rigid_body_system->batch_nv, num_envs, rigid_body_system->is_converged);
        cudaDeviceSynchronize();

        printf("dx (solution):\n");
        PrintVector<<<1, 1>>>(x, 0, 6);
        cudaDeviceSynchronize();
    }


    DEFINE_UNIQUE_CLASS(MujocoSolver, DataType3f);
}