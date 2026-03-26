#pragma once

#include "Collision/CollisionData.h"

#include "../PhysicalField.h"
#include "../../Utils/Constraints.h"
#include "../Rendering/GUI/WtGUI/NodeEditor/json.hpp"
#include <spdlog/spdlog.h>

using json = nlohmann::json;

// TODO: consider using a more flexible data structure to support more complex shapes (e.g., triangle mesh) and their parameters. For example, we can have a separate array for each shape type, and store the shape type and offset for each body to access the corresponding shape parameters.
// enum ShapeType
// {
//     Cube,
//     Sphere,
//     Capsule,
//     TriMesh
// };

// inline int GetShapeParamPadding(int shape_type)
// {
//     switch (shape_type)
//     {
//         case ShapeType::Cube:
//             return 3;   // lengths of 3 half-axes
//         case ShapeType::Sphere:
//             return 1;   // radius
//         case ShapeType::Capsule:
//             return 2;   // radius, half-length
//         case ShapeType::TriMesh:
//             return 1;   // mesh idx
//     }

//     return 0;
// }

namespace dyno {
    template<typename TDataType>
    struct RigidBody : public PhysicalFieldData<TDataType>
    {
        using Real = typename TDataType::Real;

        int                 max_bodies;
        DArray2D<int>       is_static;      // [env_id, body_id] whether the rigid body is static or dynamic
        DArray2D<int>       is_isolated;    // [env_id, body_id] whether the rigid body is isolated (not in contact with any other body)

        DArray<int>         batch_bodies;  // [env_id] num of rigid bodies in each environment
        DArray<int>         batch_body_offset; // [env_id] flattened rigid body offset in topo position/rotation arrays
    
        DArray<int>         batch_nv;      // [env_id] num of generalized DoFs
        DArray2D<int>       nv_offset;     // [env_id, body_idx] 
        
        
        DArray2D<int>       q_lengths;      // [env_id, body_id] num of generalized DoFs
        DArray2D<int>       q_offset;       // [env_id, body_id] offset of generalized DoFs;
        DArray2D<Real>      batch_qacc;    // [env_id, dof_idx] generalized acceleration
        DArray2D<Real>      batch_qvel;    // [env_id, dof_idx] generalized velocity
        DArray2D<Real>      batch_aref;    // nc * 1
        DArray2D<Real>      batch_Jaref;   // J * qacc
        DArray2D<Real>      batch_imp;
        DArray<Real>        batch_energy;
        DArray<Real>        batch_energy_ref;
        DArray2D<Real>      batch_constraint_energy;    
        DArray2D<int>       batch_unquads;   // [env_id, constraint_idx]
        DArray2D<Real>      batch_H;
        DArray2D<Real>      batch_dx;    // [env_id, dof_idx] delta for current Newton iteration


        DArray2D<int>       qpos_lengths;       // [env_id, body_id] num of generalized position
        DArray2D<int>       qpos_offset;
        DArray2D<Real>      batch_qpos;     // [env_id, dof_idx] generalized position
        
        DArray2D<Real>      batch_qM;      // [env_id, dof_idx] mass matrix    不应该显式的存，直接存LDL^T
        DArray2D<Vec3f>     batch_inertia;
        DArray2D<Real>      batch_qM_inv;
        // DArray2D<Real>      batch_qM_L;     // [env_id, dof_idx] lower triangular matrix L in the LDL^T decomposition of the mass matrix
        DArray2D<Real>      batch_qM_diag_elem;
        DArray<Real>        batch_scale;
        DArray2D<Real>      batch_cdof;    // [env_id, dof_idx] projection basis
        DArray2D<Real>      batch_cdof_dot; // [env_id, dof_idx] projection basis time derivative
        DArray2D<Real>      batch_crb;     // dense vec num_bodies * 10
        DArray2D<int>       batch_q_chain;

        DArray2D<Vec3f>     batch_pos;     // [env_id, body_id] world position of rigid body
        DArray2D<Mat3f>     batch_rot;     // [env_id, body_id] world rotation of rigid body (as rotation matrix)
        DArray2D<Quat<Real>> batch_quat;    // [env_id, body_id] world rotation of rigid body (as quaternion)
        DArray2D<Real>      batch_mass;    // [env_id, body_id] mass of rigid body
        
        DArray<Vec3f>       topo_pos_cache; // flattened body positions for topology update
        DArray<Mat3f>       topo_rot_cache; // flattened body rotations for topology update

        // For articulated bodies
        DArray2D<int>       parent_idx;   // [env_id, body_id] parent body index (-1 for root)
        DArray2D<int>       root_idx;     // [env_id, body_id] root body index
        DArray2D<Real>      subtree_mass;   // [env_id, body_id] mass of the subtree rooted at this body (including itself and all its children in the kinematic tree)
        DArray2D<Vec3f>     subtree_com;    // [env_id, body_id] center of mass of the subtree rooted at this body (including itself and all its children in the kinematic tree)
        DArray2D<Real>      subtree_inertia;
        DArray2D<Real>      subtree_com_vel;

        // solving cache
        DArray2D<Real>      batch_q_inner_force;
        DArray2D<Real>      batch_q_ex_force;
        DArray2D<Real>      batch_q_ex_acc;
        DArray2D<Real>      batch_Ma;       // qM * qacc
        DArray2D<Real>      batch_grad;     // nv * 1, Newton gradient: Ma - q_ex_force - J^T * constraint_force
        DArray2D<Real>      batch_weight_inv;    // nbody * 1
        DArray2D<Real>      batch_dof_weight_inv; // nv * 1
        DArray2D<Real>      batch_dA;       // nc * 1
        DArray2D<Real>      batch_D;        // nc * 1
        DArray<int>         is_converged;
        DArray<Real>        sys_alpha;

        DArray2D<Real>      Mat_temp1;
        DArray2D<Real>      Mat_temp2;

        // For constraints
        DArray2D<Real>               batch_J;           // [env_id, num_constraints * max_dof] Jacobian matrix of constraints   nc * nv
        DArray<int>                  num_constraints;   // [env_id] number of constraints in each environment = num_collision_constraints + num_topo_invariant_constraints
        DArray<Vec4i>                num_each_constraint; // [env_id] num of each type of constraint (Vec4i: [0] anchor, [1] friction loss, [2] joint limit, [3] collision_constraints)
        DArray<Vec4i>                constraint_offset;   // [env_id] offset of each type of constraint in the batch_J (Vec4i: [0 ~ 2] topo_invariant_constraints, [3] collision_constraints)
        DArray2D<Real>               batch_constraint_vel;
        DArray2D<Real>               batch_constraint_force;

        BatchAnchorConstraints          anchor_constraints;
        BatchFrictionLossConstraints    friction_loss_constraints;
        BatchJointLimitConstraints      joint_limit_constraints;
        CollisionConstraintParas        collision_paras;
        BatchCollisionConstraints       collision_constraints;
        

        // For joint
        DArray2D<int>           joint_type;     // [env_id, body_id] type of joint (0: none, 1: hinge, 2: slide, 3: ball)
        DArray2D<Real>          joint_qpos;
        DArray2D<Real>          joint_qpos_ref;
        DArray2D<int>           joint_qpos_offset;
        DArray2D<Vec3f>         joint_rel_pos;
        DArray2D<Quat<Real>>    joint_rel_quat;
        DArray2D<Vec3f>         joint_axis;     // [env_id, body_id] joint axis for hinge and slide joint, or initial relative rotation axis for ball joint
        DArray2D<Vec3f>         joint_axis_ref;
        DArray2D<Vec3f>         joint_anchor;   // [env_id, body_id] joint anchor in the local frame of the body
        DArray2D<Vec3f>         joint_anchor_ref;
        DArray2D<Real>          batch_cacc;
        DArray2D<Real>          batch_cforce;


        // Shape information for rendering and collision handling
        DArray2D<int>           shape_type;    // [body_id] type
        DArray2D<int>           shape_idx;     // [body_id] index to the corresponding shape parameter array (e.g., box_params, sphere_params, etc.)
        
        DArray<int>             env_num_boxes;  // [env_id] number of boxes in each environment
        DArray<int>             env_box_offset;  // [env_id] offset of boxes in the global box array
        DArray2D<BoxInfo>       boxes;
        
        DArray<int>             env_num_spheres;  // [env_id] number of spheres in each environment
        DArray<int>             env_sphere_offset;  // [env_id] offset of spheres in the global sphere array
        DArray2D<SphereInfo>    spheres;

        DArray<int>             env_num_capsules;  // [env_id] number of capsules in each environment
        DArray<int>             env_capsule_offset;  // [env_id] offset of capsules in the global capsule array
        DArray2D<CapsuleInfo>   capsules;


        DArray<Vec3i>           rendering_idx_2_rigid_body_mapping; // [env_id, shape_type, shape_idx]
        DArray2D<int>           rigid_body_2_rendering_idx_mapping; // [env_id, body_id] -> idx of its pos in topo state

    public:
        void ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num);
    };

    template<typename TDataType>
    void RigidBody<TDataType>::ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num) {
        spdlog::info("Start initializing rigid body state variables.");
        // std::cout << "primitive max_num: " << primitive_max_num[0] << primitive_max_num[1] << prim
        int env_num = envs_json.size();

        CArray2D<int> shape_type_host(env_num, body_max_num);
        CArray2D<int> shape_idx_host(env_num, body_max_num);
        CArray2D<int> parent_idx_host(env_num, body_max_num);

        CArray2D<SphereInfo> spheres_host(env_num, primitive_max_num[0]);
        CArray2D<BoxInfo> boxes_host(env_num, primitive_max_num[1]);
        CArray2D<CapsuleInfo> capsules_host(env_num, primitive_max_num[2]);

        std::vector<int> env_num_boxes_host(env_num, 0);
        std::vector<int> env_box_offset_host(env_num, 0);

        std::vector<int> env_num_spheres_host(env_num, 0);
        std::vector<int> env_sphere_offset_host(env_num, 0);

        std::vector<int> env_num_capsules_host(env_num, 0);
        std::vector<int> env_capsule_offset_host(env_num, 0);

        std::vector<int> batch_bodies_host(env_num, 0);
        std::vector<int> batch_body_offset_host(env_num, 0);

        CArray2D<Vec3f>     body_pos_host(env_num, body_max_num);
        CArray2D<Mat3f>     body_rot_host(env_num, body_max_num);
        CArray2D<Quat<Real>> batch_quat_host(env_num, body_max_num);

        CArray2D<int>           joint_type_host(env_num, body_max_num);
        CArray2D<Real>          joint_qpos_host(env_num, body_max_num * 6);
        CArray2D<Real>          joint_qpos_ref_host(env_num, body_max_num * 6);
        CArray2D<int>           joint_qpos_offset_host(env_num, body_max_num);
        CArray2D<Vec3f>         joint_rel_pos_host(env_num, body_max_num);
        CArray2D<Quat<Real>>    joint_rel_quat_host(env_num, body_max_num);
        CArray2D<Vec3f>         joint_axis_ref_host(env_num, body_max_num);
        CArray2D<Vec3f>         joint_anchor_ref_host(env_num, body_max_num);

        std::vector<Vec3i>  rendering_idx_2_rigid_body_mapping_host;
        CArray2D<int>       rigid_body_2_rendering_idx_mapping_host(env_num, body_max_num);

        CArray2D<int> is_static_host(env_num, body_max_num);
        CArray2D<Real> mass_host(env_num, body_max_num);

        // std::cout << "initial" << std::endl;

        for (int eid = 0; eid < env_num; ++eid)
        {
            for (int sid = 0; sid < body_max_num; ++sid)
            {
                shape_type_host(eid, sid) = -1;
                shape_idx_host(eid, sid) = -1;
            }

            for (int bid = 0; bid < body_max_num; ++bid)
            {
                rigid_body_2_rendering_idx_mapping_host(eid, bid) = -1;
                body_rot_host(eid, bid) = Mat3f::identityMatrix();
                batch_quat_host(eid, bid) = Quat<Real>::identity();

                joint_type_host(eid, bid) = 0;
            }
        }

        // std::cout << "initial finish" << std::endl;

        int eid = 0;
        int total_bodies = 0;
        int total_spheres = 0;
        int total_boxes = 0;
        int total_capsules = 0;

        for (const auto& env_json : envs_json) {
            if (env_json.contains("rigid_body") && env_json["rigid_body"].is_array()) {
                int bid = 0;
                int sphere_num = 0;
                int box_num = 0;
                int capsule_num = 0;
                int joint_qpos_offset = 0;

                for (const auto& rb_json : env_json["rigid_body"]) {

                    auto pos = rb_json.at("pos").get<std::vector<float>>();
                    body_pos_host(eid, bid) = Vec3f(pos[0], pos[1], pos[2]);
                    auto quat = rb_json.at("quat").get<std::vector<float>>();
                    batch_quat_host(eid, bid) = Quat<Real>(quat[0], quat[1], quat[2], quat[3]);
                    body_rot_host(eid, bid) = batch_quat_host(eid, bid).toMatrix3x3();

                    mass_host(eid, bid) = rb_json["mass"];
                    bool is_static = rb_json["is_static"];
                    is_static_host(eid, bid) = is_static ? 1 : 0;

                    if (rb_json.at("type") == "primitive") {
                        parent_idx_host(eid, bid) = -1;

                        int render_idx = -1;

                        int id = rb_json.at("ID").get<int>();
                        switch (id) {
                            case 0: {
                                shape_type_host(eid, bid) = 0;
                                shape_idx_host(eid, bid) = sphere_num;

                                spheres_host(eid, sphere_num).center = Vec3f(0, 0, 0);
                                auto halfLength = rb_json.at("size").get<std::vector<float>>();
                                spheres_host(eid, sphere_num).radius = halfLength[0];
                                // spheres_host(eid, sphere_num).rot = batch_quat_host(eid, bid);

                                sphere_num++;


                                // std::cout << "eid: " << eid << " bid: " << bid << std::endl;

                                break;
                            }
                            case 1: {
                                shape_type_host(eid, bid) = 1;
                                shape_idx_host(eid, bid) = box_num;

                                // boxes_host(eid, box_num).center = body_pos_host(eid, bid);
                                boxes_host(eid, box_num).center = Vec3f(0, 0, 0);
                                auto halfLength = rb_json.at("size").get<std::vector<float>>();
                                boxes_host(eid, box_num).halfLength = Vec3f(halfLength[0], halfLength[1], halfLength[2]);
                                // boxes_host(eid, box_num).rot = batch_quat_host(eid, bid);

                                box_num++;
                                break;
                            }
                            case 2: {
                                shape_type_host(eid, bid) = 2;
                                shape_idx_host(eid, bid) = capsule_num;

                                capsules_host(eid, capsule_num).center = Vec3f(0, 0, 0);
                                auto halfLength = rb_json.at("size").get<std::vector<float>>();
                                capsules_host(eid, capsule_num).radius= halfLength[0];
                                capsules_host(eid, capsule_num).halfLength= halfLength[1];
                                // capsules_host(eid, capsule_num).rot = batch_quat_host(eid, bid);

                                capsule_num++;
                                break;
                            }
                            default: ;
                        }
                    }

                    if (rb_json.contains("joint")) {
                        auto joint_json = rb_json["joint"];
                        parent_idx_host(eid, bid) = joint_json["parent"].get<int>();

                        auto joint_anchor_ref = joint_json.at("anchor").get<std::vector<float>>();
                        joint_anchor_ref_host(eid, bid) = Vec3f(joint_anchor_ref[0], joint_anchor_ref[1], joint_anchor_ref[2]);

                        if (rb_json.contains("axis")) {
                            auto joint_axis_ref = joint_json.at("axis").get<std::vector<float>>();
                            joint_axis_ref_host(eid, bid) = Vec3f(joint_axis_ref[0], joint_axis_ref[1], joint_axis_ref[2]);
                        }

                        joint_rel_pos_host(eid, bid) = body_pos_host(eid, bid);
                        joint_rel_quat_host(eid, bid) = batch_quat_host(eid, bid);

                        int type = joint_json.at("type").get<int>();
                        switch(type) {
                            case 1: {
                                joint_type_host(eid, bid) = 1;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("qpose").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];

                                auto joint_qpos_ref = joint_json.at("qpose_ref").get<std::vector<float>>();
                                joint_qpos_ref_host(eid, joint_qpos_offset) = joint_qpos_ref[0];

                                joint_qpos_offset++;
                                break;
                            }

                            case 2: {
                                joint_type_host(eid, bid) = 2;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("qpose").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];

                                auto joint_qpos_ref = joint_json.at("qpose_ref").get<std::vector<float>>();
                                joint_qpos_ref_host(eid, joint_qpos_offset) = joint_qpos_ref[0];

                                joint_qpos_offset++;
                                break;
                            }

                            case 3: {
                                joint_type_host(eid, bid) = 3;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("qpose").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];
                                joint_qpos_host(eid, joint_qpos_offset + 1) = joint_qpos[1];
                                joint_qpos_host(eid, joint_qpos_offset + 2) = joint_qpos[2];
                                joint_qpos_host(eid, joint_qpos_offset + 3) = joint_qpos[3];

                                auto joint_qpos_ref = joint_json.at("qpose_ref").get<std::vector<float>>();
                                joint_qpos_ref_host(eid, joint_qpos_offset) = joint_qpos_ref[0];
                                joint_qpos_ref_host(eid, joint_qpos_offset + 1) = joint_qpos_ref[1];
                                joint_qpos_ref_host(eid, joint_qpos_offset + 2) = joint_qpos_ref[2];
                                joint_qpos_ref_host(eid, joint_qpos_offset + 3) = joint_qpos_ref[3];

                                joint_qpos_offset += 4;
                                break;
                            }
                            default: ;
                        }
                    }

                    bid++;
                }

                batch_bodies_host[eid] = bid;
                env_num_spheres_host[eid] = sphere_num;
                env_num_boxes_host[eid] = box_num;
                env_num_capsules_host[eid] = capsule_num;

                batch_body_offset_host[eid] = total_bodies;
                env_sphere_offset_host[eid] = total_spheres;
                env_box_offset_host[eid] = total_boxes;
                env_capsule_offset_host[eid] = total_capsules;
                total_bodies += bid;
                total_spheres += sphere_num;
                total_boxes += box_num;
                total_capsules += capsule_num;
            }
            eid++;
        }

        // std::cout << "read info finished" << std::endl;

        rendering_idx_2_rigid_body_mapping_host.resize(total_spheres + total_boxes + total_capsules, Vec3i(-1, -1, -1));

        for (eid = 0; eid < env_num; ++eid)
        {
            for (int bid = 0; bid < batch_bodies_host[eid]; ++bid)
            {
                int st = shape_type_host(eid, bid);
                int si = shape_idx_host(eid, bid);
                if (st < 0 || si < 0)
                    continue;

                int render_idx = -1;
                if (st == 0)
                {
                    render_idx = env_sphere_offset_host[eid] + si;
                }
                else if (st == 1)
                {
                    render_idx = total_spheres + env_box_offset_host[eid] + si;
                }
                else if (st == 2)
                {
                    render_idx = total_spheres + total_boxes + env_capsule_offset_host[eid] + si;
                }

                if (render_idx >= 0 && render_idx < total_spheres + total_boxes + total_capsules)
                {
                    rigid_body_2_rendering_idx_mapping_host(eid, bid) = render_idx;
                    rendering_idx_2_rigid_body_mapping_host[render_idx] = Vec3i(eid, st, si);
                }
            }
        }

        for (eid = 0; eid < env_num; ++eid) {
            for (int bid = 0; bid < batch_bodies_host[eid]; ++bid) {
                const int pid = parent_idx_host(eid, bid);
                if (pid != -1) {
                    const auto& parent_quat = batch_quat_host(eid, pid);
                    const auto& parent_rot = body_rot_host(eid, pid);
                    const int& joint_type = joint_type_host(eid, bid);
                    const auto& joint_qpos_start = joint_qpos_offset_host(eid, bid);
                    const auto& local_axis = joint_axis_ref_host(eid, bid);
                    const auto& local_anchor = joint_anchor_ref_host(eid, bid);

                    Quat<Real> xquat_p = parent_quat * joint_rel_quat_host(eid, bid);
                    Vec3f joint_axis = xquat_p * local_axis;
                    Vec3f xanchor = xquat_p * local_anchor;
                    Vec3f xpos = parent_rot * joint_rel_pos_host(eid, bid) + body_pos_host(eid, pid);
                    xanchor += xpos;

                    if (joint_type == 2) {
                        batch_quat_host(eid, bid) = xquat_p;
                        body_rot_host(eid, bid) = xquat_p.toMatrix3x3();
                        body_pos_host(eid, bid) = xpos + (joint_qpos_host(eid, joint_qpos_start) - joint_qpos_ref_host(eid, joint_qpos_start)) * joint_axis;
                    } else {
                        Quat<Real> quat_local;
                        if (joint_type == 1)
                            quat_local.fromAxisAngle(local_axis, joint_qpos_host(eid, joint_qpos_start) - joint_qpos_ref_host(eid, joint_qpos_start));
                        else if (joint_type == 3) {
                            Quat<Real> ball_quat = Quat<Real>(
                                joint_qpos_host(eid, joint_qpos_start),
                                joint_qpos_host(eid, joint_qpos_start + 1),
                                joint_qpos_host(eid, joint_qpos_start + 2),
                                joint_qpos_host(eid, joint_qpos_start + 3));
                            ball_quat.normalize();
                            quat_local = ball_quat;
                        }

                        Quat<Real> xquat_c = xquat_p * quat_local;
                        batch_quat_host(eid, bid) = xquat_c;
                        body_rot_host(eid, bid) = xquat_c.toMatrix3x3();
                        xpos = xquat_c * local_anchor;
                        body_pos_host(eid, bid) = xanchor - xpos;
                    }
                }
            }
        }

        shape_type.assign(shape_type_host);
        shape_idx.assign(shape_idx_host);
        parent_idx.assign(parent_idx_host);

        boxes.assign(boxes_host);
        spheres.assign(spheres_host);
        capsules.assign(capsules_host);

        batch_pos.assign(body_pos_host);
        batch_rot.assign(body_rot_host);
        batch_quat.assign(batch_quat_host);

        env_num_boxes.assign(env_num_boxes_host);
        env_box_offset.assign(env_box_offset_host);
        env_num_spheres.assign(env_num_spheres_host);
        env_sphere_offset.assign(env_sphere_offset_host);
        env_num_capsules.assign(env_num_capsules_host);
        env_capsule_offset.assign(env_capsule_offset_host);

        batch_bodies.assign(batch_bodies_host);
        batch_body_offset.assign(batch_body_offset_host);

        rendering_idx_2_rigid_body_mapping.assign(rendering_idx_2_rigid_body_mapping_host);
        rigid_body_2_rendering_idx_mapping.assign(rigid_body_2_rendering_idx_mapping_host);

        is_static.assign(is_static_host);
        batch_mass.assign(mass_host);

        joint_type.assign(joint_type_host);
        joint_qpos.assign(joint_qpos_host);
        joint_qpos_ref.assign(joint_qpos_ref_host);
        joint_qpos_offset.assign(joint_qpos_offset_host);
        joint_anchor_ref.assign(joint_anchor_ref_host);
        joint_axis_ref.assign(joint_axis_ref_host);
        joint_rel_pos.assign(joint_rel_pos_host);
        joint_rel_quat.assign(joint_rel_quat_host);

        spdlog::info("Finished initializing rigid body state variables.");
    }
}



