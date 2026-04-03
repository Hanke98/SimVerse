#include "RigidBody.h"
#include "Object.h"
#include "DataTypes.h"
#include "Utils/utils.h"
#include <Utils/SimBlockVector.h>

namespace dyno
{
    template<typename TDataType>
    void RigidBody<TDataType>::ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num)
    {
        spdlog::info("Start initializing rigid body state variables.");
        // printf("connect_num: %d", connect_max);
        int env_num = envs_json.size();

        CArray2D<int> shape_type_host(env_num, body_max_num);
        CArray2D<int> shape_idx_host(env_num, body_max_num);
        std::vector<int> parent_idx_host;

        CArray2D<SphereInfo> spheres_host(env_num, primitive_max_num[0]);
        CArray2D<BoxInfo> boxes_host(env_num, primitive_max_num[1]);
        CArray2D<CapsuleInfo> capsules_host(env_num, primitive_max_num[2]);


        std::vector<int> env_num_boxes_host;
        std::vector<int> env_box_offset_host(env_num, 0);

        std::vector<int> env_num_spheres_host;
        std::vector<int> env_sphere_offset_host(env_num, 0);

        std::vector<int> env_num_capsules_host;
        std::vector<int> env_capsule_offset_host(env_num, 0);

        std::vector<int> batch_bodies_host;
        std::vector<int> batch_body_offset_host;

        CArray2D<Vec3f>     body_pos_host(env_num, body_max_num);
        CArray2D<Mat3f>     body_rot_host(env_num, body_max_num);
        CArray2D<Quat<Real>> batch_quat_host(env_num, body_max_num);

        std::vector<int>           joint_type_host;
        std::vector<Real>          joint_qpos_host;
        std::vector<int>           joint_qpos_num_host;
        std::vector<int>           joint_offset_host;
        std::vector<Real>          joint_qpos_ref_host;
        std::vector<int>           joint_qpos_offset_host;
        std::vector<Vec3f>         joint_rel_pos_host;
        std::vector<Quat<Real>>    joint_rel_quat_host;
        std::vector<Vec3f>         joint_axis_ref_host;
        std::vector<Vec3f>         joint_anchor_ref_host;

        std::vector<Real>            friction_mu_host;
        std::vector<Real>            contact_weights_host;

        std::vector<int>            jl_ref_num_host;
        std::vector<int>            jl_joint_idx_host;
        std::vector<int>            jl_is_upper_host;
        std::vector<Real>           jl_limit_host;

        std::vector<Real>          jl_tc_host;
        std::vector<Real>          jl_dr_host;
        std::vector<Real>          jl_dmax_host;
        std::vector<Real>          jl_dmin_host;
        std::vector<Real>          jl_width_host;
        std::vector<Real>          jl_midpoint_host;
        std::vector<int>           jl_power_host;

        std::vector<int>                   connect_anchor_nums_host;
        std::vector<Pair<int, int>>        connect_body_idxs_host;
        std::vector<Vec3f>                 connect_anchor_A_local_host;
        std::vector<Vec3f>                 connect_anchor_B_local_host;

        std::vector<Real>          connect_tc_host;
        std::vector<Real>          connect_dr_host;
        std::vector<Real>          connect_dmax_host;
        std::vector<Real>          connect_dmin_host;
        std::vector<Real>          connect_width_host;
        std::vector<Real>          connect_midpoint_host;
        std::vector<int>           connect_power_host;

        std::vector<int>           fl_dof_idxs_host;
        std::vector<Real>          fl_dof_frictionloss_host;
        std::vector<int>           fl_num_host;

        std::vector<Real>          fl_tc_host;
        std::vector<Real>          fl_dr_host;
        std::vector<Real>          fl_dmax_host;
        std::vector<Real>          fl_dmin_host;
        std::vector<Real>          fl_width_host;
        std::vector<Real>          fl_midpoint_host;
        std::vector<int>           fl_power_host;

        std::vector<Real>          col_tc_host;
        std::vector<Real>          col_dr_host;
        std::vector<Real>          col_dmax_host;
        std::vector<Real>          col_dmin_host;
        std::vector<Real>          col_width_host;
        std::vector<Real>          col_midpoint_host;
        std::vector<int>           col_power_host;

        std::vector<Vec3i>      rendering_idx_2_rigid_body_mapping_host;
        CArray2D<int>        rigid_body_2_rendering_idx_mapping_host(env_num, body_max_num);

        std::vector<int>    is_static_host;
        std::vector<Real>   mass_host;

        std::vector<int> num_constraints_host;
        std::vector<Vec4i> num_each_constraint_host;
        std::vector<Vec4i> constraint_offset_host;

        int eid = 0;
        int total_bodies = 0;
        int total_spheres = 0;
        int total_boxes = 0;
        int total_capsules = 0;
        int total_joint_qpos = 0;

        for (const auto& env_json : envs_json) {
            if (env_json.contains("rigid_body") && env_json["rigid_body"].is_array()) {
                int bid = 0;
                int sphere_num = 0;
                int box_num = 0;
                int capsule_num = 0;
                int joint_qpos_offset = 0;
                int jl_num = 0;

                for (const auto& rb_json : env_json["rigid_body"]) {

                    auto pos = rb_json.at("pos").get<std::vector<float>>();
                    body_pos_host(eid, bid) = Vec3f(pos[0], pos[1], pos[2]);
                    auto quat = rb_json.at("quat").get<std::vector<float>>();
                    batch_quat_host(eid, bid) = Quat<Real>(quat[0], quat[1], quat[2], quat[3]);
                    body_rot_host(eid, bid) = batch_quat_host(eid, bid).toMatrix3x3();


                    Real density = rb_json["density"].get<float>();

                    friction_mu_host.push_back(rb_json.at("friction").get<float>());
                    if (rb_json.contains("contact_weight"))
                        contact_weights_host.push_back(rb_json.at("contact_weight").get<float>());
                    else
                        contact_weights_host.push_back(1);

                    bool is_static = rb_json["is_static"].get<bool>();
                    is_static_host.push_back(is_static ? 1 : 0);

                    if (rb_json.at("type") == "primitive") {
                        int id = rb_json.at("ID").get<int>();
                        switch (id) {
                            case 0: {
                                shape_type_host(eid, bid) = 0;
                                shape_idx_host(eid, bid) = sphere_num;

                                spheres_host(eid, sphere_num).center = Vec3f(0, 0, 0);
                                auto halfLength = rb_json.at("size").get<std::vector<float>>();
                                spheres_host(eid, sphere_num).radius = halfLength[0];
                                // spheres_host(eid, sphere_num).rot = batch_quat_host(eid, bid);

                                mass_host.push_back(density * 4. / 3. * M_PI * pow(halfLength[0], 3));

                                sphere_num++;
                                break;
                            }
                            case 1: {
                                shape_type_host(eid, bid) = 1;
                                shape_idx_host(eid, bid) = box_num;

                                boxes_host(eid, box_num).center = Vec3f(0, 0, 0);
                                auto halfLength = rb_json.at("size").get<std::vector<float>>();
                                boxes_host(eid, box_num).halfLength = Vec3f(halfLength[0], halfLength[1], halfLength[2]);
                                // boxes_host(eid, box_num).rot = batch_quat_host(eid, bid);

                                mass_host.push_back(8 * density * halfLength[0] * halfLength[1] * halfLength[2]);

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

                                mass_host.push_back(density * (2 * M_PI * halfLength[1] * halfLength[0] * halfLength[0] + 4. / 3. * M_PI * pow(halfLength[0], 3)));

                                capsule_num++;
                                break;
                            }
                            default: ;
                        }
                        spdlog::info("mass: {}, density: {}", mass_host.back(), density);
                    }

                    if (rb_json.contains("joint")) {
                        auto joint_json = rb_json["joint"];
                        parent_idx_host.push_back(joint_json["parent"].get<int>());

                        if (joint_json.contains("anchor")) {
                            auto joint_anchor_ref = joint_json.at("anchor").get<std::vector<float>>();
                            joint_anchor_ref_host.push_back(Vec3f(joint_anchor_ref[0], joint_anchor_ref[1], joint_anchor_ref[2]));
                        } else
                            joint_anchor_ref_host.push_back(Vec3f(0));

                        if (joint_json.contains("axis")) {
                            auto joint_axis_ref = joint_json.at("axis").get<std::vector<float>>();
                            joint_axis_ref_host.push_back(Vec3f(joint_axis_ref[0], joint_axis_ref[1], joint_axis_ref[2]));
                        } else
                            joint_axis_ref_host.push_back(Vec3f(0));

                        joint_rel_pos_host.push_back(Vec3f(pos[0], pos[1], pos[2]));
                        joint_rel_quat_host.push_back(batch_quat_host(eid, bid));

                        if (joint_json.contains("upper")) {
                            jl_is_upper_host.push_back(1);
                            jl_joint_idx_host.push_back(bid);
                            jl_limit_host.push_back(joint_json.at("upper").get<float>());

                            if (joint_json.contains("sol_paras")) {
                                auto jl_paras = joint_json.at("sol_paras").get<std::vector<float>>();
                                jl_tc_host.push_back(jl_paras[0]);
                                jl_dr_host.push_back(jl_paras[1]);
                                jl_dmax_host.push_back(jl_paras[2]);
                                jl_dmin_host.push_back(jl_paras[3]);
                                jl_width_host.push_back(jl_paras[4]);
                                jl_midpoint_host.push_back(jl_paras[5]);
                                jl_power_host.push_back(static_cast<int>(jl_paras[6]));
                            } else {
                                jl_tc_host.push_back(0.02);
                                jl_dr_host.push_back(1);
                                jl_dmax_host.push_back(0.95);
                                jl_dmin_host.push_back(0.9);
                                jl_width_host.push_back(0.001);
                                jl_midpoint_host.push_back(0.5);
                                jl_power_host.push_back(2);
                            }

                            jl_num++;
                        }

                        if (joint_json.contains("lower")) {
                            jl_is_upper_host.push_back(0);
                            jl_joint_idx_host.push_back(bid);
                            jl_limit_host.push_back(joint_json.at("lower").get<float>());

                            if (joint_json.contains("sol_paras")) {
                                auto jl_paras = joint_json.at("sol_paras").get<std::vector<float>>();
                                jl_tc_host.push_back(jl_paras[0]);
                                jl_dr_host.push_back(jl_paras[1]);
                                jl_dmax_host.push_back(jl_paras[2]);
                                jl_dmin_host.push_back(jl_paras[3]);
                                jl_width_host.push_back(jl_paras[4]);
                                jl_midpoint_host.push_back(jl_paras[5]);
                                jl_power_host.push_back(static_cast<int>(jl_paras[6]));
                            } else {
                                jl_tc_host.push_back(0.02);
                                jl_dr_host.push_back(1);
                                jl_dmax_host.push_back(0.95);
                                jl_dmin_host.push_back(0.9);
                                jl_width_host.push_back(0.001);
                                jl_midpoint_host.push_back(0.5);
                                jl_power_host.push_back(2);
                            }

                            jl_num++;
                        }

                        int type = joint_json.at("type").get<int>();
                        switch(type) {
                            case 1: {
                                joint_type_host.push_back(1);
                                joint_qpos_offset_host.push_back(joint_qpos_offset);

                                auto joint_qpos = joint_json.at("angle").get<std::vector<float>>();
                                joint_qpos_host.push_back(joint_qpos[0]);

                                auto joint_qpos_ref = joint_json.at("angle_ref").get<std::vector<float>>();
                                joint_qpos_ref_host.push_back(joint_qpos_ref[0]);

                                joint_qpos_offset++;
                                break;
                            }

                            case 2: {
                                joint_type_host.push_back(2);
                                joint_qpos_offset_host.push_back(joint_qpos_offset);

                                auto joint_qpos = joint_json.at("displacement").get<std::vector<float>>();
                                joint_qpos_host.push_back(joint_qpos[0]);

                                auto joint_qpos_ref = joint_json.at("displacement_ref").get<std::vector<float>>();
                                joint_qpos_ref_host.push_back(joint_qpos_ref[0]);

                                joint_qpos_offset++;
                                break;
                            }

                            case 3: {
                                joint_type_host.push_back(3);
                                joint_qpos_offset_host.push_back(joint_qpos_offset);

                                auto joint_qpos = joint_json.at("orientation").get<std::vector<float>>();
                                joint_qpos_host.push_back(joint_qpos[0]);
                                joint_qpos_host.push_back(joint_qpos[1]);
                                joint_qpos_host.push_back(joint_qpos[2]);
                                joint_qpos_host.push_back(joint_qpos[3]);

                                auto joint_qpos_ref = joint_json.at("orientation_ref").get<std::vector<float>>();
                                joint_qpos_ref_host.push_back(joint_qpos_ref[0]);
                                joint_qpos_ref_host.push_back(joint_qpos_ref[1]);
                                joint_qpos_ref_host.push_back(joint_qpos_ref[2]);
                                joint_qpos_ref_host.push_back(joint_qpos_ref[3]);

                                joint_qpos_offset += 4;
                                break;
                            }
                            default: ;
                        }
                    }
                    else {
                        parent_idx_host.push_back(-1);
                        joint_type_host.push_back(0);
                        joint_qpos_offset_host.push_back(joint_qpos_offset);
                        joint_rel_pos_host.push_back(Vec3f(0));
                        joint_rel_quat_host.push_back(Quat<Real>::identity());
                        joint_axis_ref_host.push_back(Vec3f(0));
                        joint_anchor_ref_host.push_back(Vec3f(0));
                    }

                    if (rb_json.contains("sol_paras")) {
                        auto paras = rb_json.at("sol_paras").get<std::vector<float>>();
                        col_tc_host.push_back(paras[0]);
                        col_dr_host.push_back(paras[1]);
                        col_dmax_host.push_back(paras[2]);
                        col_dmin_host.push_back(paras[3]);
                        col_width_host.push_back(paras[4]);
                        col_midpoint_host.push_back(paras[5]);
                        col_power_host.push_back(static_cast<int>(paras[6]));
                    } else {
                        col_tc_host.push_back(0.02);
                        col_dr_host.push_back(1);
                        col_dmax_host.push_back(0.95);
                        col_dmin_host.push_back(0.9);
                        col_width_host.push_back(0.001);
                        col_midpoint_host.push_back(0.5);
                        col_power_host.push_back(2);
                    }

                    bid++;
                }

                Vec4i num_each_constraint = Vec4i(0, 0, 0, 0);
                Vec4i constraint_offset = Vec4i(0, 0, 0, 0);

                if(env_json.contains("connect"))
                {
                    int connect_num = 0;
                    connect_anchor_nums_host.push_back(static_cast<int>(env_json["connect"].size()));
                    for (const auto& connect_json : env_json["connect"]) {
                        connect_body_idxs_host.push_back(Pair<int, int>(connect_json.at("bodyA").get<int>(), connect_json.at("bodyB").get<int>()));
                        auto localA = connect_json.at("anchorA").get<std::vector<float>>();
                        connect_anchor_A_local_host.push_back(Vec3f{localA[0], localA[1], localA[2]});
                        auto localB = connect_json.at("anchorB").get<std::vector<float>>();
                        connect_anchor_B_local_host.push_back(Vec3f{localB[0], localB[1], localB[2]});

                        if (connect_json.contains("sol_paras")) {
                            auto paras = connect_json.at("sol_paras").get<std::vector<float>>();
                            connect_tc_host.push_back(paras[0]);
                            connect_dr_host.push_back(paras[1]);
                            connect_dmax_host.push_back(paras[2]);
                            connect_dmin_host.push_back(paras[3]);
                            connect_width_host.push_back(paras[4]);
                            connect_midpoint_host.push_back(paras[5]);
                            connect_power_host.push_back(static_cast<int>(paras[6]));
                        } else {
                            connect_tc_host.push_back(0.02);
                            connect_dr_host.push_back(1);
                            connect_dmax_host.push_back(0.95);
                            connect_dmin_host.push_back(0.9);
                            connect_width_host.push_back(0.001);
                            connect_midpoint_host.push_back(0.5);
                            connect_power_host.push_back(2);
                        }

                        connect_num++;
                    }
                    num_each_constraint.x = connect_num * 3;
                    constraint_offset[1] = num_each_constraint.x;
                }

                if(env_json.contains("friction_loss"))
                {
                    int fl_num = 0;
                    fl_num_host.push_back(static_cast<int>(env_json["friction_loss"].size()));
                    for (const auto& fl_json : env_json["friction_loss"])
                    {
                        fl_dof_idxs_host.push_back(fl_json.at("dof_id").get<int>());
                        fl_dof_frictionloss_host.push_back(fl_json.at("resistance").get<float>());

                        if (fl_json.contains("sol_paras")) {
                            auto paras = fl_json.at("sol_paras").get<std::vector<float>>();
                            fl_tc_host.push_back(paras[0]);
                            fl_dr_host.push_back(paras[1]);
                            fl_dmax_host.push_back(paras[2]);
                            fl_dmin_host.push_back(paras[3]);
                            fl_width_host.push_back(paras[4]);
                            fl_midpoint_host.push_back(paras[5]);
                            fl_power_host.push_back(static_cast<int>(paras[6]));
                        } else {
                            fl_tc_host.push_back(0.02);
                            fl_dr_host.push_back(1);
                            fl_dmax_host.push_back(0.95);
                            fl_dmin_host.push_back(0.9);
                            fl_width_host.push_back(0.001);
                            fl_midpoint_host.push_back(0.5);
                            fl_power_host.push_back(2);
                        }

                        fl_num++;
                    }

                    num_each_constraint.y = fl_num;
                    constraint_offset[2] = constraint_offset[1] + num_each_constraint.y;
                }

                num_each_constraint_host.push_back(num_each_constraint);
                constraint_offset_host.push_back(constraint_offset);
                num_constraints_host.push_back(num_each_constraint.x + num_each_constraint.y);

                batch_bodies_host.push_back(bid);
                env_num_spheres_host.push_back(sphere_num);
                env_num_boxes_host.push_back(box_num);
                env_num_capsules_host.push_back(capsule_num);

                batch_body_offset_host.push_back(total_bodies);
                joint_qpos_num_host.push_back(joint_qpos_offset);

                jl_ref_num_host.push_back(jl_num);

                env_sphere_offset_host[eid] = total_spheres;
                env_box_offset_host[eid] = total_boxes;
                env_capsule_offset_host[eid] = total_capsules;
                joint_offset_host.push_back(total_joint_qpos);
                total_bodies += bid;
                total_spheres += sphere_num;
                total_boxes += box_num;
                total_capsules += capsule_num;
                total_joint_qpos += joint_qpos_offset;
            }
            eid++;
        }

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
                    rendering_idx_2_rigid_body_mapping_host[render_idx] = Vec3i(eid, st, si);
                }
                rigid_body_2_rendering_idx_mapping_host(eid, bid) = render_idx;
            }
        }

        for (eid = 0; eid < env_num; ++eid) {
            for (int bid = 0; bid < batch_bodies_host[eid]; ++bid) {
                const int pid = parent_idx_host[batch_body_offset_host[eid] + bid];
                if (pid != -1) {
                    const auto& parent_quat = batch_quat_host(eid, pid);
                    const auto& parent_rot = body_rot_host(eid, pid);
                    const int& joint_type = joint_type_host[batch_body_offset_host[eid] + bid];
                    const auto& joint_qpos_start = joint_qpos_offset_host[batch_body_offset_host[eid] + bid];
                    const auto& local_axis = joint_axis_ref_host[batch_body_offset_host[eid] + bid];
                    const auto& local_anchor = joint_anchor_ref_host[batch_body_offset_host[eid] + bid];

                    Quat<Real> xquat_p = parent_quat * joint_rel_quat_host[batch_body_offset_host[eid] + bid];
                    Vec3f joint_axis = xquat_p * local_axis;
                    Vec3f xanchor = xquat_p * local_anchor;
                    Vec3f xpos = parent_rot * joint_rel_pos_host[batch_body_offset_host[eid] + bid] + body_pos_host(eid, pid);
                    xanchor += xpos;

                    if (joint_type == 2)
                    {
                        batch_quat_host(eid, bid) = xquat_p;
                        body_rot_host(eid, bid) = xquat_p.toMatrix3x3();
                        body_pos_host(eid, bid) = xpos + (joint_qpos_host[joint_offset_host[eid] + joint_qpos_start] - joint_qpos_ref_host[joint_offset_host[eid] + joint_qpos_start]) * joint_axis;
                    }
                    else
                    {
                        Quat<Real> quat_local;
                        if (joint_type == 1)
                            quat_local.fromAxisAngle(local_axis, joint_qpos_host[joint_offset_host[eid] + joint_qpos_start] - joint_qpos_ref_host[joint_offset_host[eid] + joint_qpos_start]);
                        else if (joint_type == 3) {
                            Quat<Real> ball_quat = Quat<Real>(
                                joint_qpos_host[joint_offset_host[eid] + joint_qpos_start],
                                joint_qpos_host[joint_offset_host[eid] + joint_qpos_start + 1],
                                joint_qpos_host[joint_offset_host[eid] + joint_qpos_start + 2],
                                joint_qpos_host[joint_offset_host[eid] + joint_qpos_start + 3]);
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

        // for (eid = 0; eid < env_num; ++eid) {
        //     for (int i = 0; i < connect_anchor_nums_host[eid]; i++)
        //         printf("anchor_id: %d, bodyA_id: %d, bodyB_id: %d, anchorA: %f %f %f, anchorB: %f %f %f\n",
        //             i, connect_body_idxs_host(eid, i).first, connect_body_idxs_host(eid, i).second,
        //             connect_anchor_A_local_host(eid, i).x, connect_anchor_A_local_host(eid, i).y, connect_anchor_A_local_host(eid, i).z,
        //             connect_anchor_B_local_host(eid, i).x, connect_anchor_B_local_host(eid, i).y, connect_anchor_B_local_host(eid, i).z);
        // }

        shape_type.assign(shape_type_host);
        shape_idx.assign(shape_idx_host);
        parent_idx.Assign(parent_idx_host, batch_bodies_host);

        boxes.assign(boxes_host);
        spheres.assign(spheres_host);
        capsules.assign(capsules_host);

        batch_pos.assign(body_pos_host);
        batch_rot.assign(body_rot_host);
        batch_quat.assign(batch_quat_host);

        spdlog::info("num boxes: {}, num spheres: {}, num capsules: {}", total_boxes, total_spheres, total_capsules);
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

        is_static.Assign(is_static_host, batch_bodies_host);
        batch_mass.Assign(mass_host, batch_bodies_host);

        if (joint_qpos_num_host.size() != 0) {
            joint_qpos.Assign(joint_qpos_host, joint_qpos_num_host);
            joint_qpos_ref.Assign(joint_qpos_ref_host, joint_qpos_num_host);
        }

        joint_type.Assign(joint_type_host, batch_bodies_host);
        joint_qpos_offset.Assign(joint_qpos_offset_host, batch_bodies_host);
        joint_anchor_ref.Assign(joint_anchor_ref_host, batch_bodies_host);
        joint_axis_ref.Assign(joint_axis_ref_host, batch_bodies_host);
        joint_rel_pos.Assign(joint_rel_pos_host, batch_bodies_host);
        joint_rel_quat.Assign(joint_rel_quat_host, batch_bodies_host);

        friction_mu.Assign(friction_mu_host, batch_bodies_host);
        contact_weights.Assign(contact_weights_host, batch_bodies_host);

        if (jl_ref_num_host.size() != 0) {
            joint_limit_constraints.ref_nums.assign(jl_ref_num_host);
            joint_limit_constraints.active_mapping.BuildFromSizes(jl_ref_num_host);
            joint_limit_constraints.is_active.BuildFromSizes(jl_ref_num_host);
            joint_limit_constraints.limit_error.BuildFromSizes(jl_ref_num_host);
            joint_limit_constraints.limit_extern.BuildFromSizes(jl_ref_num_host);
            joint_limit_constraints.joint_idx.Assign(jl_joint_idx_host, jl_ref_num_host);
            joint_limit_constraints.is_upper.Assign(jl_is_upper_host, jl_ref_num_host);
            joint_limit_constraints.limit.Assign(jl_limit_host, jl_ref_num_host);

            joint_limit_constraints.time_const.Assign(jl_tc_host, jl_ref_num_host);
            joint_limit_constraints.damp_ratio.Assign(jl_dr_host, jl_ref_num_host);
            joint_limit_constraints.dmax.Assign(jl_dmax_host, jl_ref_num_host);
            joint_limit_constraints.dmin.Assign(jl_dmin_host, jl_ref_num_host);
            joint_limit_constraints.midpoint.Assign(jl_midpoint_host, jl_ref_num_host);
            joint_limit_constraints.width.Assign(jl_width_host, jl_ref_num_host);
            joint_limit_constraints.power.Assign(jl_power_host, jl_ref_num_host);
        }

        if (connect_anchor_nums_host.size() != 0) {
            anchor_constraints.body_idxs.Assign(connect_body_idxs_host, connect_anchor_nums_host);
            anchor_constraints.anchor_A_local.Assign(connect_anchor_A_local_host, connect_anchor_nums_host);
            anchor_constraints.anchor_B_local.Assign(connect_anchor_B_local_host, connect_anchor_nums_host);
            anchor_constraints.anchor_A_world.BuildFromSizes(connect_anchor_nums_host);
            anchor_constraints.anchor_B_world.BuildFromSizes(connect_anchor_nums_host);
            anchor_constraints.anchor_error.BuildFromSizes(connect_anchor_nums_host);

            anchor_constraints.time_const.Assign(connect_tc_host, connect_anchor_nums_host);
            anchor_constraints.damp_ratio.Assign(connect_dr_host, connect_anchor_nums_host);
            anchor_constraints.dmax.Assign(connect_dmax_host, connect_anchor_nums_host);
            anchor_constraints.dmin.Assign(connect_dmin_host, connect_anchor_nums_host);
            anchor_constraints.midpoint.Assign(connect_midpoint_host, connect_anchor_nums_host);
            anchor_constraints.width.Assign(connect_width_host, connect_anchor_nums_host);
            anchor_constraints.power.Assign(connect_power_host, connect_anchor_nums_host);
        }

        if (fl_num_host.size() != 0) {
            friction_loss_constraints.dof_idxs.Assign(fl_dof_idxs_host, fl_num_host);
            friction_loss_constraints.dof_frictionloss.Assign(fl_dof_frictionloss_host, fl_num_host);

            friction_loss_constraints.time_const.Assign(fl_tc_host, fl_num_host);
            friction_loss_constraints.damp_ratio.Assign(fl_dr_host, fl_num_host);
            friction_loss_constraints.dmax.Assign(fl_dmax_host, fl_num_host);
            friction_loss_constraints.dmin.Assign(fl_dmin_host, fl_num_host);
            friction_loss_constraints.midpoint.Assign(fl_midpoint_host, fl_num_host);
            friction_loss_constraints.width.Assign(fl_width_host, fl_num_host);
            friction_loss_constraints.power.Assign(fl_power_host, fl_num_host);
        }

        collision_constraints.time_const.Assign(col_tc_host, batch_bodies_host);
        collision_constraints.damp_ratio.Assign(col_dr_host, batch_bodies_host);
        collision_constraints.dmax.Assign(col_dmax_host, batch_bodies_host);
        collision_constraints.dmin.Assign(col_dmin_host, batch_bodies_host);
        collision_constraints.midpoint.Assign(col_midpoint_host, batch_bodies_host);
        collision_constraints.width.Assign(col_width_host, batch_bodies_host);
        collision_constraints.power.Assign(col_power_host, batch_bodies_host);

        num_constraints.assign(num_constraints_host);
        num_each_constraint.assign(num_each_constraint_host);
        constraint_offset.assign(constraint_offset_host);

        std::vector<Quat<Real>> local_com_quat_host(total_bodies, Quat<Real>::identity());
        batch_local_com_pos.BuildFromSizes(batch_bodies_host);
        batch_local_com_quat.Assign(local_com_quat_host, batch_bodies_host);

        spdlog::info("Finished initializing rigid body state variables.");
    }

    DEFINE_CLASS(RigidBody)
}