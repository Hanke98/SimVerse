#include "RigidBody.h"
#include "Object.h"
#include "DataTypes.h"
#include "../../Utils/utils.h"

namespace dyno
{
    template<typename TDataType>
    void RigidBody<TDataType>::ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num,
        int joint_limit_max, int connect_max)
    {
        spdlog::info("Start initializing rigid body state variables.");
        // printf("connect_num: %d", connect_max);
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

        CArray2D<Real>            friction_mu_host(env_num, body_max_num);
        CArray2D<Real>            contact_weights_host(env_num, body_max_num);

        std::vector<int>         jl_ref_num_host(env_num, 0);
        CArray2D<int>            jl_joint_idx_host(env_num, joint_limit_max);
        CArray2D<int>            jl_is_upper_host(env_num, joint_limit_max);
        CArray2D<Real>           jl_limit_host(env_num, joint_limit_max);

        CArray2D<Real>          jl_tc_host(env_num, joint_limit_max);
        CArray2D<Real>          jl_dr_host(env_num, joint_limit_max);
        CArray2D<Real>          jl_dmax_host(env_num, joint_limit_max);
        CArray2D<Real>          jl_dmin_host(env_num, joint_limit_max);
        CArray2D<Real>          jl_width_host(env_num, joint_limit_max);
        CArray2D<Real>          jl_midpoint_host(env_num, joint_limit_max);
        CArray2D<int>           jl_power_host(env_num, joint_limit_max);

        std::vector<int>                connect_anchor_nums_host(env_num, 0);
        CArray2D<Pair<int, int>>        connect_body_idxs_host(env_num, connect_max);
        CArray2D<Vec3f>                 connect_anchor_A_local_host(env_num, connect_max);
        CArray2D<Vec3f>                 connect_anchor_B_local_host(env_num, connect_max);

        CArray2D<Real>          connect_tc_host(env_num, connect_max);
        CArray2D<Real>          connect_dr_host(env_num, connect_max);
        CArray2D<Real>          connect_dmax_host(env_num, connect_max);
        CArray2D<Real>          connect_dmin_host(env_num, connect_max);
        CArray2D<Real>          connect_width_host(env_num, connect_max);
        CArray2D<Real>          connect_midpoint_host(env_num, connect_max);
        CArray2D<int>           connect_power_host(env_num, connect_max);

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
                contact_weights_host(eid, bid) = 1;
            }

            for (int jl_id = 0; jl_id < joint_limit_max; jl_id++) {
                jl_dmax_host(eid, jl_id) = 0.95;
                jl_dmin_host(eid, jl_id) = 0.9;
                jl_width_host(eid, jl_id) = 0.001;
                jl_midpoint_host(eid, jl_id) = 0.5;
                jl_tc_host(eid, jl_id) = 0.02;
                jl_dr_host(eid, jl_id) = 1.;
                jl_power_host(eid, jl_id) = 2;
            }

            for (int connect_id = 0; connect_id < connect_max; connect_id++) {
                connect_dmax_host(eid, connect_id) = 0.95;
                connect_dmin_host(eid, connect_id) = 0.9;
                connect_width_host(eid, connect_id) = 0.001;
                connect_midpoint_host(eid, connect_id) = 0.5;
                connect_tc_host(eid, connect_id) = 0.02;
                connect_dr_host(eid, connect_id) = 1.;
                connect_power_host(eid, connect_id) = 2;
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
                int jl_num = 0;

                for (const auto& rb_json : env_json["rigid_body"]) {

                    auto pos = rb_json.at("pos").get<std::vector<float>>();
                    body_pos_host(eid, bid) = Vec3f(pos[0], pos[1], pos[2]);
                    auto quat = rb_json.at("quat").get<std::vector<float>>();
                    batch_quat_host(eid, bid) = Quat<Real>(quat[0], quat[1], quat[2], quat[3]);
                    body_rot_host(eid, bid) = batch_quat_host(eid, bid).toMatrix3x3();

                    Real density = rb_json["density"].get<float>();

                    friction_mu_host(eid, bid) = rb_json.at("friction").get<float>();
                    if (rb_json.contains("contact_weight"))
                        contact_weights_host(eid, bid) = rb_json.at("contact_weight").get<float>();

                    bool is_static = rb_json["is_static"].get<bool>();
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

                                mass_host(eid, bid) = density * 4. / 3. * M_PI * pow(halfLength[0], 3);

                                sphere_num++;
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

                                mass_host(eid, bid) = 8 * density * halfLength[0] * halfLength[1] * halfLength[2];
                                spdlog::info("box mass: {}, density: {}, size: {}", mass_host(eid, bid), density, halfLength[0] * 2 * halfLength[1] * 2 * halfLength[2] * 2);
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

                                mass_host(eid, bid) = density * (2 * M_PI * halfLength[1] * halfLength[0] * halfLength[0] + 4. / 3. * M_PI * pow(halfLength[0], 3));

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

                        if (joint_json.contains("axis")) {
                            auto joint_axis_ref = joint_json.at("axis").get<std::vector<float>>();
                            joint_axis_ref_host(eid, bid) = Vec3f(joint_axis_ref[0], joint_axis_ref[1], joint_axis_ref[2]);
                        }

                        joint_rel_pos_host(eid, bid) = body_pos_host(eid, bid);
                        joint_rel_quat_host(eid, bid) = batch_quat_host(eid, bid);

                        if (joint_json.contains("upper")) {
                            jl_is_upper_host(eid, jl_num) = 1;
                            jl_joint_idx_host(eid, jl_num) = bid;
                            jl_limit_host(eid, jl_num) = joint_json.at("upper").get<float>();

                            if (joint_json.contains("jl_paras")) {
                                auto jl_paras = joint_json.at("jl_paras").get<std::vector<float>>();
                                jl_tc_host(eid, jl_num) = jl_paras[0];
                                jl_dr_host(eid, jl_num) = jl_paras[1];
                                jl_dmax_host(eid, jl_num) = jl_paras[2];
                                jl_dmin_host(eid, jl_num) = jl_paras[3];
                                jl_width_host(eid, jl_num) = jl_paras[4];
                                jl_midpoint_host(eid, jl_num) = jl_paras[5];
                                jl_power_host(eid, jl_num) = static_cast<int>(jl_paras[6]);
                            }

                            jl_num++;
                        }

                        if (joint_json.contains("lower")) {
                            jl_is_upper_host(eid, jl_num) = 0;
                            jl_joint_idx_host(eid, jl_num) = bid;
                            jl_limit_host(eid, jl_num) = joint_json.at("lower").get<float>();
                            jl_num++;
                        }

                        jl_ref_num_host[eid] = jl_num;

                        int type = joint_json.at("type").get<int>();
                        switch(type) {
                            case 1: {
                                joint_type_host(eid, bid) = 1;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("angle").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];

                                auto joint_qpos_ref = joint_json.at("angle_ref").get<std::vector<float>>();
                                joint_qpos_ref_host(eid, joint_qpos_offset) = joint_qpos_ref[0];

                                joint_qpos_offset++;
                                break;
                            }

                            case 2: {
                                joint_type_host(eid, bid) = 2;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("displacement").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];

                                auto joint_qpos_ref = joint_json.at("displacement_ref").get<std::vector<float>>();
                                joint_qpos_ref_host(eid, joint_qpos_offset) = joint_qpos_ref[0];

                                joint_qpos_offset++;
                                break;
                            }

                            case 3: {
                                joint_type_host(eid, bid) = 3;
                                joint_qpos_offset_host(eid, bid) = joint_qpos_offset;

                                auto joint_qpos = joint_json.at("orientation").get<std::vector<float>>();
                                joint_qpos_host(eid, joint_qpos_offset) = joint_qpos[0];
                                joint_qpos_host(eid, joint_qpos_offset + 1) = joint_qpos[1];
                                joint_qpos_host(eid, joint_qpos_offset + 2) = joint_qpos[2];
                                joint_qpos_host(eid, joint_qpos_offset + 3) = joint_qpos[3];

                                auto joint_qpos_ref = joint_json.at("orientation_ref").get<std::vector<float>>();
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

                int connect_num = 0;
                connect_anchor_nums_host[eid] = static_cast<int>(env_json["connect"].size());
                for (const auto& connect_json : env_json["connect"]) {
                    connect_body_idxs_host(eid, connect_num) = Pair<int, int>(connect_json.at("bodyA").get<int>(), connect_json.at("bodyB").get<int>());
                    auto localA = connect_json.at("anchorA").get<std::vector<float>>();
                    connect_anchor_A_local_host(eid, connect_num) = Vec3f{localA[0], localA[1], localA[2]};
                    auto localB = connect_json.at("anchorB").get<std::vector<float>>();
                    connect_anchor_B_local_host(eid, connect_num) = Vec3f{localB[0], localB[1], localB[2]};
                    connect_num++;
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

        for (eid = 0; eid < env_num; ++eid) {
            for (int i = 0; i < connect_anchor_nums_host[eid]; i++)
                printf("anchor_id: %d, bodyA_id: %d, bodyB_id: %d, anchorA: %f %f %f, anchorB: %f %f %f\n",
                    i, connect_body_idxs_host(eid, i).first, connect_body_idxs_host(eid, i).second,
                    connect_anchor_A_local_host(eid, i).x, connect_anchor_A_local_host(eid, i).y, connect_anchor_A_local_host(eid, i).z,
                    connect_anchor_B_local_host(eid, i).x, connect_anchor_B_local_host(eid, i).y, connect_anchor_B_local_host(eid, i).z);
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

        friction_mu.assign(friction_mu_host);
        contact_weights.assign(contact_weights_host);

        joint_limit_constraints.ref_nums.assign(jl_ref_num_host);
        INIT_DYNO_ARRAY2D(joint_limit_constraints.active_mapping, env_num, joint_limit_max);
        INIT_DYNO_ARRAY2D(joint_limit_constraints.is_active, env_num, joint_limit_max);
        joint_limit_constraints.joint_idx.assign(jl_joint_idx_host);
        joint_limit_constraints.is_upper.assign(jl_is_upper_host);
        joint_limit_constraints.limit.assign(jl_limit_host);
        INIT_DYNO_ARRAY2D(joint_limit_constraints.limit_error, env_num, joint_limit_max);
        INIT_DYNO_ARRAY2D(joint_limit_constraints.limit_extern, env_num, joint_limit_max);

        joint_limit_constraints.time_const.assign(jl_tc_host);
        joint_limit_constraints.damp_ratio.assign(jl_dr_host);
        joint_limit_constraints.dmax.assign(jl_dmax_host);
        joint_limit_constraints.dmin.assign(jl_dmin_host);
        joint_limit_constraints.midpoint.assign(jl_midpoint_host);
        joint_limit_constraints.width.assign(jl_width_host);
        joint_limit_constraints.power.assign(jl_power_host);

        anchor_constraints.anchor_nums.assign(connect_anchor_nums_host);
        anchor_constraints.body_idxs.assign(connect_body_idxs_host);
        anchor_constraints.anchor_A_local.assign(connect_anchor_A_local_host);
        anchor_constraints.anchor_B_local.assign(connect_anchor_B_local_host);
        INIT_DYNO_ARRAY2D(anchor_constraints.anchor_A_world, env_num, connect_max);
        INIT_DYNO_ARRAY2D(anchor_constraints.anchor_B_world, env_num, connect_max);
        INIT_DYNO_ARRAY2D(anchor_constraints.anchor_error, env_num, connect_max);

        anchor_constraints.time_const.assign(connect_tc_host);
        anchor_constraints.damp_ratio.assign(connect_dr_host);
        anchor_constraints.dmax.assign(connect_dmax_host);
        anchor_constraints.dmin.assign(connect_dmin_host);
        anchor_constraints.midpoint.assign(connect_midpoint_host);
        anchor_constraints.width.assign(connect_width_host);
        anchor_constraints.power.assign(connect_power_host);

        spdlog::info("Finished initializing rigid body state variables.");
    }

    DEFINE_CLASS(RigidBody)
}