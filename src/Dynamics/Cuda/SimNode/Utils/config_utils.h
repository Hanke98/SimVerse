#pragma once

#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>

#include "SimNode/SimNode.h"


namespace dyno {
    template<typename TDataType>
    void SimNode<TDataType>::ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num) {
        spdlog::info("Start initializing rigid body state variables.");
        // std::cout << "primitive max_num: " << primitive_max_num[0] << primitive_max_num[1] << primitive_max_num[2] << std::endl;

        auto rigid_bodies = var_rigid_body.getValue();

        const auto& env_infos = var_env_infos.constDataPtr();
        int env_num = env_infos->num_envs;

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

                for (const auto& rb_json : env_json["rigid_body"]) {

                    auto pos = rb_json.at("pos").get<std::vector<float>>();
                    body_pos_host(eid, bid) = Vec3f(pos[0], pos[1], pos[2]);
                    auto quat = rb_json.at("quat").get<std::vector<float>>();
                    batch_quat_host(eid, bid) = Quat<Real>(quat[1], quat[2], quat[3], quat[0]);

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
                                spheres_host(eid, sphere_num).rot = batch_quat_host(eid, bid);

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
                                boxes_host(eid, box_num).rot = batch_quat_host(eid, bid);

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
                                capsules_host(eid, capsule_num).rot = batch_quat_host(eid, bid);

                                capsule_num++;
                                break;
                            }
                            default: ;
                        }
                        bid++;
                    }
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

        // std::cout << "write back" << std::endl;

        rigid_bodies.shape_type.assign(shape_type_host);
        rigid_bodies.shape_idx.assign(shape_idx_host);
        rigid_bodies.parent_idx.assign(parent_idx_host);

        rigid_bodies.boxes.assign(boxes_host);
        rigid_bodies.spheres.assign(spheres_host);
        rigid_bodies.capsules.assign(capsules_host);

        rigid_bodies.batch_pos.assign(body_pos_host);
        rigid_bodies.batch_rot.assign(body_rot_host);
        rigid_bodies.batch_quat.assign(batch_quat_host);

        rigid_bodies.env_num_boxes.assign(env_num_boxes_host);
        rigid_bodies.env_box_offset.assign(env_box_offset_host);
        rigid_bodies.env_num_spheres.assign(env_num_spheres_host);
        rigid_bodies.env_sphere_offset.assign(env_sphere_offset_host);
        rigid_bodies.env_num_capsules.assign(env_num_capsules_host);
        rigid_bodies.env_capsule_offset.assign(env_capsule_offset_host);

        rigid_bodies.batch_bodies.assign(batch_bodies_host);
        rigid_bodies.batch_body_offset.assign(batch_body_offset_host);

        rigid_bodies.rendering_idx_2_rigid_body_mapping.assign(rendering_idx_2_rigid_body_mapping_host);
        rigid_bodies.rigid_body_2_rendering_idx_mapping.assign(rigid_body_2_rendering_idx_mapping_host);

        rigid_bodies.is_static.assign(is_static_host);
        rigid_bodies.batch_mass.assign(mass_host);

        var_rigid_body.setValue(rigid_bodies);

        spdlog::info("Finished initializing rigid body state variables.");
    }

    template<typename TDataType>
    void SimNode<TDataType>::ParseEnv(const json& envs_json) {
        auto env_infos = var_env_infos.getValue();
        const json& envs_arr = envs_json["envs"];
        env_infos.num_envs = static_cast<int>(envs_arr.size());

        CArray<Vec3f> gravities(env_infos.num_envs);
        CArray<Real> timesteps(env_infos.num_envs);

        int env_idx = 0;
        int body_num_max = 0;
        int box_num_max = 0;
        int sphere_num_max = 0;
        int capsule_num_max = 0;

        for (const auto& env_json : envs_arr) {
            if (env_json.contains("time_step")) {
                timesteps[env_idx] = env_json["time_step"];
            }

            if (env_json.contains("gravity")) {
                auto gravity_vec = env_json.at("gravity").get<std::vector<float>>();
                gravities[env_idx] = Vec3f(gravity_vec[0], gravity_vec[1], gravity_vec[2]);
            }

            if (env_json.contains("rigid_body") && env_json["rigid_body"].is_array()) {
                int body_num = static_cast<int>(env_json["rigid_body"].size());
                if (body_num > body_num_max) {body_num_max = body_num;}

                int sphere_num = 0;
                int box_num = 0;
                int capsule_num = 0;
                for (const auto& rb_json : env_json["rigid_body"]) {
                    if (rb_json.at("type") == "primitive") {
                        int id = rb_json.at("ID").get<int>();
                        switch (id) {
                            case 0:
                                sphere_num++;
                                break;
                            case 1:
                                box_num++;
                                break;
                            case 2:
                                capsule_num++;
                                break;
                            default: ;
                        }
                    }
                }
                if (sphere_num > sphere_num_max) {sphere_num_max = sphere_num;}
                if (box_num > box_num_max) {box_num_max = box_num;}
                if (capsule_num > capsule_num_max) {capsule_num_max = capsule_num;}
            }

            env_idx++;
        }

        var_env_infos.setValue(env_infos);

        std::vector<int> primitive_max_num{sphere_num_max, box_num_max, capsule_num_max};

        // std::cout << "env_num: " << env_infos.num_envs << " body_num: " << body_num_max << std::endl;
        ParseRigidBody(envs_arr, body_num_max, primitive_max_num);
    }

    template<typename TDataType>
    void SimNode<TDataType>::ParseJson(const std::string& file_path) {
        try {
            std::ifstream json_file(file_path);
            if (!json_file.is_open()) {
                throw std::runtime_error("Unable to open file: " + file_path);
            }

            json root;
            json_file >> root;
            if (root.contains("envs") && root["envs"].is_array()) {
                ParseEnv(root);
            }
            json_file.close();

        } catch (const std::runtime_error& e) {
            std::cerr << "File error: " << e.what() << std::endl;
        } catch (const json::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Data error: " << e.what() << std::endl;
        }
    }

    template<typename TDataType>
    void SimNode<TDataType>::LoadAssets(const std::string &root_dir) {
        try {
            if (!std::filesystem::exists(root_dir) || !std::filesystem::is_directory(root_dir)) {
                throw std::runtime_error("root path invalid: " + root_dir);
            }

            for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root_dir)) {
                if (entry.is_regular_file()) {
                    const std::filesystem::path& file_path = entry.path();
                    std::string ext = file_path.extension().string();

                    if (ext == ".json") {
                        ParseJson(file_path.string());
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "File System Error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
}
