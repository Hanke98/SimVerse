#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "SimNode/SimNode.h"


namespace dyno {
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
        int joint_limit_max = 0;
        int connect_max = 0;

        for (const auto& env_json : envs_arr) {
            if (env_json.contains("time_step")) {
                timesteps[env_idx] = env_json["time_step"].get<float>();
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
                int joint_limit_num = 0;

                if (env_json.contains("connect")) {
                    int connect_num = static_cast<int>(env_json["connect"].size());
                    if (connect_num > connect_max) {connect_max = connect_num;}
                }

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

                    if (rb_json.contains("joint")) {
                        auto joint_json = rb_json["joint"];
                        if (joint_json.contains("lower"))
                            joint_limit_num++;
                        if (joint_json.contains("upper"))
                            joint_limit_num++;
                    }
                }
                if (sphere_num > sphere_num_max) {sphere_num_max = sphere_num;}
                if (box_num > box_num_max) {box_num_max = box_num;}
                if (capsule_num > capsule_num_max) {capsule_num_max = capsule_num;}
                if (joint_limit_num > joint_limit_max) {joint_limit_max = joint_limit_num;}
            }

            env_idx++;
        }

        env_infos.gravities.assign(gravities);
        env_infos.timesteps.assign(timesteps);

        var_env_infos.setValue(env_infos);

        std::vector<int> primitive_max_num{sphere_num_max, box_num_max, capsule_num_max};

        // printf("connect_max %d\n", connect_max);

        auto rigid_body = var_rigid_body.getValue();
        rigid_body.ParseRigidBody(envs_arr, body_num_max, primitive_max_num, joint_limit_max, connect_max);
        var_rigid_body.setValue(rigid_body);
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

//     template class SimNode<
//     DataTypes<float, Vector<float,3>, SquareMatrix<float,3>, Rigid<float,3>>
// >;
    DEFINE_CLASS(SimNode)
}
