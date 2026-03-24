/**
 * Copyright 2026 Xinming Pei
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "Node.h"
#include "Vector.h"
#include "Array/Array.h"
#include "Field.h"
#include "Topology/DiscreteElements.h"
#include "Topology/EdgeSet.h"

#include "PhysicalField/RigidBody/RigidBody.h"
#include "Utils/type.h"
#include "Utils/json.hpp"

#include <vector>
#include <iostream>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace dyno
{
    typedef typename ::dyno::TSphere3D<Real> Sphere3D;
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;


    /*!
    *	\class	SimNode
    *	\brief	An integrated multi-physics field Node for simulaiton,
                designed for fast simulation and robotics learning,
                developed by SimVerse-ZJU.
    *
    */
    template<typename TDataType>
    class SimNode : public Node
    {
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename TDataType::Matrix Matrix;



        using EnvInfosType = EnvironmentInfos<TDataType>;
        using RigidBodyType = RigidBody<TDataType>;

        SimNode();
        SimNode(std::string name);
        ~SimNode() override;

    protected:
        void Init();
        void Init(const std::string &root_dir);

        void ParseRigidBody(const json& envs_json, int body_max_num, std::vector<int> primitive_max_num);

        void ParseEnv(const json& envs_json);

        void ParseJson(const std::string& file_path);

        void LoadAssets(const std::string &root_dir);

        void InitRigidBody(int num_env, int num_bodies);    // TEST;


        void BindRenderingSurface(int num_env);    // TODO: collect shape information for rendering.
        void PlotWorldAxes();   


    
    public:
        DEF_VAR(EnvInfosType, env_infos, EnvInfosType{}, "A struct containing the infos of parallel environments.");
        
        // Rigid Body
        DEF_VAR(RigidBodyType, rigid_body, RigidBodyType{}, "A struct containing infos about the rigid body in all environments.");
    
        
        // For DynoRendering
        DEF_INSTANCE_STATE(DiscreteElements<TDataType>, topology, "Topology");

        
    private:
        DEF_INSTANCE_STATE(EdgeSet<TDataType>, axis_x, "Edge set for rendering x-axis");
        DEF_INSTANCE_STATE(EdgeSet<TDataType>, axis_y, "Edge set for rendering y-axis");
        DEF_INSTANCE_STATE(EdgeSet<TDataType>, axis_z, "Edge set for rendering z-axis");

    };


}// namespace dyno