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

#include "PhysicalField/RigidBody/RigidBody.h"
#include "Utils/tepy.h"

#include <vector>
#include <iostream>
#include <spdlog/spdlog.h>

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

        SimNode();
        SimNode(std::string name);
        ~SimNode() override;

    protected:
        void Init();

        void InitRigidBody(int num_env, int num_bodies);    // TEST;


        void BindRenderingSurface(int num_env);    // TODO: collect shape information for rendering.


    
    public:
        DEF_VAR(EnvInfosType, env_infos, EnvInfosType{}, "A struct containing the infos of parallel environments.");
        
        // Rigid Body
        DEF_VAR(RigidBody<TDataType>, rigid_body, RigidBody<TDataType>{}, "A struct containing infos about the rigid body in all environments.");
    
        
        // For DynoRendering
        DEF_INSTANCE_STATE(DiscreteElements<TDataType>, Topology, "Topology");

    private:
        ;

    };


}// namespace dyno