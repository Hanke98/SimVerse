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
 *
 */

#pragma once
#include "Module/ComputeModule.h"
#include "SimNode.h"

namespace dyno {

    template<typename TDataType>
    class SimModule : public ComputeModule
    {
        DECLARE_TCLASS(SimModule, TDataType)

    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename TDataType::Matrix Matrix;


        SimModule();
        ~SimModule() override;

    public:
        DEF_VAR_IN(typename SimNode<TDataType>::EnvInfosType, env_infos, "A struct containing the infos of parallel environments.");

        DEF_VAR_IN(RigidBody<TDataType>, rigid_body, "A struct containing infos about the rigid body in all environments.");


    protected:
        void compute() override;
    
        void AdvanceOneStep();
        
    private:
        inline static uint frame = 0;
    };

}// namespace dyno