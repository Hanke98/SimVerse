#include "SimNode.h"


namespace dyno {

    template<typename TDataType>
    SimNode<TDataType>::SimNode() : Node()
    {
        ;
    }

    template<typename TDataType>
    SimNode<TDataType>::SimNode(std::string name) : Node()
    {
        this->setName(name);
        spdlog::info("SimNode constructor called for node: {}", name);
        Init();
    }

    template<typename TDataType>
    SimNode<TDataType>::~SimNode()
    {
        ;
    }

    template<typename TDataType>
    void SimNode<TDataType>::Init()
    {
        auto env_infos = var_env_infos.getValue();
        env_infos.num_env = 2;
        var_env_infos.setValue(env_infos);

        InitRigidBody(env_infos.num_env, 3);    // TEST;
        // Initialize rigid body state variables on GPU:
        // auto rigid_body = var_rigid_body.getValue();

        // const uint num_env = static_cast<uint>(env_infos.num_env);
        // const uint num_body = 4;

        // CArray2D<int> h_batch_nv(num_body, num_env);
        // for (uint env_id = 0; env_id < num_env; ++env_id)
        // {
        //     for (uint body_id = 0; body_id < num_body; ++body_id)
        //     {
        //         h_batch_nv(body_id, env_id) = 6;
        //     }
        // }

        // rigid_body.batch_nv.assign(h_batch_nv);
        // var_rigid_body.setValue(rigid_body);
        
    }

    DEFINE_CLASS(SimNode)
}// namespace dyno