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
        this->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());

        auto env_infos = var_env_infos.getValue();
        env_infos.num_env = 2;
        var_env_infos.setValue(env_infos);

        InitRigidBody(env_infos.num_env, 3);    // TEST;

        BindRenderingSurface(env_infos.num_env);
        
    }

    DEFINE_CLASS(SimNode)
}// namespace dyno