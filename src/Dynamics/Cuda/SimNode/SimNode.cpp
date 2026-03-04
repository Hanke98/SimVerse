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
    }

    template<typename TDataType>
    SimNode<TDataType>::~SimNode()
    {
        ;
    }

    DEFINE_CLASS(SimNode)
}// namespace dyno