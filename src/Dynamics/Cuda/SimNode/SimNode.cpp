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
        std::cout << "SimNode constructor called for node: " << name << std::endl;
    }

    template<typename TDataType>
    SimNode<TDataType>::~SimNode()
    {
        ;
    }

    DEFINE_CLASS(SimNode)
}// namespace dyno