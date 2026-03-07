# pragma once
#include "RigidSolver.h"

#include <memory>

namespace dyno
{
    template<typename TDataType>
    class MujocoSolver : public RigidSolver<TDataType>
    {
    public:
        using Base = RigidSolver<TDataType>;
        using RigidBodyType = typename Base::RigidBodyType;

        explicit MujocoSolver(const std::shared_ptr<RigidBodyType>& rigidBody = nullptr)
            : Base(rigidBody) {};
        ~MujocoSolver() {};

        void Init();
    };
}

