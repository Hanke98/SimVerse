# pragma once

#include "../Solver.h"
#include "RigidBody.h"

#include <memory>

namespace dyno {
    template<typename TDataType>
    class RigidSolver : public SolverBase<TDataType>
    {
    public:
        using RigidBodyType = RigidBody<TDataType>;

        RigidSolver(const std::shared_ptr<RigidBodyType>& rigidBody = nullptr)
            : SolverBase<TDataType>(), m_rigidBody(rigidBody) {};
        ~RigidSolver() {};

    protected:
        std::shared_ptr<RigidBodyType> m_rigidBody = nullptr;

    };
}
