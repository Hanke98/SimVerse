# pragma once

#include "../Solver.h"
#include "RigidBody.h"
#include "../../Utils/tepy.h"

#include <memory>

namespace dyno {

    template<typename TDataType>
    class RigidSolver : public SolverBase<TDataType>
    {
    public:
        using EnvInfosType = EnvironmentInfos<TDataType>;
        using RigidBodyType = RigidBody<TDataType>;

        RigidSolver(
            const std::shared_ptr<EnvInfosType>& envInfos = nullptr,
            const std::shared_ptr<RigidBodyType>& rigidBody = nullptr)
            : SolverBase<TDataType>(), m_envInfos(envInfos), m_rigidBody(rigidBody) {};
        ~RigidSolver() {};

    protected:
        std::shared_ptr<EnvInfosType> m_envInfos = nullptr;
        std::shared_ptr<RigidBodyType> m_rigidBody = nullptr;

    };
}
