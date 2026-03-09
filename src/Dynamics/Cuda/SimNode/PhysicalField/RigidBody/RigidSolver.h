# pragma once

#include "../Solver.h"
#include "RigidBody.h"
#include "../../Utils/type.h"

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
            : SolverBase<TDataType>(), env_infos(envInfos), rigid_body(rigidBody) {};
        ~RigidSolver() {};

    protected:
        std::shared_ptr<EnvInfosType> env_infos = nullptr;
        std::shared_ptr<RigidBodyType> rigid_body = nullptr;

    };
}
