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
        using EnvInfosType = typename Base::EnvInfosType;
        using RigidBodyType = typename Base::RigidBodyType;

        MujocoSolver(
            const std::shared_ptr<EnvInfosType>& envInfos = nullptr,
            const std::shared_ptr<RigidBodyType>& rigidBody = nullptr)
            : Base(envInfos, rigidBody) {};
        ~MujocoSolver() {};

        void Init() override;
        void Step() override;
    
    private:
        void ForwardKinematics();
        
        void NewtonSolver();

        void MakeConstraints();
        void MakeJacobian();
        void ComputeAref();
        void ComputeRD();
        void ComputeEnergy();
        void BuildHessian();
        void UpdateGradient();
        void SolveSystem();
        void TimeIntegration();

    private:
        std::shared_ptr<BatchedCholeskySolver<typename TDataType::Real>> cholesky_solver = nullptr;
    };
}

