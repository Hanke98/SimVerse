// CartPoleSimulator.h
#pragma once

#include "RigidBody/BatchRigidBodySystem.h"
#include "Vector/Vector3D.h"
#include <RigidBody/MultibodySystem.h>
#include <RigidBody/RigidBodySystem.h>
#include <RigidBody/Robot.h>
#include <SceneGraph.h>
#include <UbiApp.h>
#include <memory>
#include <unordered_map>
#include <vector>
// #include "Mapping/DiscreteElementsToTriangleSet.h"
#include "Topology/DiscreteElements.h"
#include <BasicShapes/SphereModel.h>

namespace dyno
{
    template<typename TDataType>
    class RobotArmSimulator : virtual public ArticulatedBody<TDataType> {
        DECLARE_TCLASS(RobotArmSimulator, TDataType)
    public:
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemControlParamBase CtrlParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemHingeTorqueControlParam CtrlHingeParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemLocalIndexParam LocalIndexParam;

        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename dyno::Quat<Real> TQuat;

        DEF_VAR_OUT(bool, Reset, "Reset");

        RobotArmSimulator();
        ~RobotArmSimulator();

        void initBatchSolver(Real dt, Real density);
        void resetStates(CtrlParam& param);
        // void resetStates() override;
        // 场景创建相关接口
        void createScene();
        int addRigidSystem(const Vec3f& offset, Vec3f& targetPosition, float denstiy); // 返回新创建的rigidID
        int addMultiBoydSystem();
        void reset(int rigidIDs, Vec3f& targetPosition); // -1表示重置所有

        // 仿真控制接口
        void setupSceneGraph();
        void initialize(int width = 1280, int height = 768, float scale = 1.0f);
        // void stepSimulation(const std::vector<float>& forces, bool enableRendering = true);
        void stepSimulation(std::vector<std::vector<float>>& moterVelocities, bool enableRendering = true);
        void terminateSimulation();


        void applyImpulse(std::vector<std::vector<float>>& moterImpulses);

        void applyHingeTorques(CtrlHingeParam& param);

        std::shared_ptr<SceneGraph> activeScene;
        int getRigidSystemCount() const
        {
            return rigidSystems.size();
        }

        // void resetState(int rigidID);

        std::vector<TQuat> getAngelsByLocalIndex(LocalIndexParam& param);
        std::vector<Vec3f> getAngularVelocitiesByLocalIndex(LocalIndexParam& param);

        UrdfInformation getKinematicsChainInfo();

    public:
        DEF_VAR(Coord, TargetCenter, 0, "Target center");
        // DEF_VAR(Coord, FingerCenter, 0, "Finger center");
        DEF_VAR(Real, Density, 1000.0f, "Density of the rigid body");

    private:
        std::shared_ptr<BatchRigidBodySystem<TDataType>> batchSolver;
        struct RigidSystemData
        {
            std::shared_ptr<MultibodySystem<DataType3f>> system;
            std::shared_ptr<RobotArmSimulator<DataType3f>> robot;
        };

        std::shared_ptr<SceneGraph> scn;
        std::unordered_map<int, RigidSystemData> rigidSystems;
        int nextRigidID = 0;

        std::shared_ptr<MultibodySystem<DataType3f>> mbSystem;
        UbiApp app;
        bool isInitialized = false;

        // 辅助方法
        int generateRigidID();
        RigidSystemData createSingleRigidSystem(int index, const Vec3f& offset, Vec3f& targetPosition, float density);
        void resetSingleRigidSystem(int index, const Vec3f& offset, const Vec3f& targetPosition);

        // SphereModel<DataType3f>> target;

        Vec3f m_offset = Vec3f(0.0f, 0.0f, 0.0f);

        std::vector<std::vector<float>> jointsInitia;

        std::vector<std::vector<float>> motersVelocity;

        std::vector<float> hingeJointsMinAngles{ -2.8973, -1.7628, -2.8973, -0.087, -2.8973, -3.7525, -2.8973 };
        std::vector<float> hingeJointsMaxAngles{ 2.8973, 1.7628, 2.8973, 3.0, 2.8973, 0.0175, 2.8973 };

        std::vector<int> Link_main = { 8, 12, 13, 14, 19, 24, 25, 49, 53, 55, 57 };
    };
} // namespace dyno
