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

        RobotArmSimulator();
        ~RobotArmSimulator();

        void initBatchSolver();

        void addRobotArmRigidBodies(std::string urdf_fn,
                                    Vec3f base,
                                    Vec3f offset,
                                    float density = 1000,
                                    int num_copies_x = 1,
                                    int num_copies_y = 1,
                                    int num_copies_z = 1);

        void resetStates(CtrlParam& param);

        // 场景创建相关接口
        void createScene();

        // 仿真控制接口
        void setupSceneGraph();
        void initialize(int width = 1280, int height = 768, float scale = 1.0f);
        void stepSimulation(bool enableRendering = true);
        void terminateSimulation();
        void applyHingeTorques(CtrlHingeParam& param);

        std::shared_ptr<SceneGraph> activeScene;
        // -------------getters---------------
        std::vector<Vec3f> getCentersByLocalIndex(LocalIndexParam& param);
        std::vector<Vec3f> getVelocitiesByLocalIndex(LocalIndexParam& param);
        std::vector<TQuat> getAngelsByLocalIndex(LocalIndexParam& param);
        std::vector<Vec3f> getAngularVelocitiesByLocalIndex(LocalIndexParam& param);
        std::vector<float> getMassByLocalIndex(LocalIndexParam& param);
        UrdfInformation getKinematicsChainInfo();

        // -------------setters---------------
        void setAngularDamping(Real damping);
        void setDt(Real dt);
        void enableGravity(bool flag);
        void enableFriction(bool flag);

    private:
        std::shared_ptr<BatchRigidBodySystem<TDataType>> batchSolver;
        std::shared_ptr<SceneGraph> scn;
        UbiApp app;
        bool isInitialized = false;
    };
} // namespace dyno
