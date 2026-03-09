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
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemHingeTorqueControlParam HingeTorqueParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemHingeVelocityControlParam HingeVelocityParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemLocalIndexParam LocalIndexParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemMassParam MassParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemInertiaParam InertiaParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemResetParam ResetParam;
        typedef typename BatchRigidBodySystem<TDataType>::BatchRigidBodySystemHingeInitGestureParam InitHingeParam;


        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename dyno::Quat<Real> TQuat;

        RobotArmSimulator();
        ~RobotArmSimulator();

        void initBatchSolver();

        void addRobotArmRigidBodies(std::string urdf_fn,
                                    Real density = 1000,
                                    const std::vector<Vec3f> &target_position = {},
                                    bool render_boundingBox = true,
                                    bool visual_or_collision = true);

        void loadUrdf(std::string urdf_fn);

        void resetStates(ResetParam& param);
        void resetTargets(ResetParam& param);

        // 场景创建相关接口
        void createScene();

        // 仿真控制接口
        void setupSceneGraph();
        void initialize(int width = 1280, int height = 768, Real scale = 1.0f);
        void stepSimulation(bool enableRendering = true, bool enableSaveScreen = false, std::string savePath = getAssetPath() + "../examples/Cuda/RigidBody/RobotArm_Impulse/screenSave/");
        void terminateSimulation();

        std::shared_ptr<SceneGraph> activeScene;
        // -------------getters---------------
        std::vector<Vec3f> getCentersByLocalIndex(LocalIndexParam& param);
        std::vector<Vec3f> getVelocitiesByLocalIndex(LocalIndexParam& param);
        std::vector<TQuat> getAnglesByLocalIndex(LocalIndexParam& param);
        std::vector<std::vector<Real>> getAnglesVectorByLocalIndex(LocalIndexParam& param);
        std::vector<Vec3f> getAngularVelocitiesByLocalIndex(LocalIndexParam& param);
        std::vector<Real> getMassByLocalIndex(LocalIndexParam& param);
        UrdfInformation getKinematicsChainInfo();
        std::vector<Transform3f> getTransform(CtrlParam& tran_param);

        // -------------setters---------------
        void setAngularDamping(Real damping);
        void setDt(Real dt);
        void enableGravity(bool flag);
        void enableFriction(bool flag);
        void setVelocitySolverIterations(uint iterations);
        void setDisableContactReduction(bool disable);
        void setHingeTorques(HingeTorqueParam& hingetorque_param);
        void setHingeVelocities(HingeVelocityParam& hingevelocity_param);
        void setMass(MassParam& mass_param);
        void setInertia(InertiaParam& inertia_param);
        void setTransform(Vec3f base, Vec3f offset, int num_copies_x, int num_copies_y, int num_copies_z);
        void setInitGesture(InitHingeParam& hinge_param);
        void isObjYUp(bool objYUp);
        void setRenderVisualOrCollision(bool visual_or_collision);

    private:
        std::shared_ptr<BatchRigidBodySystem<TDataType>> batchSolver;
        std::shared_ptr<SceneGraph> scn;
        UbiApp app;
        bool isInitialized = false;
        bool mSimulationRunning = true;
        bool mSpacePressedLastFrame = false;
    };
} // namespace dyno
