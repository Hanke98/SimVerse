#pragma once
#include "Node.h"
#include "RigidBody/ArticulatedBody.h"
#include "RigidBody/MultibodySystem.h"
#include "RigidBodyShared.h"
#include "RigidBodySystem.h"

#include "Collision/Attribute.h"
#include "Collision/CollisionData.h"

#include <iostream>
#include <vector>
namespace dyno
{
  template<typename TDataType>
  class BatchRigidBodySystem : virtual public ArticulatedBody<TDataType> {
public:
    typedef typename TDataType::Real Real; 
    typedef typename TDataType::Coord Coord;
    typedef typename TDataType::Matrix Matrix;
    typedef typename dyno::Quat<Real> TQuat;

    struct MulitBodyChainIndices
    {
      std::vector<int> body_indices;
      std::vector<int> hinge_joint_indices;
      std::vector<int> ball_joint_indices;
      std::vector<int> slider_joint_indices;
      std::vector<int> fixed_joint_indices;
    };

    struct NonCtrlMultiBodyStatesInfo
    {
      int idx;
      int type;
    };

    struct BatchRigidBodySystemControlParamBase
    {
      int num_bodies = 0;
      std::vector<int> ids;
    };

    struct BatchRigidBodySystemResetParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<Coord> targetPosition;
    };

    struct BatchRigidBodySystemTorqueControlParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<Coord> torques;
    };

    struct BatchRigidBodySystemHingeTorqueControlParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<float>> torques;
    };

    struct BatchRigidBodySystemMassParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<float>> mass;
    };

    struct BatchRigidBodySystemInertiaParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<Matrix>> inertia;
    };

    struct BatchRigidBodySystemLocalIndexParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<int> localRigidBodyid;
    };

    BatchRigidBodySystem();
    ~BatchRigidBodySystem() override;

		DEF_VAR(FilePath, UrdfFilePath, "", "");

    // ------------------------------------
    // Control APIs
	  void createBatchMultiBodies(Coord base, Coord offset, int num_x, int num_y, int num_z);

    void resetBatchMultiBodies(BatchRigidBodySystemResetParam& reset_param);
  
    void applyTorqueControl(BatchRigidBodySystemTorqueControlParam& torque_param);

    void applyHingeTorqueControl(BatchRigidBodySystemHingeTorqueControlParam& torque_param);

    void setMass(BatchRigidBodySystemMassParam& mass_param);

    void setInertia(BatchRigidBodySystemInertiaParam& inertia_param);

    void addExampleRigidBodies(std::string urdf_fn, Vec3f base, Vec3f offset, float density, std::vector<Vec3f> targetPosition,
                               int num_copies_x, int num_copies_y, int num_copies_z);
    void addRobotArmRigidBodies(std::string urdf_fn, float density, std::vector<Vec3f> targetPosition);
    // ------------------------------------

    // ------------------------------------
    // Setters and Getters
    // void setDt(Real dt)
    // {
    // }

    void setGravityEnabled(bool enabled)
    {
      this->varGravityEnabled()->setValue(enabled);
    }

    Array<Vec3f, DeviceType::CPU> gethCenters();
    Array<TQuat, DeviceType::CPU> gethAngles();
    Array<Vec3f, DeviceType::CPU> gethVelocities();
    Array<Vec3f, DeviceType::CPU> gethAngularVelocities();
    Array<Mat3f, DeviceType::CPU> gethRotationMatrix();

    Array<float, DeviceType::CPU> gethMass();
    Array<Mat3f, DeviceType::CPU> gethInertia();

    std::vector<TQuat> getAnglesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getAngularVelocitiesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getVelocitiesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getCentersByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<float> getMassByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);

    // ------------------------------------

protected:

    void resetOneMultiBodies(int mb_id);

	  void createOneMultiBody(Coord base, Coord offset);

    void applyOneMultiBodyTorqueControl(int mb_id, Coord torque);

protected:
    std::vector<MulitBodyChainIndices> ctrl_mb_chains; // main multi-body chains with control
    std::vector<MulitBodyChainIndices> non_ctrl_mb_chains; // other multi-body chains in the env.

    std::vector<Vec3f> initialPositions; // initial position of all rigid bodies
    std::vector<TQuat> initialQuats; // initial rotation quaternion of all rigid bodies
    std::vector<Matrix> initialRotations; // initial rotation matrix of all rigid bodies
  };

} // namespace dyno
