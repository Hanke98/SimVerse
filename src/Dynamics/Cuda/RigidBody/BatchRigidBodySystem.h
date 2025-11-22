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
      int num_bodies;
      std::vector<int> ids;
    };

    struct BatchRigidBodySystemTorqueControlParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<Coord> torques;
    };

    BatchRigidBodySystem();
    ~BatchRigidBodySystem() override;

		DEF_VAR(FilePath, UrdfFilePath, "", "");

    // ------------------------------------
    // Control APIs
	  void createBatchMultiBodies(Coord base, Coord offset, int num_x, int num_y, int num_z);

    void resetBatchMultiBodies(BatchRigidBodySystemControlParamBase& param);
  
    void applyTorqueControl(BatchRigidBodySystemTorqueControlParam& torque_param);

    void addExampleRigidBodies(std::string urdf_fn, Vec3f base, Vec3f offset, int num_copies_x, int num_copies_y, int num_copies_z);
    // ------------------------------------

    // ------------------------------------
    // Setters and Getters
    void setDt(Real dt)
    {
    }

    void setGravityEnabled(bool enabled)
    {
      this->varGravityEnabled()->setValue(enabled);
    }
    // ------------------------------------

protected:

    void resetOneMultiBodies(int mb_id);

	  void createOneMultiBody(Coord base, Coord offset);

    void applyOneMultiBodyTorqueControl(int mb_id, Coord torque);

protected:
    std::vector<MulitBodyChainIndices> ctrl_mb_chains; // main multi-body chains with control
    std::vector<MulitBodyChainIndices> non_ctrl_mb_chains; // other multi-body chains in the env.
  };

} // namespace dyno
