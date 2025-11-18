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
namespace dyno {

  template<typename TDataType>
  // class BatchRigidBodySystem : virtual public ArticulatedBody<TDataType> {
  class BatchRigidBodySystem : virtual public RigidBodySystem<TDataType> {
public:
    struct BatchRigidBodySystemControlParam
    {
      std::vector<int> reset_mb_ids;
    };

    struct MulitBodyChainIndices
    {
      std::vector<int> body_indices;
      std::vector<int> hinge_joint_indices;
    };

    BatchRigidBodySystem();
    ~BatchRigidBodySystem() override;

    void addRigidBodies(std::string urdf_fn, Vec3f base, Vec3f offset, int num_copies_x, int num_copies_y, int num_copies_z);

    void reset(BatchRigidBodySystemControlParam& param);

protected:
    std::vector<MulitBodyChainIndices> multi_body_chains;
  };

} // namespace dyno
