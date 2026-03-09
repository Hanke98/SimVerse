#pragma once
#include "Collision/NeighborMeshQuery.h"
#include "Node.h"
#include "RigidBody/ArticulatedBody.h"
#include "RigidBody/MultibodySystem.h"
#include "RigidBodyShared.h"
#include "RigidBodySystem.h"

#include "Topology/TriangleSet.h"

#include "Collision/Attribute.h"
#include "Collision/CollisionData.h"

#include <iostream>
#include <memory>
#include <vector>
namespace dyno
{
  template<typename TDataType> class NeighborTriMeshQuery;
  template<typename TDataType> class TJConstraintSolver;

  template<typename TDataType>
  class BatchRigidBodySystem : virtual public ArticulatedBody<TDataType> {
public:
    typedef typename TDataType::Real Real; 
    typedef typename TDataType::Coord Coord;
    typedef typename TDataType::Matrix Matrix;
    typedef typename dyno::Quat<Real> TQuat;
    typedef typename ::dyno::HingeJoint<Real> HingeJoint;

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
      std::vector<std::vector<Real>> torques;
    };

    struct BatchRigidBodySystemHingeVelocityControlParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<Real>> motorVel;
    };

    struct BatchRigidBodySystemMassParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<Real>> mass;
    };

    struct BatchRigidBodySystemInertiaParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<Matrix>> inertia;
    };

    struct BatchRigidBodySystemLocalIndexParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<int> localRigidBodyid;
    };

    struct BatchRigidBodySystemHingeInitGestureParam: public BatchRigidBodySystemControlParamBase
    {
      std::vector<std::vector<Real>> theta;
    };

    enum CollisionDetectionType
    {
        Element,
        TriMesh
    };

    BatchRigidBodySystem();
    ~BatchRigidBodySystem() override;

		DEF_VAR(FilePath, UrdfFilePath, "", "");

    // Expose potential contact triangles on the Node so GraphicsPipeline can build a valid
    // dependency graph (Pipeline reconstruct starts from Node fields).
    DEF_INSTANCE_STATE(TriangleSet<TDataType>, PotentialTriSet, "");

    // ------------------------------------
    // Control APIs
	  void createBatchMultiBodies(Coord base, Coord offset, int num_x, int num_y, int num_z);

    void resetBatchMultiBodies(BatchRigidBodySystemResetParam& reset_param);

    void resetBatchNonCtrlBodies(BatchRigidBodySystemResetParam& reset_param);
  
    void applyTorqueControl(BatchRigidBodySystemTorqueControlParam& torque_param);

    void applyHingeTorqueControl(BatchRigidBodySystemHingeTorqueControlParam& torque_param);

    void applyHingeVelocityControl(BatchRigidBodySystemHingeVelocityControlParam& motor_param);

    void setMass(BatchRigidBodySystemMassParam& mass_param);

    void setInertia(BatchRigidBodySystemInertiaParam& inertia_param);

    void setInitGesture(BatchRigidBodySystemHingeInitGestureParam& hinge_param);

    void addExampleRigidBodies(std::string urdf_fn, Vec3f base, Vec3f offset, Real density,
                               int num_copies_x, int num_copies_y, int num_copies_z);
    void addRobotArmRigidBodies(std::string urdf_fn, Real density, std::vector<Vec3f> targetPosition,
      bool renderBoundingBox, bool visual_or_collision/* visual == 0, collision == 1*/);

    void loadUrdf(std::string urdf_fn);

    void setupNeighborTriMeshQueryFromUrdf();
    void setupNeighborMeshQueryFromUrdf();

    void pushBackCtrlMBChain(MulitBodyChainIndices mb_chain) {
      ctrl_mb_chains.push_back(mb_chain);
    }

    void pushBackShape2ElementIds(Pair<uint, uint> entry) {
      mTextureMeshShape2ElementIds.push_back(entry);
    }

    std::vector<Pair<uint, uint>>* getTextureMeshShape2ElementIds() {
      return &mTextureMeshShape2ElementIds;
    }

    void pushBackShape2ElementIdsDense(int ElementId) {
      mTextureMeshShape2ElementIdsDense.push_back(ElementId);
    }

    std::vector<Pair<uint, uint>> mTextureMeshShape2ElementIds;
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

    void setVelocitySolverIterations(uint iterations);
    void setDisableContactReduction(bool disable);

    Array<Vec3f, DeviceType::CPU> gethCenters();
    Array<TQuat, DeviceType::CPU> gethAngles();
    Array<Vec3f, DeviceType::CPU> gethVelocities();
    Array<Vec3f, DeviceType::CPU> gethAngularVelocities();
    Array<Mat3f, DeviceType::CPU> gethRotationMatrix();

    Array<Real, DeviceType::CPU> gethMass();
    Array<Mat3f, DeviceType::CPU> gethInertia();

    std::vector<TQuat> getAnglesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getAngularVelocitiesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getVelocitiesByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Vec3f> getCentersByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    std::vector<Real> getMassByLocalIndex(BatchRigidBodySystemLocalIndexParam& param);
    // ------------------------------------

public:

    DEF_VAR(Bool, VisualOrCollision, false, "False stands for Visual while ture standing for collision");
    
    DEF_VAR(CollisionDetectionType, CollisionDetectionType, TriMesh, "CollisionDetectionType");

    DEF_VAR(Bool, EnableVisualizeCollisionTriSet, true, "Enable visualize collision triSet mesh");
    
protected:
    void resetStates() override;

    void resetOneMultiBodies(int mb_id);

	  void createOneMultiBody(Coord base, Coord offset);

    void applyOneMultiBodyTorqueControl(int mb_id, Coord torque);

    

protected:
    void initCollisionPipeline();
    

    std::vector<MulitBodyChainIndices> ctrl_mb_chains; // main multi-body chains with control
    std::vector<MulitBodyChainIndices> non_ctrl_mb_chains; // other multi-body chains in the env.

    std::vector<Vec3f> initialPositions; // initial position of all rigid bodies
    std::vector<TQuat> initialQuats; // initial rotation quaternion of all rigid bodies
    std::vector<Matrix> initialRotations; // initial rotation matrix of all rigid bodies

    // std::vector<Pair<uint, uint>> mTextureMeshShape2ElementIds;
    std::vector<int> mTextureMeshShape2ElementIdsDense;
    std::vector<int> mTextureMeshShape2RigidBodyIds; 

    

    std::shared_ptr<NeighborTriMeshQuery<TDataType>> mNeighborTriMeshQuery;
    std::shared_ptr<TJConstraintSolver<TDataType>> mConstraintSolver;
    // Collision TriangleSet used by NeighborTriMeshQuery (kept in rest-world space; do NOT call update()).
    std::shared_ptr<TriangleSet<TDataType>> mCollisionTriangleSet;
  };

} // namespace dyno
