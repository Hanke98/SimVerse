#include "RigidBodySystem.h"

#include "Collision/CollistionDetectionBoundingBox.h"
#include "Collision/NeighborElementQuery.h"
#include "Primitive/Primitive3D.h"

#include "RigidBody/Module/CarDriver.h"
#include "RigidBody/Module/PCGConstraintSolver.h"
#include "RigidBody/Module/PJSConstraintSolver.h"
#include "RigidBody/Module/PJSNJSConstraintSolver.h"
#include "RigidBody/Module/PJSoftConstraintSolver.h"
#include "RigidBody/Module/TJConstraintSolver.h"
#include "RigidBody/Module/TJSoftConstraintSolver.h"

// Module headers
#include "RigidBody/Module/ContactsUnion.h"

namespace dyno
{
  typedef typename dyno::TOrientedBox3D<Real> Box3D;

  template<typename TDataType>
  RigidBodySystem<TDataType>::RigidBodySystem(std::string name)
    : Node()
  {
    this->setName(name);
    init();
  }

  template<typename TDataType>
  void RigidBodySystem<TDataType>::init()
  {
    auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
    this->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());
    //
    auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
    this->stateTopology()->connect(elementQuery->inDiscreteElements());
    this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
    this->stateAttribute()->connect(elementQuery->inAttribute());
    this->animationPipeline()->pushModule(elementQuery);

    auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
    this->stateTopology()->connect(cdBV->inDiscreteElements());
    this->animationPipeline()->pushModule(cdBV);

    auto merge = std::make_shared<ContactsUnion<TDataType>>();
    elementQuery->outContacts()->connect(merge->inContactsA());
    cdBV->outContacts()->connect(merge->inContactsB());
    this->animationPipeline()->pushModule(merge);

    auto iterSolver = std::make_shared<TJConstraintSolver<TDataType>>();
    // auto iterSolver = std::make_shared<TJSoftConstraintSolver<TDataType>>();
    this->stateTimeStep()->connect(iterSolver->inTimeStep());
    this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
    this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
    this->varGravityValue()->connect(iterSolver->varGravityValue());
    this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
    this->varSlop()->connect(iterSolver->varSlop());
    this->stateMass()->connect(iterSolver->inMass());

    this->stateExternalForce()->connect(iterSolver->inExternalForce());
    this->stateExternalTorque()->connect(iterSolver->inExternalTorque());
    this->varAngularDamping()->connect(iterSolver->varAngularDamping());

    this->stateFrictionCoefficients()->connect(iterSolver->inFrictionCoefficients());
    this->stateAttribute()->connect(iterSolver->inAttribute());
    this->stateCenter()->connect(iterSolver->inCenter());
    this->stateVelocity()->connect(iterSolver->inVelocity());
    this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
    this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
    this->stateInertia()->connect(iterSolver->inInertia());
    this->stateQuaternion()->connect(iterSolver->inQuaternion());
    this->stateInitialInertia()->connect(iterSolver->inInitialInertia());
    this->stateTopology()->connect(iterSolver->inDiscreteElements());
    merge->outContacts()->connect(iterSolver->inContacts());
    this->animationPipeline()->pushModule(iterSolver);

    this->setDt(0.016f);
  }

  DEFINE_CLASS(RigidBodySystem);
} // namespace dyno
