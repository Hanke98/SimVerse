#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>
// #include <RigidBody/MultibodySystem.h>

#include <GLPointVisualModule.h>
#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>

#include "BasicShapes/PlaneModel.h"
#include "Collision/NeighborElementQuery.h"
#include "RigidBody/BatchRigidBodySystem.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatBricks()
{
  std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

  scn->setGravity(Vec3f(0, -9.8, 0));

  auto createRrigid = [&](int idx) {
    std::string name = "mb_" + std::to_string(idx);
    auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>(name));
    return rigid;
  };

  auto addRigidArm = [&](std::shared_ptr<RigidBodySystem<DataType3f>> rigid, Vec3f _offset) {
    rigid->setDt(1 / 100.0f);

    rigid->varGravityEnabled()->setValue(true);
    rigid->varFrictionEnabled()->setValue(false);

    BoxInfo box1;
    box1.halfLength = Vec3f(1, 1, 1);
    RigidBodyInfo boxInfo1;
    boxInfo1.position = Vec3f(-1.0, 10.0, 0.0) + _offset;
    boxInfo1.friction = 0.0;
    boxInfo1.collisionMask = CT_Disabled;
    boxInfo1.motionType = Static;
    auto boxAt1 = rigid->addBox(box1, boxInfo1);

    BoxInfo box2;
    box2.halfLength = Vec3f(4, 1, 1);
    RigidBodyInfo boxInfo2;
    boxInfo2.position = Vec3f(4.0, 10.0, 0.0) + _offset;
    boxInfo2.friction = 0.0;
    boxInfo2.collisionMask = CT_Disabled;
    auto boxAt2 = rigid->addBox(box2, boxInfo2, 100.0);

    BoxInfo box3;
    box3.halfLength = Vec3f(1, 3, 1);
    RigidBodyInfo boxInfo3;
    boxInfo3.position = Vec3f(7.0, 12.0, 2.0) + _offset;
    boxInfo3.friction = 0.0;
    boxInfo3.collisionMask = CT_Disabled;
    auto boxAt3 = rigid->addBox(box3, boxInfo3, 100.0);

    auto& joint1 = rigid->createHingeJoint(boxAt1, boxAt2);
    joint1.setAnchorPoint(Vec3f(0.0, 10, 0.0) + _offset);
    joint1.setAxis(Vec3f(1.0f, 0.0f, 0.0f));

    auto& joint2 = rigid->createHingeJoint(boxAt2, boxAt3);
    joint2.setAnchorPoint(Vec3f(7.0, 10.0, 1.0) + _offset);
    joint2.setAxis(Vec3f(0.0f, 0.0f, 1.0f));
    return rigid;
  };

  auto attachRender = [&](std::shared_ptr<RigidBodySystem<DataType3f>> rigid) {
    // for rendering
    auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
    rigid->stateTopology()->connect(mapper->inDiscreteElements());
    rigid->graphicsPipeline()->pushModule(mapper);

    auto sRender = std::make_shared<GLSurfaceVisualModule>();
    sRender->setColor(Color(1, 1, 0));
    sRender->setAlpha(0.5f);
    mapper->outTriangleSet()->connect(sRender->inTriangleSet());
    rigid->graphicsPipeline()->pushModule(sRender);
  };

  Vec3f offset(0, 0, 20);
  Vec3f base(0.0f, 0.0f, 0.0f);

  std::vector<std::shared_ptr<RigidBodySystem<DataType3f>>> rigids;
  int cnt = 5;
  for (int i = 0; i < cnt; i++)
  {
    auto rigid = createRrigid(i);
    addRigidArm(rigid, base + offset * i);
    attachRender(rigid);
    rigids.push_back(rigid);
  }

  auto cnt2 = 5;
  auto _rigid = createRrigid(cnt);
  for (int i = cnt; i < cnt + cnt2; ++i)
  {
    addRigidArm(_rigid, base + offset * i);
  }
  attachRender(_rigid);

  return scn;
}

std::shared_ptr<SceneGraph> demoBatchRigidSystem()
{
  std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
  scn->setGravity(Vec3f(0, -9.8, 0));

  int num_copies = 10;
  auto batch_rigid = scn->addNode(std::make_shared<BatchRigidBodySystem<DataType3f>>());

  batch_rigid->setDt(1 / 100.0f);
  batch_rigid->varGravityEnabled()->setValue(true);
  batch_rigid->varFrictionEnabled()->setValue(false);
  // Vec3f base{ -20.0f, -0.0f, -20.0f };
  Vec3f base{ -0.0f, -0.0f, -0.0f };
  Vec3f offset{ 0.0f, 0.0f, 20.0f };
  // batch_rigid->addRigidBodies("", base, offset, 30, 30, 30);
  batch_rigid->addRigidBodies("", base, offset, 1, 1, 1);

  return scn;
}

int main()
{
  UbiApp app;
  app.setSceneGraph(demoBatchRigidSystem());
  // app.setSceneGraph(creatBricks());
  app.initialize(1280, 768);
  app.mainLoop();

  return 0;
}
