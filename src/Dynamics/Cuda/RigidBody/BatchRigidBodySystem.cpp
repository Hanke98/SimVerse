#include "BatchRigidBodySystem.h"
#include "BasicShapes/PlaneModel.h"
#include "GLSurfaceVisualModule.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"

namespace dyno {
  template<typename TDataType>
  BatchRigidBodySystem<TDataType>::BatchRigidBodySystem()
    : RigidBodySystem<TDataType>()
  // : ArticulatedBody<TDataType>()
  {
  }

  template<typename TDataType>
  BatchRigidBodySystem<TDataType>::~BatchRigidBodySystem()
  {
  }

  template<typename TDataType>
  void BatchRigidBodySystem<TDataType>::addRigidBodies(std::string urdf_fn, int num_copies)
  {
    // TODO: load urdf and create rigid bodies accordingly

    {
      auto addRigidArm = [&](Vec3f _offset) {
        MulitBodyChainIndices mb;

        auto* rigid = this;

        BoxInfo box1;
        box1.halfLength = Vec3f(1, 1, 1);

        RigidBodyInfo boxInfo1;
        boxInfo1.position = Vec3f(-1.0, 10.0, 0.0) + _offset;
        boxInfo1.friction = 0.0;
        boxInfo1.collisionMask = CT_Disabled;
        boxInfo1.motionType = Static;
        auto boxAt1 = rigid->addBox(box1, boxInfo1);
        mb.body_indices.push_back(boxAt1->idx);

        BoxInfo box2;
        box2.halfLength = Vec3f(4, 1, 1);

        RigidBodyInfo boxInfo2;

        boxInfo2.position = Vec3f(4.0, 10.0, 0.0) + _offset;
        boxInfo2.friction = 0.0;
        boxInfo2.collisionMask = CT_Disabled;
        auto boxAt2 = rigid->addBox(box2, boxInfo2, 100.0);
        mb.body_indices.push_back(boxAt2->idx);

        BoxInfo box3;
        box3.halfLength = Vec3f(1, 3, 1);

        RigidBodyInfo boxInfo3;
        boxInfo3.position = Vec3f(7.0, 12.0, 2.0) + _offset;
        boxInfo3.friction = 0.0;
        boxInfo3.collisionMask = CT_Disabled;
        auto boxAt3 = rigid->addBox(box3, boxInfo3, 100.0);
        mb.body_indices.push_back(boxAt3->idx);

        auto& joint1 = rigid->createHingeJoint(boxAt1, boxAt2);
        auto& hingeJoints = rigid->getHostHingeJoints();
        auto joint_idx = hingeJoints.size() - 1;
        mb.hinge_joint_indices.push_back(joint_idx);

        joint1.setAnchorPoint(Vec3f(0.0, 10, 0.0) + _offset);
        joint1.setAxis(Vec3f(1.0f, 0.0f, 0.0f));

        auto& joint2 = rigid->createHingeJoint(boxAt2, boxAt3);
        joint2.setAnchorPoint(Vec3f(7.0, 10.0, 1.0) + _offset);
        joint2.setAxis(Vec3f(0.0f, 0.0f, 1.0f));
        mb.hinge_joint_indices.push_back(hingeJoints.size() - 1);
        return mb;
      };

      auto attachRender = [&]() {
        auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
        auto rigid = this;
        rigid->stateTopology()->connect(mapper->inDiscreteElements());
        rigid->graphicsPipeline()->pushModule(mapper);

        auto sRender = std::make_shared<GLSurfaceVisualModule>();
        sRender->setColor(Color(1, 1, 0));
        sRender->setAlpha(0.5f);
        mapper->outTriangleSet()->connect(sRender->inTriangleSet());
        rigid->graphicsPipeline()->pushModule(sRender);
      };

      for (int i = 0; i < num_copies; i++)
      {
        Vec3f offset = Vec3f(i * 15.0f, 0.0f, 0.0f);
        auto mb = addRigidArm(offset);
        multi_body_chains.push_back(mb);
      }
      attachRender();

      // for (int i = 0; i < multi_body_chains.size(); i++)
      // {
      //   auto mb = multi_body_chains[i];
      //   printf("Multi-body chain %d:\n", i);
      //   printf("  Body indices: ");
      //   for (int j = 0; j < mb.body_indices.size(); j++)
      //   {
      //     printf("%d ", mb.body_indices[j]);
      //   }
      //   printf("\n");
      //
      //   printf("  Hinge joint indices: ");
      //   for (int j = 0; j < mb.hinge_joint_indices.size(); j++)
      //   {
      //     printf("%d ", mb.hinge_joint_indices[j]);
      //   }
      //   printf("\n");
      // }
    }
  }

  template<typename TDataType>
  void BatchRigidBodySystem<TDataType>::reset(BatchRigidBodySystemControlParam& param)
  {
  }

  DEFINE_CLASS(BatchRigidBodySystem);

} // namespace dyno
