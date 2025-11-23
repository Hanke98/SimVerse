#include "BatchRigidBodySystem.h"
#include "BasicShapes/PlaneModel.h"
#include "GLSurfaceVisualModule.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"

namespace dyno
{
    template<typename TDataType>
    BatchRigidBodySystem<TDataType>::BatchRigidBodySystem()
      : ArticulatedBody<TDataType>()
    {
        RigidBodySystem<TDataType>::init();
    }

    template<typename TDataType>
    BatchRigidBodySystem<TDataType>::~BatchRigidBodySystem()
    {
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::addExampleRigidBodies(
      std::string urdf_fn, Vec3f base, Vec3f offset, int num_copies_x, int num_copies_y, int num_copies_z)
    {
        // TODO: load urdf and create rigid bodies accordingly

        {
            auto addRigidArm = [&](Vec3f _offset) {
                MulitBodyChainIndices mb;

                auto* rigid = this;

                float scale = 0.1;
                BoxInfo box3;
                box3.halfLength = Vec3f(1, 3, 1) * scale;
                RigidBodyInfo boxInfo3;
                boxInfo3.position = Vec3f(7.0, 50.0, 2.0) * scale + _offset;
                boxInfo3.friction = 0.0;
                boxInfo3.collisionMask = CT_Disabled;
                boxInfo3.motionType = Static;
                auto boxAt3 = rigid->addBox(box3, boxInfo3, 100.0);
                mb.body_indices.push_back(boxAt3->idx);

                auto func = [&](std::shared_ptr<PdActor> lastAct) {
                    Vec3f last = lastAct->center;

                    BoxInfo box1_;
                    box1_.halfLength = Vec3f(4, 1, 1) * scale;
                    RigidBodyInfo boxInfo1_;

                    Vec3f offset_1 = Vec3f(5.0, 2.0, 0.0) * scale;
                    boxInfo1_.position = last + offset_1;
                    boxInfo1_.friction = 0.0;
                    boxInfo1_.collisionMask = CT_Disabled;
                    auto boxAt1_ = rigid->addBox(box1_, boxInfo1_, 100.0);
                    mb.body_indices.push_back(boxAt1_->idx);

                    BoxInfo box2_;
                    box2_.halfLength = Vec3f(1, 3, 1) * scale;
                    RigidBodyInfo boxInfo2_;

                    Vec3f offset_2 = Vec3f(8.0, 4.0, 2.0) * scale;
                    boxInfo2_.position = last + offset_2;
                    boxInfo2_.friction = 0.0;
                    boxInfo2_.collisionMask = CT_Disabled;
                    auto boxAt2_ = rigid->addBox(box2_, boxInfo2_, 100.0);
                    mb.body_indices.push_back(boxAt2_->idx);

                    Vec3f offset_3 = Vec3f(1.0, 2.0, 0.0) * scale;
                    auto& joint_5 = rigid->createHingeJoint(lastAct, boxAt1_);
                    joint_5.setAnchorPoint(last + offset_3);
                    joint_5.setAxis(Vec3f(1.0f, 0.0f, 0.0f));
                    mb.hinge_joint_indices.push_back(rigid->getHostHingeJoints().size() - 1);

                    Vec3f offset_4 = Vec3f(8.0, 2.0, 1.0) * scale;
                    auto& joint_6 = rigid->createHingeJoint(boxAt1_, boxAt2_);
                    joint_6.setAnchorPoint(last + offset_4);
                    joint_6.setAxis(Vec3f(0.0f, 0.0f, 1.0f));
                    joint_6.setRange(-3.14f / 2.0f, 3.14f / 2.0f);
                    mb.hinge_joint_indices.push_back(rigid->getHostHingeJoints().size() - 1);
                    return boxAt2_;
                };

                auto last_act = func(boxAt3);
                auto last_act2 = func(last_act);
                auto last_act3 = func(last_act2);
                auto last_act4 = func(last_act3);
                auto last_act5 = func(last_act4);

                auto func2 = [&](std::shared_ptr<PdActor> lastAct) {
                    Vec3f last = lastAct->center;

                    BoxInfo box1_;
                    box1_.halfLength = Vec3f(4, 1, 1) * scale;
                    RigidBodyInfo boxInfo1_;

                    Vec3f offset_1 = Vec3f(5.0, 2.0, 0.0) * scale;
                    boxInfo1_.position = last + offset_1;
                    boxInfo1_.friction = 0.0;
                    boxInfo1_.collisionMask = CT_Disabled;
                    auto boxAt1_ = rigid->addBox(box1_, boxInfo1_, 100.0);
                    mb.body_indices.push_back(boxAt1_->idx);

                    Vec3f offset_3 = Vec3f(1.0, 2.0, 0.0) * scale;
                    auto& joint_5 = rigid->createHingeJoint(lastAct, boxAt1_);
                    joint_5.setAnchorPoint(last + offset_3);
                    joint_5.setAxis(Vec3f(1.0f, 0.0f, 0.0f));
                    mb.hinge_joint_indices.push_back(rigid->getHostHingeJoints().size() - 1);

                    return boxAt1_;
                };

                return mb;
            };

            auto addRigidArmExample2 = [&](Vec3f _offset) {
                MulitBodyChainIndices mb;

                auto* rigid = this;

                float scale = 0.1;
                BoxInfo box1;
                box1.halfLength = Vec3f(4, 1, 1) * scale;
                RigidBodyInfo boxInfo1;
                boxInfo1.position = Vec3f(4.0, 50.0, 0.0) * scale + _offset;
                boxInfo1.friction = 0.0;
                boxInfo1.motionType = Static;
                boxInfo1.collisionMask = CT_Disabled;
                auto boxAt1 = rigid->addBox(box1, boxInfo1, 100.0);
                mb.body_indices.push_back(boxAt1->idx);

                auto func = [&](std::shared_ptr<PdActor> lastAct) {
                    Vec3f last = lastAct->center;
                    BoxInfo box2_;
                    box2_.halfLength = Vec3f(4, 1, 1) * scale;
                    RigidBodyInfo boxInfo2_;

                    Vec3f offset_2 = Vec3f(7.0, 0.0, 2.0) * scale;
                    boxInfo2_.position = last + offset_2 + _offset;
                    boxInfo2_.friction = 0.0;
                    boxInfo2_.collisionMask = CT_Disabled;
                    auto boxAt2_ = rigid->addBox(box2_, boxInfo2_, 100.0);
                    mb.body_indices.push_back(boxAt2_->idx);

                    Vec3f offset_4 = Vec3f(4.0, 0.0, 1.0) * scale;
                    auto& joint_6 = rigid->createHingeJoint(lastAct, boxAt2_);
                    joint_6.setAnchorPoint(last + offset_4 + _offset);
                    joint_6.setAxis(Vec3f(0.0f, 0.0f, 1.0f));
                    joint_6.setRange(-3.14f / 2.0f, 3.14f / 2.0f);
                    mb.hinge_joint_indices.push_back(rigid->getHostHingeJoints().size() - 1);
                    return boxAt2_;
                };
                auto last_act = func(boxAt1);
                auto last_act2 = func(last_act);
                auto last_act3 = func(last_act2);
                auto last_act4 = func(last_act3);
                auto last_act5 = func(last_act4);

                return mb;
            };

            int robotarmIndex = 0;
            auto addRigidArmExample3 = [&](Vec3f _offset) {
                MulitBodyChainIndices mb;

                std::string filename = getAssetPath() + "../asset/franka_description/robots/franka_panda.urdf";
                // std::string filename = getAssetPath() + "../asset/kuka_allegro_description/kuka.urdf";
                // std::string filename = getAssetPath() + "../asset/kuka_allegro_description/kuka_allegro_touch_sensor.urdf";

		        if (this->varFilePath()->getValue() != filename)
		        {
			        this->varFilePath()->setValue(FilePath(filename));
		        } else {
                    std::cout << "Robot: Skip loading file" << std::endl;
                }

                RigidBodyInfo rigidbody;
                rigidbody.bodyId = robotarmIndex;

                auto texMesh = this->stateTextureMesh()->constDataPtr();
                std::map<int, std::shared_ptr<PdActor>> actors;
                std::unordered_map<std::string, int> linkNameToActorIndex;

                std::unordered_map<std::string, std::shared_ptr<PdActor>> linkNameToActor;

                for (int l = 0; l < this->urdfInfo.links.size(); ++l) {

                    auto it = this->urdfInfo.links[l].shapeId;

                    auto up = texMesh->shapes()[it]->boundingBox.v1;
                    auto down = texMesh->shapes()[it]->boundingBox.v0;

                    rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation() + _offset;
                    // rigidbody.angle = Quat1f(instances[robotarmIndex].rotation());

                    initialPositions.push_back(rigidbody.position);
                    initialQuats.push_back(rigidbody.angle);
                    initialRotations.push_back(rigidbody.angle.toMatrix3x3());

                    if (this->urdfInfo.links[l].isRoot) {
                        rigidbody.motionType = BodyType::Static;
                    } else {
                        rigidbody.motionType = BodyType::Dynamic;
                    }

                    auto actor = this->createRigidBody(rigidbody);
                    actors[it] = actor;

                    BoxInfo box;

                    box.halfLength = (up - down) / 2;

                    this->bindBox(actor, box);

                    this->bindShape(actor, Pair<uint, uint>(it, robotarmIndex));

                    mb.body_indices.push_back(actor->idx);
                }

                for (int j = 0; j < this->urdfInfo.joints.size(); ++j) {
                    auto parentName = this->urdfInfo.joints[j].parentLink;
                    auto childName = this->urdfInfo.joints[j].childLink;

                    auto parentId = this->urdfInfo.joints[j].parentLinkId;
                    auto childId = this->urdfInfo.joints[j].childLinkId;

                    if (this->urdfInfo.joints[j].type == REVOLUTE) {
                        auto &joint = this->createHingeJoint(actors[this->urdfInfo.links[parentId].shapeId], actors[this->urdfInfo.links[childId].shapeId]);
                        joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + _offset);
                        joint.setAxis(this->urdfInfo.joints[j].originWorld.rotation() * this->urdfInfo.joints[j].axis);
                        joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        mb.hinge_joint_indices.push_back(this->getHostHingeJoints().size() - 1);
                    }
                    if (this->urdfInfo.joints[j].type == PRISMATIC) {
                        auto &joint = this->createSliderJoint(actors[this->urdfInfo.links[parentId].shapeId], actors[this->urdfInfo.links[childId].shapeId]);
                        joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + _offset);
                        joint.setAxis(this->urdfInfo.joints[j].originWorld.rotation() * this->urdfInfo.joints[j].axis);
                        joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        mb.slider_joint_indices.push_back(this->getHostSliderJoints().size() - 1);
                    }
                    if (this->urdfInfo.joints[j].type == FIXED) {
                        auto &joint = this->createFixedJoint(actors[this->urdfInfo.links[parentId].shapeId], actors[this->urdfInfo.links[childId].shapeId]);
                        joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + _offset);
                        mb.fixed_joint_indices.push_back(this->getHostFixedJoints().size() - 1);
                    }
                }


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

            for (int x = 0; x < num_copies_x; x++)
            {
                for (int y = 0; y < num_copies_y; y++)
                {
                    for (int z = 0; z < num_copies_z; z++)
                    {
                        Vec3f offset = base + Vec3f(x * 2.0f, y * 2.0f, z * 2.0f);
                        // auto mb = addRigidArm(offset);
                        auto mb = addRigidArmExample3(offset);
                        ctrl_mb_chains.push_back(mb);
                        robotarmIndex++;
                    }
                }
            }
            attachRender();

            for (int i = 0; i < ctrl_mb_chains.size(); i++)
            {
              auto mb = ctrl_mb_chains[i];
              printf("Multi-body chain %d:\n", i);
              printf("  Body indices: ");
              for (int j = 0; j < mb.body_indices.size(); j++)
              {
                printf("%d ", mb.body_indices[j]);
              }
              printf("\n");

              printf("  Hinge joint indices: ");
              for (int j = 0; j < mb.hinge_joint_indices.size(); j++)
              {
                printf("%d ", mb.hinge_joint_indices[j]);
              }
              printf("\n");
            }
        }
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::resetBatchMultiBodies(BatchRigidBodySystemControlParamBase& param)
    {


        Array<Vec3f, DeviceType::CPU> hCenters = gethCenters();
        Array<TQuat, CPU> hAngels = gethAngels();
        Array<Vec3f, DeviceType::CPU> hVelocities = gethVelocities();
        Array<Vec3f, CPU> hAngularVelocities = gethAngularVelocities();
        Array<Mat3f, CPU> hRotations = gethRotationMatrix();

        // hCenters.assign(*this->stateCenter()->getDataPtr());
        // hAngularVelocities.assign(*this->stateAngularVelocity()->getDataPtr());
        // hAngels.assign(*this->stateQuaternion()->getDataPtr());
        // hVelocities.assign(*this->stateVelocity()->getDataPtr());
        // hRotations.assign(*this->stateRotationMatrix()->getDataPtr());

        for (auto it : param.ids) {
            for (auto index : ctrl_mb_chains[it].body_indices) {
                hCenters[index] = initialPositions[index];
                hAngels[index] = initialQuats[index];
                hRotations[index] = initialRotations[index];
                hVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
                hAngularVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
            }
        }

        this->stateCenter()->assign(hCenters);
        this->stateQuaternion()->assign(hAngels);
        this->stateRotationMatrix()->assign(hRotations);
        this->stateVelocity()->assign(hVelocities);
        this->stateAngularVelocity()->assign(hAngularVelocities);
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::resetOneMultiBodies(int mb_id)
    {

    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::applyTorqueControl(BatchRigidBodySystemTorqueControlParam& torque_param)
    {
        int rigidbodys = this->stateExternalForce()->size();
        std::vector<Vec3f> systemForces(rigidbodys, Vec3f(0.0f, 0.0f, 0.0f));

        const int n = torque_param.num_bodies;
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::applyHingeTorqueControl(BatchRigidBodySystemHingeTorqueControlParam& torque_param) {

        // Array<Vec3f, DeviceType::CPU> systemTorque;
        // systemTorque.assign(*this->stateExternalTorque()->getDataPtr());

        int rigidbodys = this->stateExternalTorque()->size();
        std::vector<Vec3f> systemTorque(rigidbodys, Vec3f(0.0f, 0.0f, 0.0f));

        for (int i : torque_param.ids) {
            auto& mb_chain = ctrl_mb_chains[i];
            for (int j = 0; j < mb_chain.hinge_joint_indices.size(); ++j) {
                auto& joint = this->urdfInfo.joints[j];
                auto parentId = joint.parentLinkId;
                auto childId = joint.childLinkId;
                auto jointAxis = joint.originWorld.rotation() * joint.axis;
                auto hingeTorque = torque_param.torques[i][j] * jointAxis;

                systemTorque[mb_chain.body_indices[parentId]] -= hingeTorque;
                systemTorque[mb_chain.body_indices[childId]] += hingeTorque;
            }
        }
        this->stateExternalTorque()->assign(systemTorque);
    }

    template<typename TDataType>
    Array<Vec3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethCenters()
    {
        Array<Vec3f, DeviceType::CPU> hCenters;
        hCenters.assign(*this->stateCenter()->getDataPtr());
        return hCenters;
    }

    template<typename TDataType>
    Array<typename BatchRigidBodySystem<TDataType>::TQuat, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethAngels()
    {
        Array<TQuat, DeviceType::CPU> hAngles;
        hAngles.assign(*this->stateQuaternion()->getDataPtr());
        return hAngles;
    }

    template<typename TDataType>
    Array<Vec3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethVelocities()
    {
        Array<Vec3f, DeviceType::CPU> hVelocities;
        hVelocities.assign(*this->stateVelocity()->getDataPtr());
        return hVelocities;
    }

    template<typename TDataType>
    Array<Vec3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethAngularVelocities()
    {
        Array<Vec3f, DeviceType::CPU> hAngularVelocities;
        hAngularVelocities.assign(*this->stateAngularVelocity()->getDataPtr());
        return hAngularVelocities;
    }

    template<typename TDataType>
    Array<Mat3f, CPU> BatchRigidBodySystem<TDataType>::gethRotationMatrix()
    {
        Array<Mat3f, CPU> hRotationMatrix;
        hRotationMatrix.assign(*this->stateRotationMatrix()->getDataPtr());
        return hRotationMatrix;
    }

    DEFINE_CLASS(BatchRigidBodySystem);

} // namespace dyno
