#include "BatchRigidBodySystem.h"
#include "BasicShapes/PlaneModel.h"
#include "GLSurfaceVisualModule.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "Mapping/DiscreteSpheresToTriangleSet.h"
// #include "UrdfFunc.h"

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
      std::string urdf_fn, Vec3f base, Vec3f offset, Real density, int num_copies_x, int num_copies_y, int num_copies_z)
    {
        // TODO: load urdf and create rigid bodies accordingly

        {
            auto addRigidArm = [&](Vec3f _offset) {
                MulitBodyChainIndices mb;

                auto* rigid = this;

                Real scale = 0.1;
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

                Real scale = 0.1;
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
            auto instances = this->varVehiclesTransform()->getValue();

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
                        Vec3f _offset = base + Vec3f(x * offset.x, y * offset.y, z * offset.z);
                        auto mb = addRigidArmExample2(_offset);
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
            for (int i = 0; i < non_ctrl_mb_chains.size(); i++) {
                auto mb = non_ctrl_mb_chains[i];
                printf("Non-control multi-body chain %d:\n", i);
                printf("  Body indices: ");
                for (int j = 0; j < mb.body_indices.size(); j++)
                {
                    printf("%d ", mb.body_indices[j]);
                }
                printf("\n");
            }
        }
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::loadUrdf(std::string urdf_fn) {
        std::string filename = getAssetPath() + urdf_fn;
        if (this->varFilePath()->getValue() != filename)
        {
            this->varFilePath()->setValue(FilePath(filename));
        } else {
            std::cout << "Robot: Skip loading file" << std::endl;
        }
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::addRobotArmRigidBodies(
        std::string urdf_fn, Real density, std::vector<Vec3f> targetPosition, bool renderBoundingBox, bool visual_or_collision) {
            int robotarmIndex = 0;
            auto instances = this->varVehiclesTransform()->getValue();
            auto robotarmSize = instances.size();

            std::string filename = getAssetPath() + urdf_fn;

            if (this->varFilePath()->getValue() != filename)
            {
                this->varFilePath()->setValue(FilePath(filename));
            } else {
                std::cout << "Robot: Skip loading file" << std::endl;
            }

            this->varVisualOrCollision()->setValue(visual_or_collision);

            for (int j = 0; j < this->urdfInfo.joints.size(); ++j) {
                auto parentId = this->urdfInfo.joints[j].parentLinkId;
                auto childId = this->urdfInfo.joints[j].childLinkId;

                Transform3f T_boundingbox_world;
                if (!this->varVisualOrCollision()->getValue()) {
                    T_boundingbox_world = this->urdfInfo.links[childId].T_visual_bb_world;
                } else {
                    T_boundingbox_world = this->urdfInfo.links[childId].T_collision_bb_world;
                }

                // 获取 Parent Joint 的世界旋转矩阵 (R_PJ)
                Mat3f R_PJ = this->urdfInfo.joints[j].originWorld.rotation();
                // 获取 Bounding Box 的世界旋转矩阵 (R_BB)
                Mat3f R_BB = T_boundingbox_world.rotation();
                // if (!visual_or_collision) {
                //     R_BB = this->urdfInfo.links[childId].T_visual_bb_world.rotation();
                // } else {
                //     R_BB = this->urdfInfo.links[childId].T_collision_bb_world.rotation();
                // }
                // 计算相对旋转 (R_PJ_to_BB = R_PJ_transpose * R_BB)
                Mat3f relativeRotation = R_PJ.transpose() * R_BB;

                // 获取世界坐标系下的相对平移向量 (t_BB - t_PJ)
                Vec3f worldDeltaTranslation = T_boundingbox_world.translation()
                                              - this->urdfInfo.joints[j].originWorld.translation();
                // if (!visual_or_collision) {
                //     worldDeltaTranslation = this->urdfInfo.links[childId].T_visual_bb_world.translation() - this->urdfInfo.joints[j].originWorld.translation();
                // } else {
                //     worldDeltaTranslation = this->urdfInfo.links[childId].T_collision_bb_world.translation() - this->urdfInfo.joints[j].originWorld.translation();
                // }

                // 获取 Parent Joint 的世界旋转矩阵转置 (R_PJ_transpose)
                Mat3f R_PJ_transpose = this->urdfInfo.joints[j].originWorld.rotation().transpose();
                // 计算相对平移
                Vec3f relativeTranslation = R_PJ_transpose * worldDeltaTranslation;
                if (!visual_or_collision) {
                    this->urdfInfo.links[childId].T_visual_bb_local.translation() = relativeTranslation;
                    this->urdfInfo.links[childId].T_visual_bb_local.rotation() = relativeRotation;
                }
                else {
                    this->urdfInfo.links[childId].T_collision_bb_local.translation() = relativeTranslation;
                    this->urdfInfo.links[childId].T_collision_bb_local.rotation() = relativeRotation;
                }

            }

            auto addRigidArmExample = [&](const Vec3f& _targetPosition)
            -> std::pair<MulitBodyChainIndices, MulitBodyChainIndices>
            {

                MulitBodyChainIndices mb;
                MulitBodyChainIndices non_ctrl_mb;

                RigidBodyInfo rigidbody;
                rigidbody.bodyId = robotarmIndex;

                auto texMesh = this->stateTextureMesh()->constDataPtr();
                std::map<int, std::shared_ptr<PdActor>> actors;
                std::unordered_map<std::string, int> linkNameToActorIndex;

                std::unordered_map<std::string, std::shared_ptr<PdActor>> linkNameToActor;

                for (int l = 0; l < this->urdfInfo.links.size(); ++l) {

                    uint it;

                    if (!visual_or_collision) {
                        it = this->urdfInfo.links[l].visualShapeId;
                    } else {
                        it = this->urdfInfo.links[l].collisionShapeId;
                    }

                    auto up = texMesh->shapes()[it]->boundingBox.v1;
                    auto down = texMesh->shapes()[it]->boundingBox.v0;

                    rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation() + instances[robotarmIndex].translation();

                    initialPositions.push_back(rigidbody.position);
                    initialQuats.push_back(rigidbody.angle);
                    initialRotations.push_back(rigidbody.angle.toMatrix3x3());

                    std::cout << "Position: " << rigidbody.position << std::endl;

                    if (this->urdfInfo.links[l].isRoot) {
                        rigidbody.motionType = BodyType::Static;
                    } else {
                        rigidbody.motionType = BodyType::Dynamic;
                    }

                    auto actor = this->createRigidBody(rigidbody);
                    actors[it] = actor;

                    BoxInfo box;
                    box.halfLength = (up - down) / 2;
                    if (this->urdfInfo.links[l].isRoot) {
                        this->bindBox(actor, box, 1000000000000);
                    } else {
                        if (!visual_or_collision) {
                            this->bindBox(actor, box, density);
                        } else {
                            this->bindBox(actor, box, this->urdfInfo.links[l].volume, this->urdfInfo.links[l].localInertia, density);
                        }
                    }
                    this->bindShape(actor, Pair<uint, uint>(it, robotarmIndex));
                    mb.body_indices.push_back(actor->idx);
                }

                for (int j = 0; j < this->urdfInfo.joints.size(); ++j) {
                    auto parentName = this->urdfInfo.joints[j].parentLink;
                    auto childName = this->urdfInfo.joints[j].childLink;

                    auto parentId = this->urdfInfo.joints[j].parentLinkId;
                    auto childId = this->urdfInfo.joints[j].childLinkId;

                    if (this->urdfInfo.joints[j].type == REVOLUTE) {
                        if (!visual_or_collision) {
                            auto& joint = this->createHingeJoint(actors[this->urdfInfo.links[parentId].visualShapeId], actors[this->urdfInfo.links[childId].visualShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                            joint.setAxis(this->urdfInfo.joints[j].axisWorld);
                            joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        } else {
                            auto& joint = this->createHingeJoint(actors[this->urdfInfo.links[parentId].collisionShapeId], actors[this->urdfInfo.links[childId].collisionShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                            joint.setAxis(this->urdfInfo.joints[j].axisWorld);
                            joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        }

                        mb.hinge_joint_indices.push_back(j);
                    }
                    if (this->urdfInfo.joints[j].type == PRISMATIC) {
                        if (!visual_or_collision) {
                            auto &joint = this->createSliderJoint(actors[this->urdfInfo.links[parentId].visualShapeId], actors[this->urdfInfo.links[childId].visualShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                            joint.setAxis(this->urdfInfo.joints[j].axisWorld);
                            joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        } else {
                            auto &joint = this->createSliderJoint(actors[this->urdfInfo.links[parentId].collisionShapeId], actors[this->urdfInfo.links[childId].collisionShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                            joint.setAxis(this->urdfInfo.joints[j].axisWorld);
                            joint.setRange(this->urdfInfo.joints[j].limits.lower, this->urdfInfo.joints[j].limits.upper);
                        }
                        mb.slider_joint_indices.push_back(j);
                    }
                    if (this->urdfInfo.joints[j].type == FIXED) {
                        if (!visual_or_collision) {
                            auto &joint = this->createFixedJoint(actors[this->urdfInfo.links[parentId].visualShapeId], actors[this->urdfInfo.links[childId].visualShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                        } else {
                            auto &joint = this->createFixedJoint(actors[this->urdfInfo.links[parentId].collisionShapeId], actors[this->urdfInfo.links[childId].collisionShapeId]);
                            joint.setAnchorPoint(this->urdfInfo.joints[j].originWorld.translation() + instances[robotarmIndex].translation());
                        }
                        mb.fixed_joint_indices.push_back(j);
                    }
                }

                // create target
                SphereInfo target;
                target.radius = 0.05f;

                RigidBodyInfo target_rb;
                target_rb.position = instances[robotarmIndex].translation() + _targetPosition;
                target_rb.motionType = BodyType::Static;
                target_rb.collisionMask = CT_Disabled;
                auto target_actor = this->addSphere(target, target_rb);

                initialPositions.push_back(target_rb.position);
                initialQuats.push_back(target_rb.angle);
                initialRotations.push_back(target_rb.angle.toMatrix3x3());
                non_ctrl_mb.body_indices.push_back(target_actor->idx);

                return {mb, non_ctrl_mb};
            };

            auto attachRender = [&]() {
                if (renderBoundingBox) {
                    auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
                    auto rigid = this;
                    rigid->stateTopology()->connect(mapper->inDiscreteElements());
                    rigid->graphicsPipeline()->pushModule(mapper);

                    auto sRender = std::make_shared<GLSurfaceVisualModule>();
                    sRender->setColor(Color(1, 1, 0));
                    sRender->setAlpha(0.5f);
                    mapper->outTriangleSet()->connect(sRender->inTriangleSet());
                    rigid->graphicsPipeline()->pushModule(sRender);
                } else {
                    auto mapper = std::make_shared<DiscreteSpheresToTriangleSet<DataType3f>>();
                    auto rigid = this;
                    rigid->stateTopology()->connect(mapper->inDiscreteElements());
                    rigid->graphicsPipeline()->pushModule(mapper);

                    auto sRender = std::make_shared<GLSurfaceVisualModule>();
                    sRender->setColor(Color(1, 1, 0));
                    sRender->setAlpha(0.5f);
                    mapper->outTriangleSet()->connect(sRender->inTriangleSet());
                    rigid->graphicsPipeline()->pushModule(sRender);
                }
            };

            for (size_t i = 0; i < robotarmSize; i++) {
                const Vec3f& _targetPosition = targetPosition[robotarmIndex];
                auto [mb, non_ctrl_mb]
                    = addRigidArmExample(_targetPosition);
                ctrl_mb_chains.push_back(mb);
                non_ctrl_mb_chains.push_back(non_ctrl_mb);
                robotarmIndex++;
            }
            attachRender();
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::resetBatchMultiBodies(BatchRigidBodySystemResetParam& reset_param)
    {
        Array<Vec3f, DeviceType::CPU> hCenters = gethCenters();
        Array<TQuat, CPU> hAngels = gethAngles();
        Array<Vec3f, DeviceType::CPU> hVelocities = gethVelocities();
        Array<Vec3f, CPU> hAngularVelocities = gethAngularVelocities();
        Array<Mat3f, CPU> hRotations = gethRotationMatrix();

        auto instances = this->varVehiclesTransform()->getValue();

        for (int i = 0; i < reset_param.num_bodies; i++) {
            auto it = reset_param.ids[i];
            for (auto index : ctrl_mb_chains[it].body_indices) {
                hCenters[index] = initialPositions[index];
                hAngels[index] = initialQuats[index];
                hRotations[index] = initialRotations[index];
                hVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
                hAngularVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
            }
            for (auto index : non_ctrl_mb_chains[it].body_indices) {
                hCenters[index] = reset_param.targetPosition[i] + instances[it].translation();
            }
        }

        this->stateCenter()->assign(hCenters);
        this->stateQuaternion()->assign(hAngels);
        this->stateRotationMatrix()->assign(hRotations);
        this->stateVelocity()->assign(hVelocities);
        this->stateAngularVelocity()->assign(hAngularVelocities);
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::setInitGesture(BatchRigidBodySystemHingeInitGestureParam& hinge_param) {
        this->initialGesture.clear();
        for (int i = 0; i < ctrl_mb_chains.size(); i++) {
            this->initialGesture.push_back(this->urdfInfo); // 复制零位姿态
        }

        std::vector<std::vector<int>> linkChildJoints(this->urdfInfo.links.size());
        int rootLinkIndex = -1;

        for (int i = 0; i < this->urdfInfo.joints.size(); ++i) {
            int pId = this->urdfInfo.joints[i].parentLinkId;
            linkChildJoints[pId].push_back(i);
        }

        // 寻找根节点 (isRoot 为 true 的 link)
        for (int i = 0; i < this->urdfInfo.links.size(); ++i) {
            if (this->urdfInfo.links[i].isRoot) {
                rootLinkIndex = i;
                break;
            }
        }

        // 对每个铰链关节进行处理
        for (size_t i = 0; i < hinge_param.num_bodies; i++) {
            for (auto it : hinge_param.ids) {
                if(it >= ctrl_mb_chains.size()) continue;
                auto& initialGesture = this->initialGesture[it];

                for (int j = 0; j < ctrl_mb_chains[it].hinge_joint_indices.size(); j++) {
                    auto jointIndex = ctrl_mb_chains[it].hinge_joint_indices[j];

                    // 引用原始数据 (R_original)
                    auto& originalJoint = this->urdfInfo.joints[jointIndex];
                    auto& originalLink = this->urdfInfo.links[originalJoint.childLinkId];
                    auto& originalParentLink = this->urdfInfo.links[originalJoint.parentLinkId];

                    // 引用正在修改的姿态数据 (R_current)
                    auto& currentJointGesture = initialGesture.joints[jointIndex];
                    auto& currentLinkGesture = initialGesture.links[originalJoint.childLinkId];

                    if (originalJoint.type == REVOLUTE) {
                        const auto& theta = hinge_param.theta[i][j];
                        // std::cout << "theta: " << i << ", " << j << ", " << theta << std::endl;
                        // 轴在 Parent Link Frame 下的表示（来自原始零位姿态）
                        // Vec3f axisParent = originalJoint.originLocal.rotation() * originalJoint.axis;
                        Vec3f axisParent = originalParentLink.T_world.rotation().inverse() * originalJoint.axisWorld;
                        // if (i == 0 && j == 1) {
                        //     std::cout << "axisParent: " << i << ", " << j << ", " << axisParent << std::endl;
                        // }
                        // std::cout << "axisParent: " << axisParent.x << ", " << axisParent.y << ", " << axisParent.z << std::endl;
                        axisParent.normalize();
                        TQuat thetaQuat(theta, axisParent);
                        Mat3f rotationMatrix = thetaQuat.toMatrix3x3();

                        currentJointGesture.originLocal.rotation() = rotationMatrix * originalJoint.originLocal.rotation();

                        currentLinkGesture.T_visual_bb_local.rotation() = originalLink.T_visual_bb_local.rotation();
                        currentLinkGesture.T_collision_bb_local.rotation() = originalLink.T_collision_bb_local.rotation();

                        currentLinkGesture.T_local.rotation() = rotationMatrix * originalLink.T_local.rotation();
                    }
                }
                // 定义递归 Lambda 函数
                std::function<void(int, const Transform3f&)> updateWorldRecursive =
                [&](int currentLinkIdx, const Transform3f& parentTWorld)
                {
                    UrdfLink& currentLink = initialGesture.links[currentLinkIdx];

                    currentLink.T_world = parentTWorld;
                    currentLink.T_visual_bb_world = composeTransform(currentLink.T_world, currentLink.T_visual_bb_local);
                    currentLink.T_collision_bb_world = composeTransform(currentLink.T_world, currentLink.T_collision_bb_local);
                    std::cout << "T_collision_bb_world: " << currentLink.T_collision_bb_world.translation() << std::endl;
                    std::cout << "T_world: " << currentLink.T_world.translation() << std::endl;
                    std::cout << "T_local: " << currentLink.T_collision_bb_local.translation() << std::endl;

                    for (int jointIdx : linkChildJoints[currentLinkIdx]) {
                        UrdfJoint& childJoint = initialGesture.joints[jointIdx];

                        // JointWorld = ParentLinkWorld * JointLocal
                        childJoint.originWorld = composeTransform(currentLink.T_world, childJoint.originLocal);

                        // 计算关节轴的世界方向 (轴由 Joint 的世界旋转旋转)
                        childJoint.axisWorld = childJoint.originWorld.rotation() * this->urdfInfo.joints[jointIdx].axis; // 轴本身不变，但世界方向会变

                        updateWorldRecursive(childJoint.childLinkId, childJoint.originWorld);
                    }
                };

                if (rootLinkIndex != -1) {
                    auto& initialRootGesture = initialGesture.links[rootLinkIndex];
                    Transform3f* initialRootBBGestureWorld = nullptr;
                    Transform3f* initialRootBBGestureLocal = nullptr;
                    if (!varVisualOrCollision()->getValue()) {
                        initialRootBBGestureWorld = &initialRootGesture.T_visual_bb_world;
                        initialRootBBGestureLocal = &initialRootGesture.T_visual_bb_local;
                    } else {
                        initialRootBBGestureWorld = &initialRootGesture.T_collision_bb_world;
                        initialRootBBGestureLocal = &initialRootGesture.T_collision_bb_local;
                    }

                    Transform3f rootWorldTransform = initialRootGesture.T_world;
                    Transform3f rootLocalTransform;
                    // Root Link 的 T_bounding_box_local 平移计算
                    Vec3f worldDeltaTranslation;

                    worldDeltaTranslation = initialRootBBGestureWorld->translation()
                                            - initialRootGesture.T_world.translation();

                    Mat3f R_PJ_transpose = initialRootGesture.T_world.rotation().transpose();
                    Vec3f relativeTranslation = R_PJ_transpose * worldDeltaTranslation;
                    rootLocalTransform.translation() = relativeTranslation;
                    initialRootBBGestureLocal->translation() = rootLocalTransform.translation();

                    Mat3f R_PJ = this->urdfInfo.links[rootLinkIndex].T_world.rotation();
                    Mat3f R_BB;
                    if (!varVisualOrCollision()->getValue()) {
                        R_BB = this->urdfInfo.links[rootLinkIndex].T_visual_bb_world.rotation();
                    } else {
                        R_BB = this->urdfInfo.links[rootLinkIndex].T_collision_bb_world.rotation();
                    }
                    Mat3f relativeRotation = R_PJ.transpose() * R_BB;
                    initialRootBBGestureLocal->rotation() = relativeRotation;

                    updateWorldRecursive(rootLinkIndex, rootWorldTransform);
                }
            }
        }

        Array<Vec3f, DeviceType::CPU> hCenters = gethCenters();
        Array<TQuat, CPU> hAngles = gethAngles();
        Array<Vec3f, DeviceType::CPU> hVelocities = gethVelocities();
        Array<Vec3f, CPU> hAngularVelocities = gethAngularVelocities();
        Array<Mat3f, CPU> hRotations = gethRotationMatrix();

        auto instances = this->varVehiclesTransform()->getValue();

        for (int i = 0; i < hinge_param.num_bodies; i++) {
            auto it = hinge_param.ids[i];
            for (int j = 0; j < ctrl_mb_chains[it].body_indices.size(); j++) {
                auto index = ctrl_mb_chains[it].body_indices[j];
                if (!varVisualOrCollision()->getValue()) {
                    hCenters[index] = this->initialGesture[it].links[j].T_visual_bb_world.translation() + instances[it].translation();
                    hAngles[index] = TQuat(this->initialGesture[it].links[j].T_visual_bb_world.rotation());
                } else {
                    hCenters[index] = this->initialGesture[it].links[j].T_collision_bb_world.translation() + instances[it].translation();
                    hAngles[index] = TQuat(this->initialGesture[it].links[j].T_collision_bb_world.rotation());
                }
                std::cout << "hCenters[" << index << "]: " << hCenters[index] << std::endl;
                hRotations[index] = hAngles[index].toMatrix3x3(); // 必须基于 hAngles
                hVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
                hAngularVelocities[index] = Vec3f(0.0f, 0.0f, 0.0f);
            }
        }

        this->stateCenter()->assign(hCenters);
        this->stateQuaternion()->assign(hAngles);
        this->stateRotationMatrix()->assign(hRotations);
        this->stateVelocity()->assign(hVelocities);
        this->stateAngularVelocity()->assign(hAngularVelocities);
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::resetBatchNonCtrlBodies(BatchRigidBodySystemResetParam& reset_param)
    {
        Array<Vec3f, DeviceType::CPU> hCenters = gethCenters();

        auto instances = this->varVehiclesTransform()->getValue();

        for (int i = 0; i < reset_param.num_bodies; i++) {
            auto it = reset_param.ids[i];
            for (auto index : non_ctrl_mb_chains[it].body_indices) {
                hCenters[index] = reset_param.targetPosition[i] + instances[it].translation();
            }
        }

        this->stateCenter()->assign(hCenters);
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

        for (int i = 0; i < torque_param.num_bodies; i++) {
            auto& mb_chain = ctrl_mb_chains[torque_param.ids[i]];
            for (int j = 0; j < mb_chain.hinge_joint_indices.size(); ++j) {
                auto& joint = this->urdfInfo.joints[mb_chain.hinge_joint_indices[j]];
                auto parentId_local = joint.parentLinkId;
                auto childId_local = joint.childLinkId;
                auto parentId_global = mb_chain.body_indices[parentId_local];
                auto childId_global = mb_chain.body_indices[childId_local];

                auto jointAxisLocal = joint.axisWorld;
                auto hingeTorque = torque_param.torques[i][j] * jointAxisLocal;

                systemTorque[parentId_global] -= hingeTorque;
                systemTorque[childId_global] += hingeTorque;
            }
        }
        this->stateExternalTorque()->assign(systemTorque);
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::setMass(BatchRigidBodySystemMassParam& mass_param) {

        Array<Real, DeviceType::CPU> systemMass;
        systemMass.assign(*this->stateMass()->getDataPtr());

        for (int i = 0; i < mass_param.num_bodies; i++) {
            auto& mb_chain = ctrl_mb_chains[mass_param.ids[i]];
            for (int j = 0; j < mb_chain.body_indices.size(); ++j) {
                systemMass[mb_chain.body_indices[j]] = mass_param.mass[i][j];
            }
        }
        this->stateMass()->assign(systemMass);
    }

    template<typename TDataType>
    void BatchRigidBodySystem<TDataType>::setInertia(BatchRigidBodySystemInertiaParam& inertia_param) {

        Array<Matrix, DeviceType::CPU> systemInertia;
        systemInertia.assign(*this->stateInertia()->getDataPtr());

        for (int i = 0; i < inertia_param.num_bodies; i++) {
            auto& mb_chain = ctrl_mb_chains[inertia_param.ids[i]];
            for (int j = 0; j < mb_chain.body_indices.size(); ++j) {
                systemInertia[mb_chain.body_indices[j]] = inertia_param.inertia[i][j];
            }
        }
        this->stateInertia()->assign(systemInertia);
    }

    template<typename TDataType>
    Array<Vec3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethCenters()
    {
        Array<Vec3f, DeviceType::CPU> hCenters;
        hCenters.assign(*this->stateCenter()->getDataPtr());
        return hCenters;
    }

    template<typename TDataType>
    Array<typename BatchRigidBodySystem<TDataType>::TQuat, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethAngles()
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
    Array<Mat3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethRotationMatrix()
    {
        Array<Mat3f, DeviceType::CPU> hRotationMatrix;
        hRotationMatrix.assign(*this->stateRotationMatrix()->getDataPtr());
        return hRotationMatrix;
    }

    template<typename TDataType>
    Array<typename BatchRigidBodySystem<TDataType>::Real, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethMass()
    {
        Array<Real, DeviceType::CPU> hMass;
        hMass.assign(*this->stateMass()->getDataPtr());
        return hMass;
    }

    template<typename TDataType>
    Array<Mat3f, DeviceType::CPU> BatchRigidBodySystem<TDataType>::gethInertia()
    {
        Array<Mat3f, DeviceType::CPU> hInertia;
        hInertia.assign(*this->stateInertia()->getDataPtr());
        return hInertia;
    }

    template<typename TDataType>
    std::vector<typename BatchRigidBodySystem<TDataType>::TQuat> BatchRigidBodySystem<TDataType>::getAnglesByLocalIndex(
        BatchRigidBodySystemLocalIndexParam& param) {
        Array<TQuat, DeviceType::CPU> hAngles = gethAngles();
        std::vector<TQuat> returnAngles;
        for (int i : param.ids) {
            auto mb = ctrl_mb_chains[i];
            for (int j : param.localRigidBodyid) {
                auto index = mb.body_indices[j];
                returnAngles.push_back(hAngles[index]);
            }
        }
        return returnAngles;
    }

    template<typename TDataType>
    std::vector<Vec3f> BatchRigidBodySystem<TDataType>::getAngularVelocitiesByLocalIndex(
        BatchRigidBodySystemLocalIndexParam& param) {
        Array<Vec3f, DeviceType::CPU> hAngularVelocities = gethAngularVelocities();
        std::vector<Vec3f> returnAngularVelocities;
        for (int i : param.ids) {
            auto mb = ctrl_mb_chains[i];
            for (int j : param.localRigidBodyid) {
                auto index = mb.body_indices[j];
                returnAngularVelocities.push_back(hAngularVelocities[index]);
            }
        }
        return returnAngularVelocities;
    }

    template<typename TDataType>
    std::vector<Vec3f> BatchRigidBodySystem<TDataType>::getCentersByLocalIndex(
        BatchRigidBodySystemLocalIndexParam& param) {
        Array<Vec3f, DeviceType::CPU> hCenters = gethCenters();
        std::vector<Vec3f> returnCenters;
        for (int i : param.ids) {
            auto mb = ctrl_mb_chains[i];
            for (int j : param.localRigidBodyid) {
                auto index = mb.body_indices[j];
                returnCenters.push_back(hCenters[index]);
            }
        }
        return returnCenters;
    }

    template<typename TDataType>
    std::vector<Vec3f> BatchRigidBodySystem<TDataType>::getVelocitiesByLocalIndex(
        BatchRigidBodySystemLocalIndexParam& param) {
        Array<Vec3f, DeviceType::CPU> hVelocities = gethVelocities();
        std::vector<Vec3f> returnVelocities;
        for (int i : param.ids) {
            auto mb = ctrl_mb_chains[i];
            for (int j : param.localRigidBodyid) {
                auto index = mb.body_indices[j];
                returnVelocities.push_back(hVelocities[index]);
            }
        }
        return returnVelocities;
    }

    template<typename TDataType>
    std::vector<typename BatchRigidBodySystem<TDataType>::Real> BatchRigidBodySystem<TDataType>::getMassByLocalIndex(
        BatchRigidBodySystemLocalIndexParam &param) {
        Array<Real, DeviceType::CPU> hMass = gethMass();
        std::vector<Real> returnMass;
        for (int i : param.ids) {
            auto mb = ctrl_mb_chains[i];
            for (int j : param.localRigidBodyid) {
                auto index = mb.body_indices[j];
                returnMass.push_back(hMass[index]);
            }
        }
        return returnMass;
    }

    DEFINE_CLASS(BatchRigidBodySystem);

} // namespace dyno
