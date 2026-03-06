// #include <QtApp.h>

#include <GlfwApp.h>
#include <SceneGraph.h>

#include <RigidBody/ArticulatedBody.h>
// #include <RigidBody/MultibodySystem.h>
#include <RigidBody/BatchRigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include <Module/GLPhotorealisticInstanceRender.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"
#include "RigidBody/BatchRigidBodySystem.h"
#include "RigidBody/Vehicle.h"
#include "Vector/Vector3D.h"

#include <cstdlib>


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto multiRobotArm = scn->addNode(std::make_shared<BatchRigidBodySystem<DataType3f>>());
	multiRobotArm->varFilePath()->setValue(getAssetPath() + "../asset/NTQ_test/scene_cube_cube_cube.urdf");

	std::vector<Transform3f> vehiclesTransform;
	Transform3f Transform0(Vec3f(0.0f), Quat1f(0.0f, 0.0f, 0.0f, 1.0f).toMatrix3x3(), Vec3f(1.0f));
	Transform3f Transform1(Vec3f(0.0f, 0.0f, 6.0f), Quat1f(0.0f, 0.0f, 0.0f, 1.0f).toMatrix3x3(), Vec3f(1.0f));
	Transform3f Transform2(Vec3f(6.0f, 0.0f, 0.0f), Quat1f(0.0f, 0.0f, 0.0f, 1.0f).toMatrix3x3(), Vec3f(1.0f));
	Transform3f Transform3(Vec3f(6.0f, 0.0f, 6.0f), Quat1f(0.0f, 0.0f, 0.0f, 1.0f).toMatrix3x3(), Vec3f(1.0f));

	// int transformMode = 1;
	// if (const char* v = std::getenv("NMQ_SCENE_TRANSFORM"))
	// 	transformMode = std::atoi(v);

	// if (transformMode == 0)
	// 	vehiclesTransform.push_back(Transform0);
	// else
	// 	vehiclesTransform.push_back(Transform1);

	// const Vec3f& tScene = (transformMode == 0) ? Transform0.translation() : Transform1.translation();
	// printf("[NMQ_test] scene_transform=%d translation=(%.6f, %.6f, %.6f)\n",
	// 	transformMode,
	// 	(double)tScene.x,
	// 	(double)tScene.y,
	// 	(double)tScene.z);

	vehiclesTransform.push_back(Transform0);
	vehiclesTransform.push_back(Transform1);
	vehiclesTransform.push_back(Transform2);
	vehiclesTransform.push_back(Transform3);

	multiRobotArm->varVehiclesTransform()->setValue(vehiclesTransform);
	auto instances = multiRobotArm->varVehiclesTransform()->getValue();
	auto texMesh = multiRobotArm->stateTextureMesh()->constDataPtr();
	const auto& urdfLinks = multiRobotArm->urdfInfo.links;
	const int baseShapeCount = static_cast<int>(urdfLinks.size());

	std::map<int, std::shared_ptr<PdActor>> actors;

	std::cout << texMesh->shapes().size() << std::endl;
	std::cout << instances.size() << std::endl;

    multiRobotArm->mTextureMeshShape2ElementIds.clear();

	for (int i = 0; i < instances.size(); i++) {
		BatchRigidBodySystem<DataType3f>::MulitBodyChainIndices mb;
		for (int localShapeId = 0; localShapeId < baseShapeCount; localShapeId++) {
			const auto& link = urdfLinks[localShapeId];
			const uint renderShapeId = multiRobotArm->varVisualOrCollision()->getValue()
				? link.collisionShapeId
				: link.visualShapeId;
			if (renderShapeId >= texMesh->shapes().size())
				continue;

			RigidBodyInfo rigidbody;

			auto up = texMesh->shapes()[renderShapeId]->boundingBox.v1;
			auto down = texMesh->shapes()[renderShapeId]->boundingBox.v0;

			rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[renderShapeId]->boundingTransform.translation())
								+ instances[i].translation();

			rigidbody.angle = Quat1f(instances[i].rotation());
			rigidbody.motionType = BodyType::Dynamic;

			auto actor = multiRobotArm->createRigidBody(rigidbody);
			actors[renderShapeId] = actor;

			BoxInfo box;

			box.halfLength = (up - down) / 2;

            int oldBoxCount = multiRobotArm->getHostBoxesSize();

			multiRobotArm->bindBox(actor, box, 1000000);
			multiRobotArm->bindShape(actor, Pair<uint, uint>(renderShapeId, i));
			mb.body_indices.push_back(actor->idx);

            int newBoxCount = multiRobotArm->getHostBoxesSize();
            uint boxLocalId = -1;
            if (oldBoxCount >= 0 && newBoxCount == oldBoxCount + 1)
            {
                boxLocalId = (uint)(newBoxCount - 1);
            }
            else
            {
                printf("[BatchRigidBodySystem] TextureMesh shape to box mapping mismatch (shapeId=%u, oldBoxCount=%d, newBoxCount=%d).\n",
                    renderShapeId,
                    oldBoxCount,
                    newBoxCount);
            }

			// NeighborTriMeshQuery expects one contiguous global shape id per instance/link pair.
			const uint globalShapeId = static_cast<uint>(i * baseShapeCount + localShapeId);
            Pair<uint, uint> entry;
            entry.first = globalShapeId;
            entry.second = boxLocalId;
			multiRobotArm->pushBackShape2ElementIds(entry);
		}
		multiRobotArm->pushBackCtrlMBChain(mb);
	}

    {
        auto topo = multiRobotArm->stateTopology()->getDataPtr();
        if (topo == nullptr)
        {
            printf("[BatchRigidBodySystem] TextureMesh shape to element mapping not ready yet (topology unavailable).\n");
        }
        else
        {
            auto elementOffset = topo->calculateElementOffset();
            uint boxStart = (uint)elementOffset.boxIndex();
            for (auto& entry : multiRobotArm->mTextureMeshShape2ElementIds)
            {
                entry.second = boxStart + entry.second;
				multiRobotArm->pushBackShape2ElementIdsDense(entry.second);
            }
            printf("[BatchRigidBodySystem] TextureMesh shape to element mapping ready.\n");
        }
    }



	// multiRobotArm->setupNeighborMeshQueryFromUrdf();

	// auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	// multiRobotArm->connect(multibody->importVehicles());

	// auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	// plane->varLocation()->setValue(Vec3f(0, 0, 0));
	// plane->varScale()->setValue(Vec3f(300.0f));
	// plane->stateTriangleSet()->connect(multibody->inTriangleSet());

	// auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	// multiRobotArm->stateTopology()->connect(mapper->inDiscreteElements());
	// multiRobotArm->graphicsPipeline()->pushModule(mapper);
	//
	// auto sRender = std::make_shared<GLSurfaceVisualModule>();
	// sRender->setColor(Color(1, 1, 0));
	// sRender->setAlpha(0.2);
	// mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	// multiRobotArm->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	auto scene = creatScene();
	bool headless = false;
	if (const char* v = std::getenv("NMQ_HEADLESS"))
		headless = std::atoi(v) != 0;

	if (headless)
	{
		int frames = 240;
		if (const char* v = std::getenv("NMQ_HEADLESS_FRAMES"))
		{
			int n = std::atoi(v);
			if (n > 0)
				frames = n;
		}

		float dt = 1.0f / 60.0f;
		if (const char* v = std::getenv("NMQ_HEADLESS_DT"))
		{
			float parsed = (float)std::atof(v);
			if (parsed > 0.0f)
				dt = parsed;
		}

		printf("[NMQ_test] headless=1 frames=%d dt=%.6f\n", frames, (double)dt);
		scene->reset();
		scene->setFrameRate(1.0f / dt);
		for (int i = 0; i < frames; ++i)
			scene->takeOneFrame();
		return 0;
	}

	GlfwApp app;
	app.setSceneGraph(scene);
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	// app.renderWindow()->getCamera()->setUnitScale(7.0f);
	// app.renderWindow()->getCamera()->setEyePos(Vec3f(1.36, 1.6, 2.44));
	// app.renderWindow()->getCamera()->setTargetPos(Vec3f(0, 1.1, 0));

	app.renderWindow()->getCamera()->setUnitScale(10.0f);

	app.mainLoop();

	return 0;
}
