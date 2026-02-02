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
	multiRobotArm->varFilePath()->setValue(getAssetPath() + "../asset/NTQ_test/scene_cube_sphere_cube.urdf");

	std::vector<Transform3f> vehiclesTransform;
	Transform3f Transform0(Vec3f(0.0f), Quat1f(0.0f, 0.0f, 0.0f, 1.0f).toMatrix3x3(), Vec3f(1.0f));
	vehiclesTransform.push_back(Transform0);

	multiRobotArm->varVehiclesTransform()->setValue(vehiclesTransform);
	auto instances = multiRobotArm->varVehiclesTransform()->getValue();
	auto texMesh = multiRobotArm->stateTextureMesh()->constDataPtr();

	std::map<int, std::shared_ptr<PdActor>> actors;

	std::cout << texMesh->shapes().size() << std::endl;
	std::cout << instances.size() << std::endl;

    multiRobotArm->mTextureMeshShape2ElementIds.clear();

	for (int i = 0; i < instances.size(); i++) {
		BatchRigidBodySystem<DataType3f>::MulitBodyChainIndices mb;
		for (int it = 0; it < texMesh->shapes().size(); it++) {
			RigidBodyInfo rigidbody;

			auto up = texMesh->shapes()[it]->boundingBox.v1;
			auto down = texMesh->shapes()[it]->boundingBox.v0;

			rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation())
								+ instances[i].translation();

			rigidbody.angle = Quat1f(instances[i].rotation());
			rigidbody.motionType = BodyType::Dynamic;

			auto actor = multiRobotArm->createRigidBody(rigidbody);
			actors[it] = actor;

			BoxInfo box;

			box.halfLength = (up - down) / 2;

            int oldBoxCount = multiRobotArm->getHostBoxesSize();

			multiRobotArm->bindBox(actor, box, 1000);
			multiRobotArm->bindShape(actor, Pair<uint, uint>(it, i));
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
                    it,
                    oldBoxCount,
                    newBoxCount);
            }

            // Store the mapping from texture mesh shape to element id
            // auto& entry = mTextureMeshShape2ElementIds[it];
            Pair<uint, uint> entry;
            entry.first = it;
            entry.second = boxLocalId;
            // multiRobotArm->mTextureMeshShape2ElementIds.push_back(entry);
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
    multiRobotArm->setupNeighborTriMeshQueryFromUrdf();



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
	GlfwApp app;
	app.setSceneGraph(creatScene());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
