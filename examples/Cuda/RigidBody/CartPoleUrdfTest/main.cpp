// #include <QtApp.h>

#include <GlfwApp.h>
#include <SceneGraph.h>

#include <RigidBody/ArticulatedBody.h>
#include <RigidBody/MultibodySystem.h>

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


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto cartpole = scn->addNode(std::make_shared<ArticulatedBody<DataType3f>>());
	cartpole->varFilePath()->setValue(getAssetPath() + "../asset/CartPoleUrdf/cartpole.urdf");

	auto instances = cartpole->varVehiclesTransform()->getValue();
	auto texMesh = cartpole->stateTextureMesh()->constDataPtr();

	std::map<int, std::shared_ptr<PdActor>> actors;

	std::cout << texMesh->shapes().size() << std::endl;
	for (int it = 0; it < texMesh->shapes().size(); it++) {
		RigidBodyInfo rigidbody;

		auto up = texMesh->shapes()[it]->boundingBox.v1;
		auto down = texMesh->shapes()[it]->boundingBox.v0;

		rigidbody.position = Quat1f(instances[0].rotation()).rotate(texMesh->shapes()[it]->boundingTransform.translation())
							+ instances[0].translation();

		rigidbody.angle = Quat1f(instances[0].rotation());
		rigidbody.motionType = BodyType::Dynamic;

		auto actor = cartpole->createRigidBody(rigidbody);
		actors[it] = actor;

		BoxInfo box;

		box.halfLength = (up - down) / 2;

		cartpole->bindBox(actor, box);

		cartpole->bindShape(actor, Pair<uint, uint>(it, 0));
	}

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	cartpole->connect(multibody->importVehicles());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLocation()->setValue(Vec3f(0, 0, 0));
	plane->varScale()->setValue(Vec3f(300.0f));
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	cartpole->stateTopology()->connect(mapper->inDiscreteElements());
	cartpole->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(1, 1, 0));
	sRender->setAlpha(0.2);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	cartpole->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	// QtApp app;
	GlfwApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}


