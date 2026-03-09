#include <GlfwApp.h>
#include <SceneGraph.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <Mapping/DiscreteElementsToTriangleSet.h>

#include <SimNode/SimNode.h>
#include <SimNode/SimModule.h>

std::shared_ptr<dyno::SceneGraph> CreateScene()
{
	// Create a scene graph
	std::shared_ptr<dyno::SceneGraph> scn = std::make_shared<dyno::SceneGraph>();

	// Create a SimNode and add it to the scene graph
	auto sim_node = scn->addNode(std::make_shared<dyno::SimNode<dyno::DataType3f>>("SimNodeZJU"));

	// Create SimModule and connect it to the SimNode
	auto sim_module = std::make_shared<dyno::SimModule<dyno::DataType3f>>();
	sim_module->varForceUpdate()->setValue(true);
	sim_node->varenv_infos()->connect(sim_module->inenv_infos());
	sim_node->varrigid_body()->connect(sim_module->inrigid_body());
	sim_node->statetopology()->connect(sim_module->intopology());
	sim_module->Init();
	sim_node->animationPipeline()->pushModule(sim_module);

	// Rendering
	auto mapper = std::make_shared<dyno::DiscreteElementsToTriangleSet<dyno::DataType3f>>();
	sim_node->statetopology()->connect(mapper->inDiscreteElements());
	sim_node->graphicsPipeline()->pushModule(mapper);

	auto surface_render = std::make_shared<dyno::GLSurfaceVisualModule>();
	surface_render->setColor(dyno::Color(1.f, 1.f, 0.f));
	surface_render->setAlpha(0.5f);
	mapper->outTriangleSet()->connect(surface_render->inTriangleSet());
	sim_node->graphicsPipeline()->pushModule(surface_render);

	return scn;
}


int main()
{
	spdlog::info("SimNode Init Example");

	dyno::GlfwApp app;
	app.setSceneGraph(CreateScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}