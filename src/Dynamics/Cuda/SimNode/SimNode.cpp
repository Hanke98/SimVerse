#include "SimNode.h"
#include <GLWireframeVisualModule.h>


namespace dyno {

    template<typename TDataType>
    SimNode<TDataType>::SimNode() : Node()
    {

    }

    template<typename TDataType>
    SimNode<TDataType>::SimNode(std::string name) : Node()
    {
        this->setName(name);
        spdlog::info("SimNode constructor called for node: {}", name);

        Init();
        // BuildAxes();
        PlotWorldAxes();
    }

    template<typename TDataType>
    SimNode<TDataType>::~SimNode()
    {
        ;
    }

    template<typename TDataType>
    void SimNode<TDataType>::Init()
    {
        this->statetopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());

        auto env_infos = var_env_infos.getValue();
        env_infos.num_envs = 2;
        var_env_infos.setValue(env_infos);

        InitRigidBody(env_infos.num_envs, 3);    // TEST;

        BindRenderingSurface(env_infos.num_envs);
        
    }

    template<typename TDataType>
    void SimNode<TDataType>::PlotWorldAxes()
    {
        std::vector<dyno::Vec3f> points = {
                dyno::Vec3f(0.f, 0.f, 0.f),
                dyno::Vec3f(5.f, 0.f, 0.f),
                dyno::Vec3f(0.f, 5.f, 0.f),
                dyno::Vec3f(0.f, 0.f, 5.f)};
            
        std::vector<dyno::TopologyModule::Edge> edge_x = { {0, 1} };
        std::vector<dyno::TopologyModule::Edge> edge_y = { {0, 2} };
        std::vector<dyno::TopologyModule::Edge> edge_z = { {0, 3} };
        
        // auto edge_set_x = std::make_shared<dyno::EdgeSet<TDataType>>();
        // edge_set_x->setPoints(points);
        // edge_set_x->setEdges(edge_x);
        // this->stateaxis_x()->setDataPtr(edge_set_x);
        // auto x_render = std::make_shared<dyno::GLWireframeVisualModule>();
        // x_render->setColor(dyno::Color(1.f, 0.f, 0.f));
        // x_render->varLineWidth()->setValue(10.f);
        // this->stateaxis_x()->connect(x_render->inEdgeSet());
        // this->graphicsPipeline()->pushModule(x_render);
        // x_render->varForceUpdate()->setValue(true);

        #define SET_AXIS_RENDER(axis, color) \
        auto edge_set_##axis = std::make_shared<dyno::EdgeSet<TDataType>>(); \
        edge_set_##axis->setPoints(points);  \
        edge_set_##axis->setEdges(edge_##axis);   \
        this->stateaxis_##axis()->setDataPtr(edge_set_##axis);    \
        auto render_##axis = std::make_shared<dyno::GLWireframeVisualModule>();  \
        render_##axis->setColor(color); \
        render_##axis->varLineWidth()->setValue(3.f);   \
        this->stateaxis_##axis()->connect(render_##axis->inEdgeSet());    \
        this->graphicsPipeline()->pushModule(render_##axis); \
        render_##axis->varForceUpdate()->setValue(true);

        SET_AXIS_RENDER(x, Color(1.f, 0.f, 0.f));
        SET_AXIS_RENDER(y, Color(0.f, 1.f, 0.f));
        SET_AXIS_RENDER(z, Color(0.f, 0.f, 1.f));

    }

    DEFINE_CLASS(SimNode)
}// namespace dyno