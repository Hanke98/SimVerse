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

        // Init();
        Init("assets");
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
        env_infos.num_envs = 1;
        
        CArray<Vec3f> gravities(env_infos.num_envs);
        CArray<Real> timesteps(env_infos.num_envs);
        for (int i = 0; i < env_infos.num_envs; ++i)
        {
            gravities[i] = Vec3f(0.f, -9.81f, 0.f);
            timesteps[i] = 6e-3f;
        }
        env_infos.gravities.assign(gravities);
        env_infos.timesteps.assign(timesteps);       


        var_env_infos.setValue(env_infos);

        InitRigidBody(env_infos.num_envs, 3);    // TEST;

        BindRenderingSurface(env_infos.num_envs);
        
    }

    template<typename TDataType>
    void SimNode<TDataType>::Init(const std::string &root_dir)
    {
        this->statetopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());

        LoadAssets(root_dir);

        auto env_infos = var_env_infos.getValue();

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