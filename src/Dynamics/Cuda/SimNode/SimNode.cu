#include "SimNode.h"
#include "Utils/utils.h"

namespace dyno
{
        __global__ void PrintBatchNvKernel(DArray<int> batch_nv)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_nv.size())
        {
            printf("batch_nv[%d] = %d\n", idx, batch_nv[idx]);
        }
    }

    template<typename Real>
    __global__ void PrintBatchQaccKernel(DArray2D<Real> batch_qacc)
    {
        uint env_id = blockIdx.x;
        uint dof_idx = threadIdx.x;

        if (env_id < batch_qacc.nx() && dof_idx < batch_qacc.ny())
            printf("batch_qacc(%u, %u) = %f\n", env_id, dof_idx, batch_qacc(env_id, dof_idx));
    }

    __global__ void BindRenderBoxesKernel(
        DArray2D<BoxInfo> boxes,
        DArray<int> env_num_boxes,
        DArray<int> env_box_offset,
        DArray<Box3D> render_boxes)
    {
        int env_id = blockIdx.x;
        if (env_id >= env_num_boxes.size())
            return;
        int num_boxes = env_num_boxes[env_id];
        int box_offset = env_box_offset[env_id];

        for (int local_sid = threadIdx.x; local_sid < num_boxes; local_sid += blockDim.x)
        {
            int render_idx = box_offset + local_sid;
            if (render_idx >= render_boxes.size())
                continue;

            render_boxes[render_idx].center = boxes(env_id, local_sid).center;
            render_boxes[render_idx].extent = boxes(env_id, local_sid).halfLength;

            Mat3f rot = boxes(env_id, local_sid).rot.toMatrix3x3(); 
            
            render_boxes[render_idx].u = rot * Vec3f(1.f, 0.f, 0.f);
            render_boxes[render_idx].v = rot * Vec3f(0.f, 1.f, 0.f);
            render_boxes[render_idx].w = rot * Vec3f(0.f, 0.f, 1.f);
        }
    }

    __global__ void BindRenderSpheresKernel(
        DArray2D<SphereInfo> spheres,
        DArray<int> env_num_spheres,
        DArray<int> env_sphere_offset,
        DArray<Sphere3D> render_spheres)
    {
        int env_id = blockIdx.x;
        if (env_id >= env_num_spheres.size())
            return;
        int num_spheres = env_num_spheres[env_id];
        int sphere_offset = env_sphere_offset[env_id];

        for (int local_sid = threadIdx.x; local_sid < num_spheres; local_sid += blockDim.x)
        {
            int render_idx = sphere_offset + local_sid;
            if (render_idx >= render_spheres.size())
                continue;

            render_spheres[render_idx].center = spheres(env_id, local_sid).center;
            render_spheres[render_idx].radius = spheres(env_id, local_sid).radius;
            render_spheres[render_idx].rotation = spheres(env_id, local_sid).rot;
        }
    }

    __global__ void BindRenderCapsulesKernel(
        DArray2D<CapsuleInfo> capsules,
        DArray<int> env_num_capsules,
        DArray<int> env_capsule_offset,
        DArray<Capsule3D> render_capsules)
    {
        int env_id = blockIdx.x;
        if (env_id >= env_num_capsules.size())
            return;

        int num_capsules = env_num_capsules[env_id];
        int capsule_offset = env_capsule_offset[env_id];

        for (int local_sid = threadIdx.x; local_sid < num_capsules; local_sid += blockDim.x)
        {
            int render_idx = capsule_offset + local_sid;
            if (render_idx >= render_capsules.size())
                continue;

            render_capsules[render_idx].center = capsules(env_id, local_sid).center;
            render_capsules[render_idx].rotation = capsules(env_id, local_sid).rot;
            render_capsules[render_idx].radius = capsules(env_id, local_sid).radius;
            render_capsules[render_idx].halfLength = capsules(env_id, local_sid).halfLength;
        }
    }

    __global__ void InitShape2RigidBodyMappingKernel(DArray<Pair<uint, uint>> mapping)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= mapping.size())
            return;

        mapping[tid] = Pair<uint, uint>((uint)tid, (uint)-1);
    }

    __global__ void BuildShape2RigidBodyMappingKernel(
        DArray<Pair<uint, uint>> mapping,
        DArray2D<int> rigid_body_2_rendering_idx_mapping,
        DArray<int> batch_bodies,
        DArray<int> batch_body_offset)
    {
        int env_id = blockIdx.x;
        if (env_id >= rigid_body_2_rendering_idx_mapping.nx() || env_id >= batch_bodies.size())
            return;

        int num_bodies = batch_bodies[env_id];
        int body_offset = batch_body_offset[env_id];
        for (int body_id = threadIdx.x; body_id < num_bodies; body_id += blockDim.x)
        {
            int render_idx = rigid_body_2_rendering_idx_mapping(env_id, body_id);
            if (render_idx < 0 || render_idx >= mapping.size())
                continue;

            uint global_body_id = (uint)(body_offset + body_id);
            mapping[render_idx] = Pair<uint, uint>((uint)render_idx, global_body_id);
        }
    }


}// for cuda kernels


namespace dyno
{
    // template<typename TDataType>
    // void SimNode<TDataType>::InitRigidBody(int num_env, int num_bodies)
    // {
    //     // ========================= TEST FUNCs =========================
    //     auto AddShapes = [&](RigidBody<TDataType>& rigid_bodies){
    //         const int slots_per_env = 2;
    //         CArray2D<int> shape_type_host(num_env, slots_per_env);
    //         CArray2D<int> shape_idx_host(num_env, slots_per_env);
    //         CArray2D<BoxInfo> boxes_host(num_env, slots_per_env);
    //         CArray2D<SphereInfo> spheres_host(num_env, slots_per_env);
    //
    //         std::vector<int> env_num_boxes_host(num_env, 0);
    //         std::vector<int> env_box_offset_host(num_env, 0);
    //
    //         std::vector<int> env_num_spheres_host(num_env, 0);
    //         std::vector<int> env_sphere_offset_host(num_env, 0);
    //
    //         std::vector<int> env_num_capsules_host(num_env, 0);
    //         std::vector<int> env_capsule_offset_host(num_env, 0);
    //
    //         std::vector<int> batch_bodies_host(num_env, 0);
    //         std::vector<int> batch_body_offset_host(num_env, 0);
    //
    //         CArray2D<Vec3f>     body_pos_host(num_env, num_bodies);
    //         CArray2D<Mat3f>     body_rot_host(num_env, num_bodies);
    //         std::vector<Vec3i>  rendering_idx_2_rigid_body_mapping_host; // [env_id, shape_type, shape_idx]
    //         CArray2D<int>       rigid_body_2_rendering_idx_mapping_host(num_env, num_bodies);
    //
    //         rigid_bodies.env_num_boxes.resize(num_env);
    //         rigid_bodies.env_box_offset.resize(num_env);
    //
    //         rigid_bodies.env_num_spheres.resize(num_env);
    //         rigid_bodies.env_sphere_offset.resize(num_env);
    //
    //         rigid_bodies.env_num_capsules.resize(num_env);
    //         rigid_bodies.env_capsule_offset.resize(num_env);
    //
    //         rigid_bodies.batch_bodies.resize(num_env);
    //         rigid_bodies.batch_body_offset.resize(num_env);
    //
    //         // Initialize all shapes to -1 (indicating no shape)
    //         for (int eid = 0; eid < num_env; ++eid)
    //         {
    //             for (int sid = 0; sid < slots_per_env; ++sid)
    //             {
    //                 shape_type_host(eid, sid) = -1;
    //                 shape_idx_host(eid, sid) = -1;
    //             }
    //
    //             for (int bid = 0; bid < num_bodies; ++bid)
    //             {
    //                 rigid_body_2_rendering_idx_mapping_host(eid, bid) = -1;
    //                 body_rot_host(eid, bid) = Mat3f::identityMatrix();
    //             }
    //         }
    //
    //         // Manually add shapes to environments for testing.
    //         // In a real scenario, this would come from a configuration file or procedural generation logic.
    //
    //         // For example, let's say we have 2 environments.
    //         // The first environment has 2 boxes, and the second environment has 2 spheres.
    //         if (num_env > 0)
    //         {
    //             shape_type_host(0, 0) = 0;
    //             shape_type_host(0, 1) = 0;
    //             shape_idx_host(0, 0) = 0;
    //             shape_idx_host(0, 1) = 1;
    //
    //             boxes_host(0, 0).center = Vec3f(-0.8f, 0.f, 0.0f);
    //             boxes_host(0, 0).halfLength = Vec3f(0.2f, 0.3f, 0.4f);
    //
    //             boxes_host(0, 1).center = Vec3f(-0.5f, 0.f, 0.0f);
    //             boxes_host(0, 1).halfLength = Vec3f(0.5f, 0.2f, 0.2f);
    //
    //             body_pos_host(0, 0) = Vec3f(-0.8f, 0.f, 0.0f);
    //             body_pos_host(0, 1) = Vec3f(-0.5f, 0.f, 0.0f);
    //         }
    //
    //         if (num_env > 1)
    //         {
    //             shape_type_host(1, 0) = 1;
    //             shape_type_host(1, 1) = 1;
    //             shape_idx_host(1, 0) = 0;
    //             shape_idx_host(1, 1) = 1;
    //
    //             spheres_host(1, 0).center = Vec3f(0.5f, 0.0f, 0.0f);
    //             spheres_host(1, 0).radius = Real(0.25);
    //
    //             spheres_host(1, 1).center = Vec3f(0.8f, 0.0f, 0.0f);
    //             spheres_host(1, 1).radius = Real(0.45);
    //
    //             body_pos_host(1, 0) = Vec3f(0.5f, 0.0f, 0.0f);
    //             body_pos_host(1, 1) = Vec3f(0.8f, 0.0f, 0.0f);
    //         }
    //
    //         // 1) Count shapes in each environment.
    //         for (int eid = 0; eid < num_env; ++eid)
    //         {
    //             for (int sid = 0; sid < slots_per_env; ++sid)
    //             {
    //                 int st = shape_type_host(eid, sid);
    //                 if (st == 0) env_num_boxes_host[eid]++;
    //                 else if (st == 1) env_num_spheres_host[eid]++;
    //                 else if (st == 2) env_num_capsules_host[eid]++;
    //             }
    //
    //             int active_bodies = 0;
    //             int body_count = num_bodies < slots_per_env ? num_bodies : slots_per_env;
    //             for (int bid = 0; bid < body_count; ++bid)
    //             {
    //                 if (shape_type_host(eid, bid) >= 0)
    //                     active_bodies++;
    //             }
    //             batch_bodies_host[eid] = active_bodies;
    //         }
    //
    //         // 2) Build per-shape-type global offsets by env.
    //         int total_boxes = 0;
    //         int total_spheres = 0;
    //         int total_capsules = 0;
    //         int total_bodies = 0;
    //         for (int eid = 0; eid < num_env; ++eid)
    //         {
    //             env_box_offset_host[eid] = total_boxes;
    //             env_sphere_offset_host[eid] = total_spheres;
    //             env_capsule_offset_host[eid] = total_capsules;
    //             batch_body_offset_host[eid] = total_bodies;
    //             total_boxes += env_num_boxes_host[eid];
    //             total_spheres += env_num_spheres_host[eid];
    //             total_capsules += env_num_capsules_host[eid];
    //             total_bodies += batch_bodies_host[eid];
    //         }
    //
    //         // 3) Build mapping between rendering index and rigid body index.
    //         // DiscreteElements order is sphere -> box -> tet -> capsule -> triangle.
    //         const int sphere_base = 0;
    //         const int box_base = total_spheres;
    //         const int capsule_base = total_spheres + total_boxes;
    //         const int total_render_shapes = capsule_base + total_capsules;
    //         rendering_idx_2_rigid_body_mapping_host.resize(total_render_shapes, Vec3i(-1, -1, -1));
    //
    //         for (int eid = 0; eid < num_env; ++eid)
    //         {
    //             int body_count = num_bodies < slots_per_env ? num_bodies : slots_per_env;
    //             for (int bid = 0; bid < body_count; ++bid)
    //             {
    //                 int st = shape_type_host(eid, bid);
    //                 int si = shape_idx_host(eid, bid);
    //                 if (st < 0 || si < 0)
    //                     continue;
    //
    //                 int render_idx = -1;
    //                 if (st == 0)
    //                 {
    //                     render_idx = box_base + env_box_offset_host[eid] + si;
    //                 }
    //                 else if (st == 1)
    //                 {
    //                     render_idx = sphere_base + env_sphere_offset_host[eid] + si;
    //                 }
    //                 else if (st == 2)
    //                 {
    //                     render_idx = capsule_base + env_capsule_offset_host[eid] + si;
    //                 }
    //
    //                 if (render_idx >= 0 && render_idx < total_render_shapes)
    //                 {
    //                     rigid_body_2_rendering_idx_mapping_host(eid, bid) = render_idx;
    //                     rendering_idx_2_rigid_body_mapping_host[render_idx] = Vec3i(eid, st, si);
    //                 }
    //             }
    //         }
    //
    //         rigid_bodies.shape_type.assign(shape_type_host);
    //         rigid_bodies.shape_idx.assign(shape_idx_host);
    //         rigid_bodies.boxes.assign(boxes_host);
    //         rigid_bodies.spheres.assign(spheres_host);
    //         rigid_bodies.batch_pos.assign(body_pos_host);
    //         rigid_bodies.batch_rot.assign(body_rot_host);
    //
    //         rigid_bodies.env_num_boxes.assign(env_num_boxes_host);
    //         rigid_bodies.env_box_offset.assign(env_box_offset_host);
    //         rigid_bodies.env_num_spheres.assign(env_num_spheres_host);
    //         rigid_bodies.env_sphere_offset.assign(env_sphere_offset_host);
    //         rigid_bodies.env_num_capsules.assign(env_num_capsules_host);
    //         rigid_bodies.env_capsule_offset.assign(env_capsule_offset_host);
    //
    //         rigid_bodies.batch_bodies.assign(batch_bodies_host);
    //         rigid_bodies.batch_body_offset.assign(batch_body_offset_host);
    //
    //         rigid_bodies.rendering_idx_2_rigid_body_mapping.assign(rendering_idx_2_rigid_body_mapping_host);
    //         rigid_bodies.rigid_body_2_rendering_idx_mapping.assign(rigid_body_2_rendering_idx_mapping_host);
    //
    //         // 到这里假设所有的shape都已经通过配置文件已经分配到不同的envriment了。
    //         // 已经知道了，所有物体在env内部的分布
    //         // 接下来要更新mapping
    //         // std::vector<Vec3i>  rendering_idx_2_rigid_body_mapping_host; // [env_id, shape_type, shape_idx]
    //         // std::vector<int>    rigid_body_2_rendering_idx_mapping_host;
    //
    //         for(int i = 0; i < total_render_shapes; i++)
    //         {
    //             Vec3i mapping = rendering_idx_2_rigid_body_mapping_host[i];
    //             spdlog::info("Render shape {} maps to env {}, shape type {}, shape idx {}", i, mapping.x, mapping.y, mapping.z);
    //         }
    //
    //     };
    //
    //     auto OneCubeCase = [&](RigidBody<TDataType>& rigid_bodies){
    //
    //         const auto& env_infos = var_env_infos.constDataPtr();
    //         env_infos->num_envs = 1;
    //
    //         int envs = 1;
    //         int bodies_per_env = 1;
    //
    //
    //         CArray2D<int> shape_type_host(envs, bodies_per_env);
    //         CArray2D<int> shape_idx_host(envs, bodies_per_env);
    //         CArray2D<BoxInfo> boxes_host(envs, bodies_per_env);
    //         CArray2D<SphereInfo> spheres_host(envs, bodies_per_env);
    //         CArray2D<int> parent_idx_host(envs, bodies_per_env);
    //
    //         std::vector<int> env_num_boxes_host(envs, 0);
    //         std::vector<int> env_box_offset_host(envs, 0);
    //
    //         std::vector<int> env_num_spheres_host(envs, 0);
    //         std::vector<int> env_sphere_offset_host(envs, 0);
    //
    //         std::vector<int> env_num_capsules_host(envs, 0);
    //         std::vector<int> env_capsule_offset_host(envs, 0);
    //
    //         std::vector<int> batch_bodies_host(envs, 0);
    //         std::vector<int> batch_body_offset_host(envs, 0);
    //
    //         CArray2D<Vec3f>     body_pos_host(envs, bodies_per_env);
    //         CArray2D<Mat3f>     body_rot_host(envs, bodies_per_env);
    //         CArray2D<Quat<Real>> batch_quat_host(envs, bodies_per_env);
    //         std::vector<Vec3i>  rendering_idx_2_rigid_body_mapping_host; // [env_id, shape_type, shape_idx]
    //         CArray2D<int>       rigid_body_2_rendering_idx_mapping_host(envs, bodies_per_env);
    //
    //         rigid_bodies.env_num_boxes.resize(envs);
    //         rigid_bodies.env_box_offset.resize(envs);
    //
    //         rigid_bodies.env_num_spheres.resize(envs);
    //         rigid_bodies.env_sphere_offset.resize(envs);
    //
    //         rigid_bodies.env_num_capsules.resize(envs);
    //         rigid_bodies.env_capsule_offset.resize(envs);
    //
    //         rigid_bodies.batch_bodies.resize(envs);
    //         rigid_bodies.batch_body_offset.resize(envs);
    //
    //         // Initialize all shapes to -1 (indicating no shape)
    //         for (int eid = 0; eid < envs; ++eid)
    //         {
    //             for (int sid = 0; sid < bodies_per_env; ++sid)
    //             {
    //                 shape_type_host(eid, sid) = -1;
    //                 shape_idx_host(eid, sid) = -1;
    //             }
    //
    //             for (int bid = 0; bid < bodies_per_env; ++bid)
    //             {
    //                 rigid_body_2_rendering_idx_mapping_host(eid, bid) = -1;
    //                 body_rot_host(eid, bid) = Mat3f::identityMatrix();
    //                 batch_quat_host(eid, bid) = Quat<Real>::identity();
    //             }
    //         }
    //
    //         // Add a cube in env0 manully.
    //         shape_type_host(0, 0) = 0;
    //         shape_idx_host(0, 0) = 0;
    //
    //         boxes_host(0, 0).center = Vec3f(0.f, 0.f, 0.0f);
    //         boxes_host(0, 0).halfLength = Vec3f(0.2f, 0.1f, 0.1f);
    //         // boxes_host(0, 0).rot = Quat<Real>(-0.545f, -0.635f, -0.313f, 0.449f);
    //
    //         body_pos_host(0, 0) = Vec3f(0.f, 1.f, 0.0f);
    //         batch_quat_host(0, 0) = Quat<Real>(-0.545f, -0.635f, -0.313f, 0.449f);
    //         body_rot_host(0, 0) = batch_quat_host(0, 0).toMatrix3x3();
    //
    //         // 1) Count shapes in each environment.
    //         for (int eid = 0; eid < envs; ++eid)
    //         {
    //             for (int sid = 0; sid < bodies_per_env; ++sid)
    //             {
    //                 int st = shape_type_host(eid, sid);
    //                 if (st == 0) env_num_boxes_host[eid]++;
    //                 else if (st == 1) env_num_spheres_host[eid]++;
    //                 else if (st == 2) env_num_capsules_host[eid]++;
    //             }
    //
    //             int active_bodies = 0;
    //             int body_count = 1;
    //             for (int bid = 0; bid < body_count; ++bid)
    //             {
    //                 if (shape_type_host(eid, bid) >= 0)
    //                     active_bodies++;
    //             }
    //             batch_bodies_host[eid] = active_bodies;
    //         }
    //
    //         // 2) Build per-shape-type global offsets by env.
    //         int total_boxes = 0;
    //         int total_spheres = 0;
    //         int total_capsules = 0;
    //         int total_bodies = 0;
    //         for (int eid = 0; eid < envs; ++eid)
    //         {
    //             env_box_offset_host[eid] = total_boxes;
    //             env_sphere_offset_host[eid] = total_spheres;
    //             env_capsule_offset_host[eid] = total_capsules;
    //             batch_body_offset_host[eid] = total_bodies;
    //             total_boxes += env_num_boxes_host[eid];
    //             total_spheres += env_num_spheres_host[eid];
    //             total_capsules += env_num_capsules_host[eid];
    //             total_bodies += batch_bodies_host[eid];
    //         }
    //
    //         // 3) Build mapping between rendering index and rigid body index.
    //         // DiscreteElements order is sphere -> box -> tet -> capsule -> triangle.
    //         const int sphere_base = 0;
    //         const int box_base = total_spheres;
    //         const int capsule_base = total_spheres + total_boxes;
    //         const int total_render_shapes = capsule_base + total_capsules;
    //         rendering_idx_2_rigid_body_mapping_host.resize(total_render_shapes, Vec3i(-1, -1, -1));
    //
    //         for (int eid = 0; eid < envs; ++eid)
    //         {
    //             int body_count = 1;
    //             for (int bid = 0; bid < body_count; ++bid)
    //             {
    //                 int st = shape_type_host(eid, bid);
    //                 int si = shape_idx_host(eid, bid);
    //                 if (st < 0 || si < 0)
    //                     continue;
    //
    //                 int render_idx = -1;
    //                 if (st == 0)
    //                 {
    //                     render_idx = box_base + env_box_offset_host[eid] + si;
    //                 }
    //                 else if (st == 1)
    //                 {
    //                     render_idx = sphere_base + env_sphere_offset_host[eid] + si;
    //                 }
    //                 else if (st == 2)
    //                 {
    //                     render_idx = capsule_base + env_capsule_offset_host[eid] + si;
    //                 }
    //
    //                 if (render_idx >= 0 && render_idx < total_render_shapes)
    //                 {
    //                     rigid_body_2_rendering_idx_mapping_host(eid, bid) = render_idx;
    //                     rendering_idx_2_rigid_body_mapping_host[render_idx] = Vec3i(eid, st, si);
    //                 }
    //             }
    //         }
    //
    //         rigid_bodies.shape_type.assign(shape_type_host);
    //         rigid_bodies.shape_idx.assign(shape_idx_host);
    //         rigid_bodies.boxes.assign(boxes_host);
    //         rigid_bodies.spheres.assign(spheres_host);
    //         rigid_bodies.batch_pos.assign(body_pos_host);
    //         rigid_bodies.batch_rot.assign(body_rot_host);
    //         rigid_bodies.batch_quat.assign(batch_quat_host);
    //
    //         rigid_bodies.env_num_boxes.assign(env_num_boxes_host);
    //         rigid_bodies.env_box_offset.assign(env_box_offset_host);
    //         rigid_bodies.env_num_spheres.assign(env_num_spheres_host);
    //         rigid_bodies.env_sphere_offset.assign(env_sphere_offset_host);
    //         rigid_bodies.env_num_capsules.assign(env_num_capsules_host);
    //         rigid_bodies.env_capsule_offset.assign(env_capsule_offset_host);
    //
    //         rigid_bodies.batch_bodies.assign(batch_bodies_host);
    //         rigid_bodies.batch_body_offset.assign(batch_body_offset_host);
    //
    //         rigid_bodies.rendering_idx_2_rigid_body_mapping.assign(rendering_idx_2_rigid_body_mapping_host);
    //         rigid_bodies.rigid_body_2_rendering_idx_mapping.assign(rigid_body_2_rendering_idx_mapping_host);
    //
    //         // No parent
    //         parent_idx_host(0, 0) = -1;
    //         rigid_bodies.parent_idx.assign(parent_idx_host);
    //
    //         // Not static
    //         CArray2D<int> is_static_host(envs, bodies_per_env);
    //         is_static_host(0, 0) = 0;
    //         rigid_bodies.is_static.assign(is_static_host);
    //
    //         // Mass
    //         CArray2D<Real> mass_host(envs, bodies_per_env);
    //         mass_host(0, 0) = 1000.0f;
    //         rigid_bodies.batch_mass.assign(mass_host);
    //
    //         for(int i = 0; i < total_render_shapes; i++)
    //         {
    //             Vec3i mapping = rendering_idx_2_rigid_body_mapping_host[i];
    //             spdlog::info("Render shape {} maps to env {}, shape type {}, shape idx {}", i, mapping.x, mapping.y, mapping.z);
    //         }
    //
    //     };
    //
    //
    //     // ========================= TEST FUNCs =========================
    //
    //
    //     spdlog::info("Start initializing rigid body state variables.");
    //     // using Real = typename TDataType::Real;
    //
    //     // // batch_nv test
    //     // std::vector<int> batch_nv_host;
    //     // for(int eid = 0; eid < num_env; eid++)
    //     // {
    //     //     batch_nv_host.push_back(num_bodies * (eid + 1));
    //     //     spdlog::info("batch_nv_host[{}] = {}", eid, batch_nv_host[eid]);
    //     // }
    //
    //
    //     // // batch_qacc test
    //     // CArray2D<Real> batch_qacc_host(num_env, num_bodies * 6);
    //     // for(int eid = 0; eid < num_env; eid++)
    //     // {
    //     //     for(int bid = 0; bid < num_bodies; bid++)
    //     //         for(int i = 0; i < 6; i++)
    //     //         {
    //     //             int dof_id = bid * 6 + i;
    //     //             batch_qacc_host(eid, bid * 6 + i) = (10000*eid + dof_id) * 1.0f;
    //     //             spdlog::info("batch_qacc_host({},{}) = {}", eid, bid * 6 + i, batch_qacc_host(eid, bid * 6 + i));
    //     //         }
    //     // }
    //
    //
    //     auto rigid_body = var_rigid_body.getValue();
    //     // rigid_body.batch_nv.resize(num_env);
    //     // rigid_body.batch_nv.assign(batch_nv_host);
    //
    //     // rigid_body.batch_qacc.resize(num_env, num_bodies * 6);
    //     // rigid_body.batch_qacc.assign(batch_qacc_host);
    //
    //     // cuExecute(rigid_body.batch_nv.size(),
    //     //     PrintBatchNvKernel,
    //     //     rigid_body.batch_nv);
    //
    //     // // cuExecute2D(make_uint2(rigid_body.batch_qacc.nx(), rigid_body.batch_qacc.ny()),
    //     // //     PrintBatchQaccKernel<typename TDataType::Real>,
    //     // //     rigid_body.batch_qacc);
    //     // PrintBatchQaccKernel<Real><<<rigid_body.batch_qacc.nx(), rigid_body.batch_qacc.ny()>>>(rigid_body.batch_qacc);
    //
    //     // AddShapes(rigid_body);
    //     OneCubeCase(rigid_body);
    //
    //     var_rigid_body.setValue(rigid_body);
    //
    //
    //     spdlog::info("Finished initializing rigid body state variables.");
    // }

    template<typename TDataType>
    void SimNode<TDataType>::BindRenderingSurface(int num_env)
    {
        auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->statetopology()->getDataPtr());
        auto& topo_boxes = topo->boxesInLocal();
		auto& topo_spheres = topo->spheresInLocal();
		auto& topo_capsules = topo->capsulesInLocal();
        
        
        Reduction<int> reduce_int;
        
        // 1. collect shape information from rigid body
        auto rigid_body = var_rigid_body.constDataPtr();
        int total_boxes = reduce_int.accumulate(rigid_body->env_num_boxes.begin(), rigid_body->env_num_boxes.size());
        int total_spheres = reduce_int.accumulate(rigid_body->env_num_spheres.begin(), rigid_body->env_num_spheres.size());
        int total_capsules = reduce_int.accumulate(rigid_body->env_num_capsules.begin(), rigid_body->env_num_capsules.size());
        topo_boxes.resize(total_boxes);
        topo_spheres.resize(total_spheres);
        topo_capsules.resize(total_capsules);
        
        spdlog::info("Total number of boxes across all environments: {}", total_boxes);
        spdlog::info("Total number of spheres across all environments: {}", total_spheres);
        spdlog::info("Total number of capsules across all environments: {}", total_capsules);

        BindRenderBoxesKernel<<<num_env, 32>>>(rigid_body->boxes, rigid_body->env_num_boxes, rigid_body->env_box_offset, topo_boxes);
        BindRenderSpheresKernel<<<num_env, 32>>>(rigid_body->spheres, rigid_body->env_num_spheres, rigid_body->env_sphere_offset, topo_spheres);
        BindRenderCapsulesKernel<<<num_env, 32>>>(rigid_body->capsules, rigid_body->env_num_capsules, rigid_body->env_capsule_offset, topo_capsules);

        cudaDeviceSynchronize();

        // setup shape to renderable mapping
        auto& mapping = topo->shape2RigidBodyMapping();
        uint totalSize = topo->totalSize();
        spdlog::info("Total size of renderable shapes: {}", totalSize);

        if (mapping.size() != totalSize)
        {
            mapping.resize(totalSize);
            cuExecute(totalSize,
                InitShape2RigidBodyMappingKernel,
                mapping);

            BuildShape2RigidBodyMappingKernel<<<rigid_body->rigid_body_2_rendering_idx_mapping.nx(), 128>>>(
                mapping,
                rigid_body->rigid_body_2_rendering_idx_mapping,
                rigid_body->batch_bodies,
                rigid_body->batch_body_offset);
        }

        int total_rigid_bodies = reduce_int.accumulate(rigid_body->batch_bodies.begin(), rigid_body->batch_bodies.size());
        FlattenArray2D(rigid_body->batch_pos, rigid_body->topo_pos_cache, total_rigid_bodies,
            rigid_body->batch_bodies, rigid_body->batch_body_offset);
        FlattenArray2D(rigid_body->batch_rot, rigid_body->topo_rot_cache, total_rigid_bodies,
            rigid_body->batch_bodies, rigid_body->batch_body_offset);

        topo->setPosition(rigid_body->topo_pos_cache);
        topo->setRotation(rigid_body->topo_rot_cache);
        topo->update();
        
    }

    DEFINE_CLASS(SimNode);
}// dyno