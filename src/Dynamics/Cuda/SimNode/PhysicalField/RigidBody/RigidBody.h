#pragma once

#include "Collision/CollisionData.h"

#include "../PhysicalField.h"

// TODO: consider using a more flexible data structure to support more complex shapes (e.g., triangle mesh) and their parameters. For example, we can have a separate array for each shape type, and store the shape type and offset for each body to access the corresponding shape parameters.
// enum ShapeType
// {
//     Cube,
//     Sphere,
//     Capsule,
//     TriMesh
// };

// inline int GetShapeParamPadding(int shape_type)
// {
//     switch (shape_type)
//     {
//         case ShapeType::Cube:
//             return 3;   // lengths of 3 half-axes
//         case ShapeType::Sphere:
//             return 1;   // radius
//         case ShapeType::Capsule:
//             return 2;   // radius, half-length
//         case ShapeType::TriMesh:
//             return 1;   // mesh idx
//     }

//     return 0;
// }

namespace dyno {
    template<typename TDataType>
    struct RigidBody : public PhysicalFieldData<TDataType>
    {
        using Real = typename TDataType::Real;
        
        DArray<int>         batch_bodies;  // [env_id] num of rigid bodies in each environment
        DArray<int>         batch_body_offset; // [env_id] flattened rigid body offset in topo position/rotation arrays
        DArray<int>         batch_nv;      // [env_id] num of generalized DoFs
        DArray2D<Real>      batch_qacc;    // [env_id, dof_idx] generalized acceleration
        DArray2D<Vec3f>     batch_pos;     // [env_id, body_id] world position of rigid body
        DArray2D<Mat3f>     batch_rot;     // [env_id, body_id] world rotation of rigid body (as rotation matrix)
        DArray<Vec3f>       topo_pos_cache; // flattened body positions for topology update
        DArray<Mat3f>       topo_rot_cache; // flattened body rotations for topology update



        // Shape information for rendering and collision handling
        DArray2D<int>           shape_type;    // [body_id] type
        DArray2D<int>           shape_idx;     // [body_id] index to the corresponding shape parameter array (e.g., box_params, sphere_params, etc.)
        
        DArray<int>             env_num_boxes;  // [env_id] number of boxes in each environment
        DArray<int>             env_box_offset;  // [env_id] offset of boxes in the global box array
        DArray2D<BoxInfo>       boxes;
        
        DArray<int>             env_num_spheres;  // [env_id] number of spheres in each environment
        DArray<int>             env_sphere_offset;  // [env_id] offset of spheres in the global sphere array
        DArray2D<SphereInfo>    spheres;

        DArray<int>             env_num_capsules;  // [env_id] number of capsules in each environment
        DArray<int>             env_capsule_offset;  // [env_id] offset of capsules in the global capsule array
        DArray2D<CapsuleInfo>   capsules;


        DArray<Vec3i>           rendering_idx_2_rigid_body_mapping; // [env_id, shape_type, shape_idx]
        DArray2D<int>           rigid_body_2_rendering_idx_mapping; // [env_id, body_id] -> idx of its pos in topo state
    };
}



