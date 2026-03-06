#pragma once

#include "Collision/CollisionData.h"

#include "../PhysicalNode.h"


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
    struct RigidBody : public PhysicalNode<TDataType>
    {
        // template<typename T>
        // using DArray = dyno::DArray<T>;
        // template<typename T>
        // using DArray2D = dyno::DArray2D<T>;
        using Real = typename TDataType::Real;
        

        DArray<int>      batch_nv;      // [env_id] num of generalized DoFs
        DArray2D<Real>   batch_qacc;    // [dof_id, env_id] generalized acceleration


        // Shape information for rendering and collision handling
        DArray2D<int>           shape_type;    // [body_id] type
        DArray2D<int>           shape_idx;     // [body_id] index to the corresponding shape parameter array (e.g., box_params, sphere_params, etc.)
        DArray2D<BoxInfo>       boxes;
        DArray2D<SphereInfo>    spheres;
        DArray2D<CapsuleInfo>   capsules;

        // DArray<int>      shape_offset;  // [body_id] offset in shape parameter array
        // DArray<Real>     shape_params;  // [param_idx ~ param_idx + params_padding] shape parameters (e.g., half extents for box, radius for sphere, etc.)

    };
}



