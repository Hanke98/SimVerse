#pragma once

#include "Collision/CollisionData.h"

#include "../PhysicalField.h"
#include "../../Utils/Constraints.h"

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

        int                 max_bodies;
        DArray2D<int>       is_static;      // [env_id, body_id] whether the rigid body is static or dynamic
        DArray2D<int>       is_isolated;    // [env_id, body_id] whether the rigid body is isolated (not in contact with any other body)

        DArray<int>         batch_bodies;  // [env_id] num of rigid bodies in each environment
        DArray<int>         batch_body_offset; // [env_id] flattened rigid body offset in topo position/rotation arrays
    
        DArray<int>         batch_nv;      // [env_id] num of generalized DoFs
        DArray2D<int>       nv_offset;     // [env_id, body_idx] 
        
        
        DArray2D<int>       q_lengths;      // [env_id, body_id] num of generalized DoFs
        DArray2D<int>       q_offset;       // [env_id, body_id] offset of generalized DoFs;
        DArray2D<Real>      batch_qacc;    // [env_id, dof_idx] generalized acceleration
        DArray2D<Real>      batch_qvel;    // [env_id, dof_idx] generalized velocity
        DArray2D<Real>      batch_aref;    // nc * 1
        DArray2D<Real>      batch_Jaref;   // J * qacc
        DArray2D<Real>      batch_imp;
        DArray<Real>        batch_energy;
        DArray<Real>        batch_energy_ref;
        DArray2D<Real>      batch_constraint_energy;    
        DArray2D<int>       batch_unquads;   // [env_id, constraint_idx]
        DArray2D<Real>      batch_H;
        DArray2D<Real>      batch_dx;    // [env_id, dof_idx] delta for current Newton iteration


        DArray2D<int>       qpos_lengths;       // [env_id, body_id] num of generalized position
        DArray2D<int>       qpos_offset;
        DArray2D<Real>      batch_qpos;     // [env_id, dof_idx] generalized position
        
        DArray2D<Real>      batch_qM;      // [env_id, dof_idx] mass matrix    不应该显式的存，直接存LDL^T
        DArray2D<Vec3f>     batch_inertia;
        DArray2D<Real>      batch_qM_inv;
        // DArray2D<Real>      batch_qM_L;     // [env_id, dof_idx] lower triangular matrix L in the LDL^T decomposition of the mass matrix
        DArray2D<Real>      batch_qM_diag_elem;
        DArray<Real>        batch_scale;
        DArray2D<Real>      batch_cdof;    // [env_id, dof_idx] projection basis
        DArray2D<Real>      batch_cdof_dot; // [env_id, dof_idx] projection basis time derivative
        DArray2D<Real>      batch_crb;     // dense vec num_bodies * 10
        DArray2D<int>       batch_q_chain;

        DArray2D<Real>      dof_frictionloss;


        DArray2D<Vec3f>     batch_pos;     // [env_id, body_id] world position of rigid body
        DArray2D<Mat3f>     batch_rot;     // [env_id, body_id] world rotation of rigid body (as rotation matrix)
        DArray2D<Quat<Real>> batch_quat;    // [env_id, body_id] world rotation of rigid body (as quaternion)
        DArray2D<Real>      batch_mass;    // [env_id, body_id] mass of rigid body
        
        DArray<Vec3f>       topo_pos_cache; // flattened body positions for topology update
        DArray<Mat3f>       topo_rot_cache; // flattened body rotations for topology update

        // For articulated bodies
        DArray2D<int>       parent_idx;   // [env_id, body_id] parent body index (-1 for root)
        DArray2D<int>       root_idx;     // [env_id, body_id] root body index
        DArray2D<Real>      subtree_mass;   // [env_id, body_id] mass of the subtree rooted at this body (including itself and all its children in the kinematic tree)
        DArray2D<Vec3f>     subtree_com;    // [env_id, body_id] center of mass of the subtree rooted at this body (including itself and all its children in the kinematic tree)
        DArray2D<Real>      subtree_inertia;
        DArray2D<Real>      subtree_com_vel;

        // solving cache
        DArray2D<Real>      batch_q_inner_force;
        DArray2D<Real>      batch_q_ex_force;
        DArray2D<Real>      batch_q_ex_acc;
        DArray2D<Real>      batch_Ma;       // qM * qacc
        DArray2D<Real>      batch_grad;     // nv * 1, Newton gradient: Ma - q_ex_force - J^T * constraint_force
        DArray2D<Real>      batch_weight_inv;    // nbody * 1
        DArray2D<Real>      batch_dof_weight_inv; // nv * 1
        DArray2D<Real>      batch_dA;       // nc * 1
        DArray2D<Real>      batch_D;        // nc * 1
        DArray<int>         is_converged;
        DArray<Real>        sys_alpha;

        DArray2D<Real>      Mat_temp1;
        DArray2D<Real>      Mat_temp2;

        // For constraints
        DArray2D<Real>               batch_J;           // [env_id, num_constraints * max_dof] Jacobian matrix of constraints
        DArray<int>                  num_constraints;   // [env_id] number of constraints in each environment = num_collision_constraints + num_topo_invariant_constraints
        DArray<Vec4i>                num_each_constraint; // [env_id] num of each type of constraint (Vec4i: [0 ~ 2] topo_invariant_constraints, [3] collision_constraints)
        DArray<Vec4i>                constraint_offset;   // [env_id] offset of each type of constraint in the batch_J (Vec4i: [0 ~ 2] topo_invariant_constraints, [3] collision_constraints)
        DArray2D<Real>               batch_constraint_vel;
        DArray2D<Real>               batch_constraint_force;


        BatchConstraintParas         constraint_paras;
        // Collision
        CollisionConstraintParas     collision_paras;
        BatchCollisionConstraints    collision_constraints;
        

        // For joint
        DArray2D<int>           joint_type;     // [env_id, body_id] type of joint (0: none, 1: hinge, 2: slide, 3: ball)
        DArray2D<Real>          joint_qpos;
        DArray2D<Real>          joint_qpos_ref;
        DArray2D<int>           joint_qpos_offset;
        DArray2D<Vec3f>         joint_rel_pos;
        DArray2D<Quat<Real>>    joint_rel_quat;
        DArray2D<Vec3f>         joint_axis;     // [env_id, body_id] joint axis for hinge and slide joint, or initial relative rotation axis for ball joint
        DArray2D<Vec3f>         joint_axis_ref;
        DArray2D<Vec3f>         joint_anchor;   // [env_id, body_id] joint anchor in the local frame of the body
        DArray2D<Vec3f>         joint_anchor_ref;
        DArray2D<Real>          batch_cacc;
        DArray2D<Real>          batch_cforce;


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



