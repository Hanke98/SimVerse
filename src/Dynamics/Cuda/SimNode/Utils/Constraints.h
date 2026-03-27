#pragma once

#include <Array/Array.h>
#include <Array/Array2D.h>
#include <STL/Pair.h>
#include <Vector.h>

namespace dyno
{
    
    struct BatchConstraintParas
    {
        DArray2D<Real>      time_const;    // [env_id, constraint_id] constraint time constant
        DArray2D<Real>      damp_ratio;   // [env_id, constraint_id] constraint damping ratio
        DArray2D<Real>      dmax;          // [env_id, constraint_id] limit of impedance
        DArray2D<Real>      dmin;          // [env_id, constraint_id] lower limit of impedance
        DArray2D<Real>      width;         // [env_id, constraint_id] domain of the impedance function
        DArray2D<Real>      midpoint;      // [env_id, constraint_id] inflection point of the impedance function
        DArray2D<int>       power;         // [env_id, constraint_id] power of impedance
    };

    struct CollisionConstraintParas
    {
        Real    time_const;
        Real    damp_ratio;
        Real    dmax;
        Real    dmin;
        Real    width;
        Real    midpoint;
        int     power;
    };

    struct BatchCollisionConstraints
    {
        DArray<int>                 collision_nums;    // [env_id] number of collision constraints in each environment
        DArray2D<Pair<int, int>>    body_idxs;         // [env_id, constraint_id] pair of body indices involved in the constraint
        DArray2D<Real>              depth;             // [env_id, constraint_id] penetration depth of the constraint
        DArray2D<Vec3f>             normal;            // [env_id, constraint_id] contact normal of the constraint
        DArray2D<Vec3f>             point;             // [env_id, constraint_id] contact point of the constraint
        DArray2D<Real>              mu;                // [env_id, constraint_id] friction coefficient of the constraint
    };

    struct BatchFrictionLossConstraints : public BatchConstraintParas
    {
        DArray2D<int>              dof_idxs;
        DArray2D<Real>             dof_frictionloss;
    };

    struct BatchAnchorConstraints: public BatchConstraintParas
    {
        DArray<int>                anchor_nums;
        DArray2D<Pair<int, int>>   body_idxs;         // [env_id, constraint_id] pair of body indices involved in the constraint
        DArray2D<Vec3f>            anchor_A_local;    // [env_id, constraint_id] anchor point in local frame of body A
        DArray2D<Vec3f>            anchor_B_local;    // [env_id, constraint_id] anchor point in local frame of body B
        DArray2D<Vec3f>            anchor_A_world;    // [env_id, constraint_id] anchor point in world frame of body A
        DArray2D<Vec3f>            anchor_B_world;    // [env_id, constraint_id] anchor point in world frame of body B
        DArray2D<Vec3f>            anchor_error;      // [env_id, constraint_id] anchor error (world_A - world_B)
    };

    struct BatchJointLimitConstraints : public BatchConstraintParas
    {
        DArray<int>             ref_nums;
        DArray2D<int>           active_mapping;
        DArray2D<int>           is_active;
        DArray2D<int>           joint_idx;
        DArray2D<int>           is_upper;
        DArray2D<Real>          limit;

        DArray2D<Real>          limit_error;      // active_nums
        DArray2D<Vec3f>         limit_extern;     // active nums
    };


    // TODO: Other constraint types
}