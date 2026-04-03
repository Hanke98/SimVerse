#pragma once

#include <Array/Array.h>
#include <Array/Array2D.h>
#include <STL/Pair.h>
#include <Vector.h>
#include "Utils/SimBlockVector.h"

namespace dyno
{
    
    struct BatchConstraintParas
    {
        DevBlockVector<Real>      time_const;    // constraint time constant
        DevBlockVector<Real>      damp_ratio;    // constraint damping ratio
        DevBlockVector<Real>      dmax;          // limit of impedance
        DevBlockVector<Real>      dmin;          // lower limit of impedance
        DevBlockVector<Real>      width;         // domain of the impedance function
        DevBlockVector<Real>      midpoint;      // inflection point of the impedance function
        DevBlockVector<int>       power;         // power of impedance
    };

    struct BatchCollisionConstraints : public BatchConstraintParas
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
        DevBlockVector<int>              dof_idxs;
        DevBlockVector<Real>             dof_frictionloss;
    };

    struct BatchAnchorConstraints: public BatchConstraintParas
    {
        DevBlockVector<Pair<int, int>>   body_idxs;         // [env_id, constraint_id] pair of body indices involved in the constraint
        DevBlockVector<Vec3f>            anchor_A_local;    // [env_id, constraint_id] anchor point in local frame of body A
        DevBlockVector<Vec3f>            anchor_B_local;    // [env_id, constraint_id] anchor point in local frame of body B
        DevBlockVector<Vec3f>            anchor_A_world;    // [env_id, constraint_id] anchor point in world frame of body A
        DevBlockVector<Vec3f>            anchor_B_world;    // [env_id, constraint_id] anchor point in world frame of body B
        DevBlockVector<Vec3f>            anchor_error;      // [env_id, constraint_id] anchor error (world_A - world_B)
    };

    struct BatchJointLimitConstraints : public BatchConstraintParas
    {
        DArray<int>                   ref_nums;
        DevBlockVector<int>           active_mapping;
        DevBlockVector<int>           is_active;
        DevBlockVector<int>           joint_idx;        // i.e., body index
        DevBlockVector<int>           is_upper;         // 0 if lower limit, 1 if upper limit`
        DevBlockVector<Real>          limit;            // lower limit or upper limit

        DevBlockVector<Real>          limit_error;
        DevBlockVector<Vec3f>         limit_extern;
    };

}