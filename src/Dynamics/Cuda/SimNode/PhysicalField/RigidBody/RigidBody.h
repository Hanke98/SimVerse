#pragma once

#include "../PhysicalNode.h"

template<typename TDataType>
struct RigidBody : public PhysicalNode<TDataType>
{
    using Real = typename TDataType::Real;

    dyno::DArray<int>      batch_nv;      // [env_id] num of generalized DoFs
    dyno::DArray2D<Real>   batch_qacc;    // [dof_id, env_id] generalized acceleration
};