//
// Created by wjv on 2025/11/28.
//
#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/DiscreteElements.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
    template<typename TDataType>
    class DiscreteSpheresToTriangleSet : public TopologyMapping
    {
        DECLARE_TCLASS(DiscreteElementsToTriangleSet, TDataType);
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;

        DiscreteSpheresToTriangleSet();

    protected:
        bool apply() override;

    public:
        DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

        DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

    private:
        TriangleSet<TDataType> mStandardSphere;
    };

    IMPLEMENT_TCLASS(DiscreteSpheresToTriangleSet, TDataType);
}