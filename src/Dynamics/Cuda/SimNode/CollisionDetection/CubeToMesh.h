#pragma once

#include "MeshCollisionTypes.h"

namespace dyno {

    // Generates a unit cube mesh template (vertices at +/-1 in each axis,
    // 8 vertices, 12 triangles) and populates a MeshTemplateData with
    // geometry, patching, topology, and a pre-built rest-space patch BVH.
    //
    // At runtime, per-body scaling is applied using BoxInfo::halfLength.
    template<typename TDataType>
    void GenerateUnitCubeMesh(MeshTemplateData<TDataType>& outTemplate);

} // namespace dyno
