//
// Created by wjv on 2025/11/18.
//
#pragma once

#include "Topology/TextureMesh.h"
#include <Field/FilePath.h>
#include "UrdfParser.h"
// #inclued "Material.h"


namespace dyno {
    bool loadURDFTextureMesh(std::shared_ptr<TextureMesh> texMesh, const FilePath& urdfFullPath, UrdfParser& parser);
    // bool loadURDFTextureMesh(std::shared_ptr<TextureMesh> texMesh, const std::string& filepath, bool useToCenter = true);
    // bool loadTextureMeshFromObj_new(const FilePath& meshPath,
    //                         std::vector<dyno::Vec3f>& outVertices,
    //                         std::vector<dyno::Vec3f>& outNormals,
    //                         std::vector<dyno::Vec2f>& outTexCoords,
    //                         std::vector<uint>& outShapeIds,
    //                         std::vector<std::shared_ptr<Material>>& outMaterials);
}
