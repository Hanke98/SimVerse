//
// Created by wjv on 2025/11/18.
//
#pragma once

#include "Topology/TextureMesh.h"
#include <Field/FilePath.h>
#include "UrdfParser.h"


namespace dyno {
    bool loadURDFTextureMesh(std::shared_ptr<TextureMesh> texMesh, const FilePath& urdfFullPath, UrdfInformation& urdfInfo, bool objYUp);
}
