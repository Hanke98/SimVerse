#pragma once

#include <string>

// Forward declarations
namespace tinygltf {
    class Model;
}

template <typename T>
int addAccessor(tinygltf::Model &model, const std::vector<T> &data,
                int type, int componentType, int elemSize);

/// rpy (roll pitch yaw) 转四元数
std::array<double, 4> rpyToQuat(double roll, double pitch, double yaw);

/// 将URDF文件转换为glTF格式
bool urdfToGltf(const std::string &urdfPath, const std::string &outputPath);