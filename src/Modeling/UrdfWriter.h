#ifndef URDFWRITER_H
#define URDFWRITER_H

#include "UrdfParser.h"  // For UrdfInformation
#include <tinyxml2.h>
#include <string>
#include <vector>

namespace dyno {

struct PatchWriteOptions
{
    // if the <patch> node already exists in the link:
    // - false: do nothing, skip writing
    // - true : delete existing <patch> and write new one
    bool overwriteExistingPatch = false;

    // true : 仅当 link.hasPatch == false 时才写入（“缺失才补写”）
    // false: 只要 patchFaces/patchOffsets 有数据就写入
    bool writeMissingOnly = true;

    // 写到 <patch> 的元信息（可用于调试/版本管理）
    std::string method = "kmeans";
    int version = 1;

    // bin 输出目录（例如 "patched_bins" 或与 urdf 同目录）
    std::string patchBinDir = "/home/wjv/SimVerse_01/SimVerse/data/../asset/franka_description/robots/";

    // URDF 中写入的 filename 是否只写 basename
    bool urdfUseBasenameOnly = false;
};

std::string joinInts(const std::vector<int>& v);

tinyxml2::XMLElement* findLinkElementByName(tinyxml2::XMLElement* robotElem, const std::string& linkName);

bool hasPatchNode(tinyxml2::XMLElement* linkElem);

// 可选：从“原始 URDF 文件”扫描并刷新 urdfInfo.links[i].hasPatch
bool RefreshHasPatchFlags(const std::string& inputUrdfPath, UrdfInformation& urdfInfo);

// 主函数：复制 inputUrdfPath 的 XML，并注入 <patch>，保存为 outputUrdfPath
bool WriteUrdfWithPatches(const std::string& inputUrdfPath,
                          const std::string& outputUrdfPath,
                          const UrdfInformation& urdfInfo,
                          const PatchWriteOptions& opt = PatchWriteOptions());

} // namespace dyno

#endif // URDFWRITER_H