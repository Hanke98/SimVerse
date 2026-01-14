// UrdfWriter.cpp
#include "UrdfWriter.h"      
#include <tinyxml2.h>

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace dyno {

std::string joinInts(const std::vector<int>& v)
{
    // 生成空格分隔的序列："0 5 2 9 ..."
    std::string out;
    out.reserve(v.size() * 3); // 粗略预估
    for (size_t i = 0; i < v.size(); ++i)
    {
        if (i) out.push_back(' ');
        out += std::to_string(v[i]);
    }
    return out;
}

tinyxml2::XMLElement* findLinkElementByName(tinyxml2::XMLElement* robotElem, const std::string& linkName)
{
    if (!robotElem) return nullptr;

    for (tinyxml2::XMLElement* linkElem = robotElem->FirstChildElement("link");
         linkElem;
         linkElem = linkElem->NextSiblingElement("link"))
    {
        const char* nm = linkElem->Attribute("name");
        if (nm && linkName == nm)
            return linkElem;
    }
    return nullptr;
}

bool hasPatchNode(tinyxml2::XMLElement* linkElem)
{
    if (!linkElem) return false;
    tinyxml2::XMLElement* visualElem = linkElem->FirstChildElement("visual");
    if (!visualElem) return false;
    return (visualElem->FirstChildElement("patch") != nullptr);
}

// 可选：从“原始 URDF 文件”扫描并刷新 urdfInfo.links[i].hasPatch
bool RefreshHasPatchFlags(const std::string& inputUrdfPath, UrdfInformation& urdfInfo)
{
    tinyxml2::XMLDocument doc;
    const tinyxml2::XMLError err = doc.LoadFile(inputUrdfPath.c_str());
    if (err != tinyxml2::XML_SUCCESS)
        return false;

    tinyxml2::XMLElement* robotElem = doc.FirstChildElement("robot");
    if (!robotElem)
        return false;

    for (auto& link : urdfInfo.links)
    {
        tinyxml2::XMLElement* linkElem = findLinkElementByName(robotElem, link.name);
        link.hasPatch = hasPatchNode(linkElem);
    }
    return true;
}

static std::string sanitizeFilename(const std::string& s)
{
    std::string out = s;
    for (char& c : out)
    {
        // 允许字母数字、下划线、连字符；其余替换为 '_'
        const bool ok =
            (c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c == '_' || c == '-');
        if (!ok) c = '_';
    }
    return out;
}

static std::string joinPath(const std::string& dir, const std::string& file)
{
    if (dir.empty() || dir == ".") return file;
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\'))
        return dir + file;
    return dir + "/" + file;
}

static bool writePatchBin(const std::string& filepath,
                          const std::vector<int>& patchFaces,
                          const std::vector<int>& patchOffsets,
                          int version)
{
    std::ofstream os(filepath, std::ios::binary);
    if (!os.is_open()) return false;

    const uint32_t magic = 0x48435450u; // 'P''T''C''H' little-endian
    const uint32_t ver = static_cast<uint32_t>(version);
    const uint32_t numFaces = static_cast<uint32_t>(patchFaces.size());
    const uint32_t numOffsets = static_cast<uint32_t>(patchOffsets.size());

    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    os.write(reinterpret_cast<const char*>(&numFaces), sizeof(numFaces));
    os.write(reinterpret_cast<const char*>(&numOffsets), sizeof(numOffsets));

    if (!patchOffsets.empty())
        os.write(reinterpret_cast<const char*>(patchOffsets.data()), sizeof(int) * patchOffsets.size());
    if (!patchFaces.empty())
        os.write(reinterpret_cast<const char*>(patchFaces.data()), sizeof(int) * patchFaces.size());

    return os.good();
}

// 主函数：复制 inputUrdfPath 的 XML，并注入 <patch>，保存为 outputUrdfPath
bool WriteUrdfWithPatches(const std::string& inputUrdfPath,
                          const std::string& outputUrdfPath,
                          const UrdfInformation& urdfInfo,
                          const PatchWriteOptions& opt)
{
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError err = doc.LoadFile(inputUrdfPath.c_str());
    if (err != tinyxml2::XML_SUCCESS)
        return false;

    tinyxml2::XMLElement* robotElem = doc.FirstChildElement("robot");
    if (!robotElem)
        return false;

    for (const auto& link : urdfInfo.links)
    {
        // 是否有可写入的数据
        const bool hasData = (!link.patchOffsets.empty() && !link.patchFaces.empty());
        if (!hasData)
            continue;

        // “只在缺失时写入”
        if (opt.writeMissingOnly && link.hasPatch)
            continue;

        tinyxml2::XMLElement* linkElem = findLinkElementByName(robotElem, link.name);
        if (!linkElem)
            continue;

        tinyxml2::XMLElement* visualElem = linkElem->FirstChildElement("visual");
        if (!visualElem)
        {
            continue;
        }

        // 处理已有 patch
        if (tinyxml2::XMLElement* existingPatch = visualElem->FirstChildElement("patch"))
        {
            if (!opt.overwriteExistingPatch)
                continue;
            visualElem->DeleteChild(existingPatch);
        }

        // patchOffsets 基本校验
        if (link.patchOffsets.size() < 2)
            continue;
        if (link.patchOffsets.front() != 0)
            continue;

        const int numFaces = static_cast<int>(link.patchFaces.size());
        if (link.patchOffsets.back() != numFaces)
            continue;

        const int numPatches = static_cast<int>(link.patchOffsets.size()) - 1;

        // 写出二进制 patch 文件
        const std::string binBaseName = sanitizeFilename(link.name) + ".bin";
        const std::string binFullPath = joinPath(opt.patchBinDir, binBaseName);

        if (!writePatchBin(binFullPath, link.patchFaces, link.patchOffsets, opt.version))
        {
            std::cout << "Failed to write patch file: " << binFullPath << std::endl;
            continue;
        }

        // 构造 <patch>
        tinyxml2::XMLElement* patchElem = doc.NewElement("patch");
        patchElem->SetAttribute("method", opt.method.c_str());
        patchElem->SetAttribute("version", opt.version);
        patchElem->SetAttribute("numPatches", numPatches);
        patchElem->SetAttribute("numFaces", numFaces);

        // // <faceID ID="..."/>
        // {
        //     tinyxml2::XMLElement* faceElem = doc.NewElement("faceID");
        //     const std::string faceStr = joinInts(link.patchFaces);
        //     faceElem->SetAttribute("ID", faceStr.c_str());
        //     patchElem->InsertEndChild(faceElem);
        // }

        // // <offset prefixsum="..."/>
        // {
        //     tinyxml2::XMLElement* offElem = doc.NewElement("offset");
        //     const std::string offStr = joinInts(link.patchOffsets);
        //     offElem->SetAttribute("prefixsum", offStr.c_str());
        //     patchElem->InsertEndChild(offElem);
        // }

        // <information filename="..."/>
        {
            tinyxml2::XMLElement* infoElem = doc.NewElement("information");

            // URDF 中写 basename 或写相对/绝对路径，由 opt 控制
            if (opt.urdfUseBasenameOnly)
                infoElem->SetAttribute("filename", binBaseName.c_str());
            else
                infoElem->SetAttribute("filename", binFullPath.c_str());

            patchElem->InsertEndChild(infoElem);
        }

        // 插入位置：尽量放在 <geometry> 后（纯粹是美观/一致性）
        if (tinyxml2::XMLElement* geomElem = visualElem->FirstChildElement("geometry"))
        {
            if (geomElem->NextSiblingElement())
                visualElem->InsertAfterChild(geomElem, patchElem);
            else
                visualElem->InsertEndChild(patchElem);
        }
        else
        {
            visualElem->InsertEndChild(patchElem);
        }
    }

    err = doc.SaveFile(outputUrdfPath.c_str());
    return (err == tinyxml2::XML_SUCCESS);
}

} // namespace dyno