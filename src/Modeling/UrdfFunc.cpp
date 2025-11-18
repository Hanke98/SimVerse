// UrdfFunc.cpp
#include "tinyxml/tinyxml2.h"
#include "tinyobjloader/tiny_obj_loader.h"
#include "tinygltf/tiny_gltf.h"
#include "../Core/Platform.h"

#include <iostream>
#include <unordered_map>
#include <map>
#include <cmath>
#include <vector>
#include <string>
#include <set>
#include <array>
#include <functional>
#include <cstring>

// ----------------- 类型别名 -----------------
using Mat16 = std::array<double,16>;

static Mat16 identityMatrixRowMajor()
{
    Mat16 M{};
    for (int i=0;i<16;i++) M[i]=0.0;
    M[0]=M[5]=M[10]=M[15]=1.0;
    return M;
}

static Mat16 zUpToYUp()
{
    Mat16 M = identityMatrixRowMajor();
    M[0] = 0.0;   M[1] = 1.0;
    M[5] = 0.0;   M[6] = 1.0;
    M[8] = 1.0;   M[10] = 0.0;
    return M;
}

static Mat16 transposeRowMajor(const Mat16 &M)
{
    Mat16 T{};
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
            T[r*4 + c] = M[c*4 + r];
    return T;
}

// rpy -> quaternion (x,y,z,w)
static inline std::array<double,4> rpyToQuat(double roll, double pitch, double yaw)
{
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);

    return { sr*cp*cy - cr*sp*sy,
             cr*sp*cy + sr*cp*sy,
             cr*cp*sy - sr*sp*cy,
             cr*cp*cy + sr*sp*sy };
}

// TRS -> 4x4 row-major 矩阵
static inline void makeMatrixFromTRS_rowMajor(const std::array<double,3> &t,
                                              const std::array<double,4> &q,
                                              Mat16 &out)
{
    double x=q[0], y=q[1], z=q[2], w=q[3];
    double xx=x*x, yy=y*y, zz=z*z;
    double xy=x*y, xz=x*z, yz=y*z;
    double wx=w*x, wy=w*y, wz=w*z;

    // out[0]  = 1.0 - 2.0*(yy+zz); out[1]  = 2.0*(xy + wz); out[2]  = 2.0*(xz - wy); out[3]  = 0.0;
    // out[4]  = 2.0*(xy - wz); out[5]  = 1.0 - 2.0*(xx+zz); out[6]  = 2.0*(yz + wx); out[7]  = 0.0;
    // out[8]  = 2.0*(xz + wy); out[9]  = 2.0*(yz - wx); out[10] = 1.0 - 2.0*(xx+yy); out[11] = 0.0;
    // out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1.0; // need to be Transposed

    // y' axis = z axis, z' axis = -y axis

    // double x=q[0], y=q[2], z=-q[1], w=q[3];
    // double xx=x*x, yy=y*y, zz=z*z;
    // double xy=x*y, xz=x*z, yz=y*z;
    // double wx=w*x, wy=w*y, wz=w*z;
    //
    out[0]  = 1.0 - 2.0*(yy+zz); out[1]  = 2.0*(xy - wz); out[2]  = 2.0*(xz + wy); out[3]  = t[0];
    out[4]  = 2.0*(xy + wz); out[5]  = 1.0 - 2.0*(xx+zz); out[6]  = 2.0*(yz - wx); out[7]  = t[1];
    out[8]  = 2.0*(xz - wy); out[9]  = 2.0*(yz + wx); out[10] = 1.0 - 2.0*(xx+yy); out[11] = t[2];
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1.0;
}

static Mat16 multiplyRowMajor(const Mat16 &A, const Mat16 &B)
{
    Mat16 M{};
    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
        {
            M[r*4 + c] = 0.0;
            for (int k=0; k<4; k++)
                M[r*4 + c] += A[r*4 + k] * B[k*4 + c];
        }
    return M;
}

// 把 row-major 写成 glTF 要求的 column-major 向量
static inline void setNodeMatrixColumnMajor(tinygltf::Node &node, const Mat16 &rowMajor)
{
    node.matrix.clear();
    for (int c=0;c<4;c++)
        for (int r=0;r<4;r++)
            node.matrix.push_back(rowMajor[r*4 + c]);

    node.translation.clear();
    node.rotation.clear();
    node.scale.clear();
}

// parse <origin xyz="..." rpy="...">
static inline void parseOrigin(tinyxml2::XMLElement *originElem,
                               std::array<double,3> &out_xyz,
                               std::array<double,4> &out_quat)
{
    out_xyz = {0.0,0.0,0.0};
    out_quat = {0.0,0.0,0.0,1.0};
    if (!originElem) return;

    const char *xyzAttr = originElem->Attribute("xyz");
    const char *rpyAttr = originElem->Attribute("rpy");
    if (xyzAttr) sscanf(xyzAttr, "%lf %lf %lf", &out_xyz[0], &out_xyz[1], &out_xyz[2]);
    if (rpyAttr) {
        double r, p, y;
        sscanf(rpyAttr, "%lf %lf %lf", &r, &p, &y);
        out_quat = rpyToQuat(r, p, y);
    }
}

static inline void parseOrigin_obj(tinyxml2::XMLElement *originElem,
                               std::array<double,3> &out_xyz,
                               std::array<double,4> &out_quat)
{
    out_xyz = {0.0,0.0,0.0};
    out_quat = {0.0,0.0,0.0,1.0};
    if (!originElem) return;

    const char *xyzAttr = originElem->Attribute("xyz");
    const char *rpyAttr = originElem->Attribute("rpy");
    if (xyzAttr) sscanf(xyzAttr, "%lf %lf %lf", &out_xyz[0], &out_xyz[1], &out_xyz[2]);
    if (rpyAttr) {
        double r, p, y;
        sscanf(rpyAttr, "%lf %lf %lf", &r, &p, &y);
        out_quat = rpyToQuat(r, p, y);
    }
}

// addAccessor (与之前实现兼容)
template <typename T>
int addAccessor(tinygltf::Model &model, const std::vector<T> &data,
                int type, int componentType, int elemSize)
{
    if (data.empty()) return -1;
    std::vector<unsigned char> bytes(sizeof(T) * data.size());
    memcpy(bytes.data(), data.data(), sizeof(T) * data.size());

    if (model.buffers.empty())
    {
        tinygltf::Buffer buffer;
        buffer.name = "buffer0";
        model.buffers.push_back(buffer);
    }
    int bufferIndex = 0;
    size_t offset = model.buffers[bufferIndex].data.size();
    model.buffers[bufferIndex].data.insert(model.buffers[bufferIndex].data.end(), bytes.begin(), bytes.end());

    tinygltf::BufferView bufferView;
    bufferView.buffer = bufferIndex;
    bufferView.byteOffset = offset;
    bufferView.byteLength = bytes.size();
    int bufferViewIndex = model.bufferViews.size();
    model.bufferViews.push_back(bufferView);

    tinygltf::Accessor accessor;
    accessor.bufferView = bufferViewIndex;
    accessor.byteOffset = 0;
    accessor.componentType = componentType;
    accessor.count = data.size() / elemSize;
    accessor.type = type;

    int accessorIndex = model.accessors.size();
    model.accessors.push_back(accessor);
    return accessorIndex;
}

// ----------------- 主函数：URDF -> glTF -----------------
bool urdfToGltf(const std::string &urdfPath, const std::string &outputPath)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(urdfPath.c_str()) != tinyxml2::XML_SUCCESS)
    {
        std::cerr << "Failed to load URDF file: " << urdfPath << std::endl;
        return false;
    }

    tinygltf::Model model;
    model.scenes.resize(1);
    model.defaultScene = 0;
    model.asset.version = "2.0";
    model.asset.generator = "urdfToGltf converter";

    tinyxml2::XMLElement *robotElem = doc.FirstChildElement("robot");
    if (!robotElem) { std::cerr << "No <robot> element in URDF." << std::endl; return false; }

    std::unordered_map<std::string,int> linkNodeMap;
    std::map<std::string, Mat16> jointLocalMatrix;
    std::map<std::string, std::string> parentOf;
    std::map<std::string, std::vector<std::string>> childrenOf;
    std::map<int, std::vector<int>> linkToMeshNodes;
    std::map<int, Mat16> meshLocalMatrix;
    std::set<std::string> allLinkNames;

    // 1) 创建所有 link 的 node
    for (auto link = robotElem->FirstChildElement("link"); link; link = link->NextSiblingElement("link"))
    {
        const char *linkName = link->Attribute("name");
        if (!linkName) continue;
        tinygltf::Node ln;
        ln.name = linkName;
        int idx = model.nodes.size();
        model.nodes.push_back(ln);
        linkNodeMap[linkName] = idx;
        allLinkNames.insert(linkName);
    }

    // 2) joints
    for (auto joint = robotElem->FirstChildElement("joint"); joint; joint = joint->NextSiblingElement("joint"))
    {
        auto parentElem = joint->FirstChildElement("parent");
        auto childElem  = joint->FirstChildElement("child");
        if (!parentElem || !childElem) continue;
        const char *parentName = parentElem->Attribute("link");
        const char *childName  = childElem->Attribute("link");
        if (!parentName || !childName) continue;

        std::array<double,3> jt; std::array<double,4> jq;
        parseOrigin(joint->FirstChildElement("origin"), jt, jq);
        Mat16 local; makeMatrixFromTRS_rowMajor(jt, jq, local);

        jointLocalMatrix[childName] = local;
        parentOf[childName] = parentName;
        childrenOf[parentName].push_back(childName);
    }

    // 3) visual
    for (auto link = robotElem->FirstChildElement("link"); link; link = link->NextSiblingElement("link"))
    {
        const char *linkName = link->Attribute("name");
        if (!linkName) continue;
        int linkIdx = linkNodeMap[linkName];

        for (auto visual = link->FirstChildElement("visual"); visual; visual = visual->NextSiblingElement("visual"))
        {
            auto geometry = visual->FirstChildElement("geometry");
            if (!geometry) continue;
            auto mesh = geometry->FirstChildElement("mesh");
            if (!mesh) continue;
            const char *filename = mesh->Attribute("filename");
            if (!filename) continue;

            std::string meshPath = filename;
            if (meshPath.find("package://") == 0) meshPath = meshPath.substr(10);
            meshPath = getAssetPath() + "/../asset/" + meshPath;
            // meshPath = "/home/wjv/SimVerse_01/SimVerse/asset/" + meshPath;
            if (meshPath.size() > 4 && meshPath.substr(meshPath.size()-4) == ".dae")
                meshPath = meshPath.substr(0, meshPath.size()-4) + ".obj";

            // std::cout << "meshPath: " << meshPath << std::endl;

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;
            std::string baseDir;
            auto pos = meshPath.find_last_of('/');
            if (pos != std::string::npos) baseDir = meshPath.substr(0, pos+1);

            bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str(), baseDir.c_str(), true);
            if (!ok)
            {
                std::cerr << "Failed to load OBJ: " << meshPath << "  warn=" << warn << " err=" << err << std::endl;
                continue;
            }

            tinygltf::Mesh gmesh;
            gmesh.name = std::string(linkName) + "_mesh";

            for (size_t s = 0; s < shapes.size(); ++s)
            {
                tinygltf::Primitive prim;
                prim.mode = TINYGLTF_MODE_TRIANGLES;

                std::vector<float> positions;
                std::vector<float> normals;
                std::vector<unsigned short> indices;

                for (size_t f = 0; f < shapes[s].mesh.indices.size(); ++f)
                {
                    auto idx = shapes[s].mesh.indices[f];
                    indices.push_back((unsigned short)f);

                    positions.push_back(attrib.vertices[3*idx.vertex_index + 0]);
                    positions.push_back(attrib.vertices[3*idx.vertex_index + 1]);
                    positions.push_back(attrib.vertices[3*idx.vertex_index + 2]);

                    if (idx.normal_index >= 0)
                    {
                        normals.push_back(attrib.normals[3*idx.normal_index + 0]);
                        normals.push_back(attrib.normals[3*idx.normal_index + 1]);
                        normals.push_back(attrib.normals[3*idx.normal_index + 2]);
                    }
                }

                int posAcc = addAccessor(model, positions, TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, 3);
                int norAcc = addAccessor(model, normals, TINYGLTF_TYPE_VEC3, TINYGLTF_COMPONENT_TYPE_FLOAT, 3);
                int idxAcc = addAccessor(model, indices, TINYGLTF_TYPE_SCALAR, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, 1);

                prim.attributes["POSITION"] = posAcc;
                if (norAcc >= 0) prim.attributes["NORMAL"] = norAcc;
                prim.indices = idxAcc;
                gmesh.primitives.push_back(prim);
            }

            int gmeshIndex = model.meshes.size();
            model.meshes.push_back(gmesh);

            tinygltf::Node meshNode;
            meshNode.name = std::string(linkName) + "_visual";
            meshNode.mesh = gmeshIndex;

            std::array<double,3> mt; std::array<double,4> mq;
            parseOrigin(visual->FirstChildElement("origin"), mt, mq);

            // 1) 先根据 URDF origin 做一个变换矩阵（link -> visual 原本的定义）
            Mat16 originM;
            makeMatrixFromTRS_rowMajor(mt, mq, originM);

            // 2) 再做一个把 Y-up 的 mesh 旋到 Z-up 的固定旋转（绕 X 轴 +90°）
            std::array<double,3> fixT{0.0, 0.0, 0.0};
            std::array<double,4> fixQ = rpyToQuat(M_PI * 0.5, 0.0, 0.0);
            // std::array<double,4> fixQ{0.5, 0.5, 0.5, 0.5};

            Mat16 fixM;
            makeMatrixFromTRS_rowMajor(fixT, fixQ, fixM);
            // std::cout << "Matrix: " << fixM[0] << " "<< fixM[1]  << " " << fixM[2] << "\n" <<
            //     fixM[4] << " "<< fixM[5]  << " " << fixM[6] << "\n" <<
            //         fixM[8] << " "<< fixM[9]  << " " << fixM[10] << "\n" << std::endl;

            // 3) 最终的 meshLocal：先做 URDF 的 origin，再做 Y-up -> Z-up 旋转
               // 列向量约定：p_world = linkWorld * originM * fixM * p_mesh
            Mat16 meshLocal = multiplyRowMajor(originM, fixM);

            // Mat16 meshLocal; makeMatrixFromTRS_rowMajor(mt, mq, meshLocal);

            int meshNodeIndex = model.nodes.size();
            model.nodes.push_back(meshNode);

            linkToMeshNodes[linkIdx].push_back(meshNodeIndex);
            meshLocalMatrix[meshNodeIndex] = meshLocal;
        }
    }

    // 4) find roots
    std::vector<std::string> roots;
    for (const auto &ln : allLinkNames)
    {
        if (parentOf.find(ln) == parentOf.end())
            roots.push_back(ln);
    }
    if (roots.empty() && !allLinkNames.empty())
        roots.push_back(*allLinkNames.begin());

    // 5) DFS 组织层级关系
    std::function<void(const std::string&, const Mat16&)> dfs;
    dfs = [&](const std::string &linkName, const Mat16 &parentWorld)
    {
        int linkIdx = linkNodeMap[linkName];
        Mat16 linkLocal = identityMatrixRowMajor();

        // auto itJointLocal = jointLocalMatrix.find(linkName);
        // if (itJointLocal != jointLocalMatrix.end())
        //     linkLocal = itJointLocal->second;

        // // 累乘出当前 link 的世界矩阵
        // Mat16 linkWorld = multiplyRowMajor(parentWorld, linkLocal);
        Mat16 linkWorld = parentWorld;
        auto itJointLocal = jointLocalMatrix.find(linkName);
        if (itJointLocal != jointLocalMatrix.end())
            linkWorld = multiplyRowMajor(parentWorld, itJointLocal->second);

        setNodeMatrixColumnMajor(model.nodes[linkIdx], linkWorld);

        // 打印矩阵
        // std::cout << "Link: " << linkName << "\nWorldMatrix:";
        // for (int i=0;i<16;i++){ if (i%4==0) std::cout<<"\n"; std::cout<<linkWorld[i]<<" "; }
        // std::cout << "\n";

        // 子 mesh
        auto itMeshes = linkToMeshNodes.find(linkIdx);
        if (itMeshes != linkToMeshNodes.end())
        {
            for (int meshNodeIdx : itMeshes->second)
            {
                Mat16 meshLocal = meshLocalMatrix[meshNodeIdx];
                Mat16 meshWorld = multiplyRowMajor(linkWorld, meshLocal);
                setNodeMatrixColumnMajor(model.nodes[meshNodeIdx], meshWorld);

                // std::cout << "  MeshNodeIdx: " << meshNodeIdx << " (" << model.nodes[meshNodeIdx].name << ")";
                // std::cout << "\n  MeshWorldMatrix:";
                // for (int i=0;i<16;i++){ if (i%4==0) std::cout<<"\n  "; std::cout<<meshWorld[i]<<" "; }
                // std::cout << "\n";

                model.nodes[linkIdx].children.push_back(meshNodeIdx);
            }
        }

        // 递归子 link
        auto itChildren = childrenOf.find(linkName);
        if (itChildren != childrenOf.end())
        {
            for (const std::string &childName : itChildren->second)
            {
                model.nodes[linkIdx].children.push_back(linkNodeMap[childName]);
                dfs(childName, linkWorld);
            }
        }
    };
    for (const std::string &root : roots)
    {
        // root link 可附加一个 Z->Y 转换
        dfs(root, zUpToYUp());
        // dfs(root, identityMatrixRowMajor());
        model.scenes[0].nodes.push_back(linkNodeMap[root]);
    }

    tinygltf::TinyGLTF ctx;
    if (!ctx.WriteGltfSceneToFile(&model, outputPath, true, true, true, false))
    {
        std::cerr << "Failed to write glTF file: " << outputPath << std::endl;
        return false;
    }

    return true;
}
