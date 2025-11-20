#include "UrdfParser.h"
#include "UrdfFunc.h"
#include "ImageLoader.h"
#include "GltfFunc.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <filesystem>

namespace dyno
{

bool loadURDFTextureMesh(std::shared_ptr<TextureMesh> texMesh,
                         const FilePath& urdfFullPath)
{
    // 解析 URDF
    UrdfParser parser;

    // URDF 所在目录，用来拼 mesh 的相对路径
    auto urdfPath  = urdfFullPath;
    auto urdfRoot  = urdfPath.path().parent_path();

    if (!parser.parse(urdfPath.string().c_str()))
    {
        std::cerr << "Failed to parse URDF: " << urdfPath.string().c_str() << std::endl;
        return false;
    }

    auto& links = parser.links;

    // 清空 texMesh 里旧的数据
    texMesh->clear();

    // 准备全局 CPU 端容器（最后一次性 assign 到 DArray）
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> texCoords;
    std::vector<uint>  shapeIds;

    auto& reShapes = texMesh->shapes();
    auto& reMats   = texMesh->materials();

    reShapes.clear();
    reMats.clear();

    uint globalShapeId = 0;

    // 遍历每一个 link，只处理 visualMeshPath
    for (const auto& link : links) {
        if (link.visualMeshPath.empty())
            continue;

        // 拼出这个 link 的 obj 完整路径
        auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.visualMeshPath);
        std::string meshFile   = meshFull.string();
        std::string meshFolder = meshFull.path().parent_path().string();

        // 用 tinyobj 加载这个 obj
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(
            &attrib,
            &shapes,
            &materials,
            &warn,
            &err,
            meshFile.c_str(),
            meshFolder.c_str()
        );

        if (!warn.empty())
            std::cerr << "tinyobj warn: " << warn << std::endl;
        if (!err.empty())
        {
            std::cerr << "tinyobj err: " << err << std::endl;
            continue;
        }
        if (!ret)
        {
            std::cerr << "Failed to load obj: " << meshFile << std::endl;
            continue;
        }

        // 记录当前全局顶点 / 法线 / UV 的起始 offset
        size_t vOffset = vertices.size();
        size_t nOffset = normals.size();
        size_t tOffset = texCoords.size();

        bool hasNormals   = !attrib.normals.empty();
        bool hasTexcoords = !attrib.texcoords.empty();

        // 把这个 obj 的 attrib 数据追加到全局容器
        for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        {
            vertices.push_back(Vec3f(
                attrib.vertices[i + 0],
                attrib.vertices[i + 1],
                attrib.vertices[i + 2]
            ));
        }
        if (hasNormals) {
            for (size_t i = 0; i < attrib.normals.size(); i += 3)
            {
                normals.push_back(Vec3f(
                    attrib.normals[i + 0],
                    attrib.normals[i + 1],
                    attrib.normals[i + 2]
                ));
            }
        }

        if (hasTexcoords) {
            for (size_t i = 0; i < attrib.texcoords.size(); i += 2)
            {
                texCoords.push_back(Vec2f(
                    attrib.texcoords[i + 0],
                    attrib.texcoords[i + 1]
                ));
            }
        }

        if (!hasTexcoords)
        {
            // 确保 texCoords 至少有与 vertices 一样多的元素
            if (texCoords.size() < vertices.size())
            {
                texCoords.resize(vertices.size());
            }

            // 为这一段 [vOffset, vertices.size()) 的顶点设置默认 UV
            for (size_t vi = vOffset; vi < vertices.size(); ++vi)
            {
                texCoords[vi] = Vec2f(0.0f, 0.0f);
            }
        }

        // 形状 ID 要覆盖到新的全部顶点长度
        shapeIds.resize(vertices.size());

        // 先把 tinyobj 的材质转成引擎的 Material，追加到 reMats
        uint matOffset = static_cast<uint>(reMats.size());
        reMats.resize(reMats.size() + materials.size());

        dyno::CArray2D<dyno::Vec4f> texture(1, 1);
        texture[0, 0] = dyno::Vec4f(1);

        for (size_t mId = 0; mId < materials.size(); ++mId)
        {
            const auto& mtl = materials[mId];
            reMats[matOffset + mId] = std::make_shared<Material>();
            auto& mat = reMats[matOffset + mId];

            mat->baseColor = Vec3f(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);

            std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();

            // diffuse 纹理
            if (!mtl.diffuse_texname.empty())
            {
                std::cout << "Loading diffuse texture: " << mtl.diffuse_texname << std::endl;
                auto tex_path = (urdfRoot / mtl.diffuse_texname).string();
                if (loader->loadImage(tex_path.c_str(), texture))
                {
                    mat->texColor.assign(texture);
                }
            }

            // bump / normal 贴图
            if (!mtl.bump_texname.empty())
            {
                std::cout << "Loading bump texture: " << mtl.bump_texname << std::endl;
                auto tex_path = (urdfRoot/ mtl.bump_texname).string();
                if (loader->loadImage(tex_path.c_str(), texture))
                {
                    mat->texBump.assign(texture);
                    auto texOpt = mtl.bump_texopt;
                    mat->bumpScale = texOpt.bump_multiplier;
                }
            }
        }

        // 4.6 为这个 obj 里的每一个 tinyobj::shape_t 创建一个 Shape
        for (const auto& tshape : shapes)
        {
            const auto& mesh = tshape.mesh;

            auto shape = std::make_shared<Shape>();

            std::vector<TopologyModule::Triangle> vertexIndex;
            std::vector<TopologyModule::Triangle> normalIndex;
            std::vector<TopologyModule::Triangle> texCoordIndex;

            // 绑定材质（tinyobj 每个 shape 可以有 material_ids）
            if (!mesh.material_ids.empty() && mesh.material_ids[0] >= 0)
            {
                int localMatId  = mesh.material_ids[0];

                int globalMatId = static_cast<int>(matOffset) + localMatId;

                if (globalMatId >= 0 && globalMatId < static_cast<int>(reMats.size()))
                    shape->material = reMats[globalMatId];
            }

            Vec3f lo( REAL_MAX);
            Vec3f hi(-REAL_MAX);

            // tinyobj 里 indices 是三角形列表（每个 index 里有 v / n / t 下标）
            for (size_t i = 0; i < mesh.indices.size(); i += 3)
            {
                auto idx0 = mesh.indices[i + 0];
                auto idx1 = mesh.indices[i + 1];
                auto idx2 = mesh.indices[i + 2];

                // 加上 offset，把局部下标变成全局下标
                int v0 = idx0.vertex_index + static_cast<int>(vOffset);
                int v1 = idx1.vertex_index + static_cast<int>(vOffset);
                int v2 = idx2.vertex_index + static_cast<int>(vOffset);

                TopologyModule::Triangle tri(v0, v1, v2);

                vertexIndex.push_back(tri);

                if (hasNormals && idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
                    int n0 = (idx0.normal_index  >= 0) ? idx0.normal_index  + static_cast<int>(nOffset) : -1;
                    int n1 = (idx1.normal_index  >= 0) ? idx1.normal_index  + static_cast<int>(nOffset) : -1;
                    int n2 = (idx2.normal_index  >= 0) ? idx2.normal_index  + static_cast<int>(nOffset) : -1;
                    normalIndex.push_back(TopologyModule::Triangle(n0, n1, n2));
                } else {
                    normalIndex.push_back(tri);
                }

                if (hasTexcoords && idx0.texcoord_index >= 0 && idx1.texcoord_index >= 0 && idx2.texcoord_index >= 0) {
                    int t0 = (idx0.texcoord_index >= 0) ? idx0.texcoord_index + static_cast<int>(tOffset) : -1;
                    int t1 = (idx1.texcoord_index >= 0) ? idx1.texcoord_index + static_cast<int>(tOffset) : -1;
                    int t2 = (idx2.texcoord_index >= 0) ? idx2.texcoord_index + static_cast<int>(tOffset) : -1;
                    texCoordIndex.push_back(TopologyModule::Triangle(t0, t1, t2));
                } else {
                    texCoordIndex.push_back(tri);
                }

                // 更新包围盒
                lo = lo.minimum(vertices[v0]);
                lo = lo.minimum(vertices[v1]);
                lo = lo.minimum(vertices[v2]);

                hi = hi.maximum(vertices[v0]);
                hi = hi.maximum(vertices[v1]);
                hi = hi.maximum(vertices[v2]);

                // 填 shapeIds：把这几个顶点标记为当前 globalShapeId
                shapeIds[v0] = globalShapeId;
                shapeIds[v1] = globalShapeId;
                shapeIds[v2] = globalShapeId;
            }

            shape->vertexIndex.assign(vertexIndex);
            shape->normalIndex.assign(normalIndex);
            shape->texCoordIndex.assign(texCoordIndex);

            // 包围盒与中心
            auto shapeCenter = (lo + hi) * Real(0.5);
            shape->boundingBox       = TAlignedBox3D<Real>(lo, hi);
            shape->boundingTransform = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));

            reShapes.push_back(shape);
            globalShapeId++;
        }


        // 应用 URDF 的 <visual><origin> 变换到属于这个 link 的顶点
        // 也就是把 [vOffset, vertices.size()) 这一段的顶点乘以 T_world*link.meshTransform
        // p_world = T_world * link.meshTransform * p_mesh
        Transform3f T_world_mesh = composeTransform(link.T_world, link.meshTransform);
        auto R = T_world_mesh.rotation();
        for (size_t i = vOffset; i < vertices.size(); ++i)
        {
            vertices[i] = T_world_mesh * vertices[i];
        }
        if (hasNormals)
        {
            for (size_t i = nOffset; i < normals.size(); ++i)
            {
                normals[i] = R * normals[i];
                Real len = normals[i].norm();
                if (len > Real(1e-8)) normals[i] /= len;
            }
        }
    }

    // 全部 link 处理完毕，一次性把 std::vector 拷到 DArray
    texMesh->vertices().assign(vertices);
    texMesh->normals().assign(normals);
    texMesh->texCoords().assign(texCoords);
    texMesh->shapeIds().assign(shapeIds);

    auto shapeNum = texMesh->shapes().size();

    CArray<Vec3f> c_shapeCenter;
    c_shapeCenter.resize(shapeNum);
    //counter
    for (uint i = 0; i < shapeNum; i++)
    {
        DArray<int> counter;
        counter.resize(texMesh->vertices().size());

        Shape_PointCounter(counter,
            texMesh->shapeIds(),
            i);

        Reduction<int> reduce;
        int num = reduce.accumulate(counter.begin(), counter.size());

        DArray<Vec3f> targetPoints;
        targetPoints.resize(num);

        Scan<int> scan;
        scan.exclusive(counter.begin(), counter.size());

        setupPoints(
            targetPoints,
            texMesh->vertices(),
            counter
        );

        Reduction<Vec3f> reduceBounding;

        auto& bounding = texMesh->shapes()[i]->boundingBox;
        Vec3f lo = reduceBounding.minimum(targetPoints.begin(), targetPoints.size());
        Vec3f hi = reduceBounding.maximum(targetPoints.begin(), targetPoints.size());

        bounding.v0 = lo;
        bounding.v1 = hi;
        texMesh->shapes()[i]->boundingTransform.translation() = (lo + hi) / 2;

        c_shapeCenter[i] = (lo + hi) / 2;

        targetPoints.clear();

        counter.clear();
    }

    DArray<Vec3f> d_ShapeCenter;
    DArray<Vec3f> unCenterPosition;

    d_ShapeCenter.assign(c_shapeCenter);	// Used to "ToCenter"
    unCenterPosition.assign(texMesh->vertices());

    //ToCenter
    if (true)//varUseInstanceTransform()->getValue()
    {
        shapeToCenter(unCenterPosition,
            texMesh->vertices(),
            texMesh->shapeIds(),
            d_ShapeCenter);


        auto& reShapes = texMesh->shapes();

        for (size_t i = 0; i < shapeNum; i++)
        {
            reShapes[i]->boundingTransform.translation() = reShapes[i]->boundingTransform.translation() ;//+ this->varLocation()->getValue()
        }
    }
    else
    {
        auto& reShapes = texMesh->shapes();

        for (size_t i = 0; i < shapeNum; i++)
        {
            reShapes[i]->boundingTransform.translation() = Vec3f(0);
        }
    }

    return true;
}

} // namespace dyno