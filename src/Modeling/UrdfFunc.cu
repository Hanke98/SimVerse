#include "UrdfFunc.h"
#include "ImageLoader.h"
#include "GltfFunc.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <filesystem>

namespace dyno
{
// ---------------------------------------------------------
// Helper: Compute Volume and Inertia Tensor (Density = 1.0)
// ---------------------------------------------------------
void computeMeshPhysicalProperties(
    const std::vector<Vec3f>& vertices,
    const std::vector<TopologyModule::Triangle>& indices,
    Real& outVolume,
    Vec3f& outCenterOfMass,
    Mat3f& outInertiaTensor)
{
    double integral[10] = {0.0}; // 1, x, y, z, x^2, y^2, z^2, xy, yz, zx

    // 预计算乘法因子
    const double f1 = 1.0 / 60.0;
    const double f2 = 1.0 / 120.0;

    for (const auto& face : indices)
    {
        // 获取三角形顶点
        Vec3f v0 = vertices[face[0]];
        Vec3f v1 = vertices[face[1]];
        Vec3f v2 = vertices[face[2]];

        // 计算相对于原点的四面体有向体积的6倍 (detJ)
        // detJ = v0 . (v1 x v2)
        double detJ = v0.dot(v1.cross(v2));

        // 1. 体积积分 (Integral of 1)
        integral[0] += detJ;

        // 2. 一阶矩积分 (Center of Mass term) -> Integral of x, y, z
        Vec3f sum = v0 + v1 + v2;
        integral[1] += detJ * sum.x;
        integral[2] += detJ * sum.y;
        integral[3] += detJ * sum.z;

        // 3. 二阶矩积分 (Covariance terms)
        // 这里的公式基于 canonical tetrahedron integration
        // Integral(x^2)
        integral[4] += detJ * (v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + sum.x*sum.x);
        // Integral(y^2)
        integral[5] += detJ * (v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + sum.y*sum.y);
        // Integral(z^2)
        integral[6] += detJ * (v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + sum.z*sum.z);

        // Integral(xy)
        integral[7] += detJ * (v0.x*v0.y + v1.x*v1.y + v2.x*v2.y + sum.x*sum.y);
        // Integral(yz)
        integral[8] += detJ * (v0.y*v0.z + v1.y*v1.z + v2.y*v2.z + sum.y*sum.z);
        // Integral(zx)
        integral[9] += detJ * (v0.z*v0.x + v1.z*v1.x + v2.z*v2.x + sum.z*sum.x);
    }

    // --- 归一化与后处理 ---

    // 6.0 是因为 detJ 是 6*Vol
    outVolume = static_cast<Real>(integral[0] / 6.0);

    // 防止除以零（空网格）
    if (std::abs(outVolume) < Real(1e-9)) {
        outVolume = 0;
        outCenterOfMass = Vec3f(0);
        outInertiaTensor = Mat3f(0);
        return;
    }

    // 计算质心 (CoM)
    // integral[1] 是 int(x) * 24 (这里因为上面的系数没除尽，需要仔细推导)
    // 简单做法：上面的积分累加的是 detJ * sum，实际上是 24 * int(x)
    // 标准公式：Int(x) = detJ/24 * (v0+v1+v2+v3)，原点v3=0 -> sum
    // 我们累加的是 detJ * sum，所以总和除以 (24 * Volume) 得到 CoM?
    // 验证：Vol = sum(detJ)/6. Int(x) = sum(detJ * sum_x) / 24.
    // CoM = Int(x) / Vol = (sum(detJ*sum_x)/24) / (sum(detJ)/6) = sum(detJ*sum_x) / (4 * sum(detJ))

    // 更正系数：
    // Volume = integral[0] / 6.0
    // CoM = (Integral 1st Moment) / (24.0) / Volume ->
    // CoM = (integral[1..3] / 24.0) / (integral[0] / 6.0) = integral[1..3] / (4.0 * integral[0])

    double inv_4_vol = 1.0 / (4.0 * integral[0]);
    outCenterOfMass = Vec3f(
        static_cast<Real>(integral[1] * inv_4_vol),
        static_cast<Real>(integral[2] * inv_4_vol),
        static_cast<Real>(integral[3] * inv_4_vol)
    );

    // 计算相对于原点的协方差矩阵对角线与非对角线
    // 系数：Integral(x^2) = sum(...) * detJ / 120
    // 我们累加了 sum(...) * detJ，所以需要除以 120，然后除以体积得到均值？
    // 不，我们直接要积分值。

    double i_xx = integral[4] / 60.0; // Wait, formula is usually 60 or 120.
    // Mirtich formula: \int x^2 = detJ/120 * (sum(x_i^2) + sum(x)^2).
    // 我上面的代码里用的乘数是 let sum=v0+v1+v2. (v0^2 + ... + sum^2) 是标准形式的 2 倍?
    // 简便起见，使用标准系数组合： detJ / 120 * (terms) 是正确的积分值。
    // 但是为了保持精度，最后再除。

    double xx = integral[4] / 120.0;
    double yy = integral[5] / 120.0;
    double zz = integral[6] / 120.0;
    double xy = integral[7] / 120.0;
    double yz = integral[8] / 120.0;
    double zx = integral[9] / 120.0;

    // 构造相对于原点的惯量张量 (I_origin)
    // I_xx = int(y^2 + z^2) dm
    // I_xy = - int(xy) dm
    Mat3f I_origin;
    I_origin(0, 0) = static_cast<Real>(yy + zz);
    I_origin(1, 1) = static_cast<Real>(xx + zz);
    I_origin(2, 2) = static_cast<Real>(xx + yy);
    I_origin(0, 1) = I_origin(1, 0) = static_cast<Real>(-xy);
    I_origin(0, 2) = I_origin(2, 0) = static_cast<Real>(-zx);
    I_origin(1, 2) = I_origin(2, 1) = static_cast<Real>(-yz);

    // 平行轴定理：移至质心
    // I_cm = I_origin - Mass * (r^2 * Identity - r * rT)
    // 其中 r 是 CoM 向量。Mass = Volume * 1.0

    Real m = outVolume;
    Vec3f c = outCenterOfMass;
    Real c_sq = c.dot(c); // x^2+y^2+z^2

    Mat3f correction;
    correction(0, 0) = m * (c_sq - c.x * c.x);
    correction(1, 1) = m * (c_sq - c.y * c.y);
    correction(2, 2) = m * (c_sq - c.z * c.z);
    correction(0, 1) = correction(1, 0) = m * (-c.x * c.y);
    correction(0, 2) = correction(2, 0) = m * (-c.x * c.z);
    correction(1, 2) = correction(2, 1) = m * (-c.y * c.z);

    outInertiaTensor = I_origin - correction;
}

bool loadURDFTextureMesh(std::shared_ptr<TextureMesh> texMesh,
                         const FilePath& urdfFullPath,
                         UrdfInformation& urdfInfo, bool objYUp)
{
    // Parse the URDF and extract the mesh
    UrdfParser parser;

    auto urdfPath  = urdfFullPath;
    auto urdfRoot  = urdfPath.path().parent_path();

    if (!parser.parse(urdfPath.string().c_str(), urdfInfo, objYUp))
    {
        std::cerr << "Failed to parse URDF: " << urdfPath.string().c_str() << std::endl;
        return false;
    }

    auto& links = urdfInfo.links;

    texMesh->clear();

    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> texCoords;
    std::vector<uint>  shapeIds;

    auto& reShapes = texMesh->shapes();
    auto& reMats   = texMesh->materials();
    reShapes.clear();
    reMats.clear();

    uint globalShapeId = 0;

    // Iterate through each link and handle the visual mesh
    for (auto& link : links) {
        if (link.visualMeshPath.empty())
            continue;

        // Construct the complete path to the mesh
        auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.visualMeshPath);
        std::string meshFile   = meshFull.string();
        std::string meshFolder = meshFull.path().parent_path().string();

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

        size_t vOffset = vertices.size();
        size_t nOffset = normals.size();
        size_t tOffset = texCoords.size();

        bool hasNormals   = !attrib.normals.empty();
        bool hasTexcoords = !attrib.texcoords.empty();

        // Append the data from the obj file
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
        } else {
            if (texCoords.size() < vertices.size())
            {
                texCoords.resize(vertices.size());
            }

            for (size_t vi = vOffset; vi < vertices.size(); ++vi)
            {
                texCoords[vi] = Vec2f(0.0f, 0.0f);
            }
        }

        shapeIds.resize(vertices.size());

        // convert materials of tinyobj to engine, and add them into reMats
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
                auto tex_path = (urdfRoot / mtl.diffuse_texname).string();
                if (loader->loadImage(tex_path.c_str(), texture))
                {
                    mat->texColor.assign(texture);
                }
            }

            // bump / normal 贴图
            if (!mtl.bump_texname.empty())
            {
                auto tex_path = (urdfRoot/ mtl.bump_texname).string();
                if (loader->loadImage(tex_path.c_str(), texture))
                {
                    mat->texBump.assign(texture);
                    auto texOpt = mtl.bump_texopt;
                    mat->bumpScale = texOpt.bump_multiplier;
                }
            }
        }

        // Merge all shapes into a single shape
        std::shared_ptr<Shape> mergedShape = std::make_shared<Shape>();
        std::vector<TopologyModule::Triangle> vertexIndex;
        std::vector<TopologyModule::Triangle> normalIndex;
        std::vector<TopologyModule::Triangle> texCoordIndex;

        Transform3f T_world_mesh = composeTransform(link.T_world, link.meshTransform);
        Vec3f lo( REAL_MAX);
        Vec3f hi(-REAL_MAX);

        for (const auto& tshape : shapes)
        {
            const auto& mesh = tshape.mesh;

            // auto shape = std::make_shared<Shape>();

            // std::vector<TopologyModule::Triangle> vertexIndex;
            // std::vector<TopologyModule::Triangle> normalIndex;
            // std::vector<TopologyModule::Triangle> texCoordIndex;

            // 绑定材质（tinyobj 每个 shape 可以有 material_ids）
            // if (!mesh.material_ids.empty() && mesh.material_ids[0] >= 0)
            // {
            //     int localMatId  = mesh.material_ids[0];
            //
            //     int globalMatId = static_cast<int>(matOffset) + localMatId;
            //
            //     if (globalMatId >= 0 && globalMatId < static_cast<int>(reMats.size()))
            //         shape->material = reMats[globalMatId];
            // }

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
// TODO: Use oriented bounding box, and transform the bb later.
                Vec3f transformedV0 = T_world_mesh * vertices[v0];
                Vec3f transformedV1 = T_world_mesh * vertices[v1];
                Vec3f transformedV2 = T_world_mesh * vertices[v2];

                // Update the bounding box with transformed vertices
                lo = lo.minimum(transformedV0);
                lo = lo.minimum(transformedV1);
                lo = lo.minimum(transformedV2);

                hi = hi.maximum(transformedV0);
                hi = hi.maximum(transformedV1);
                hi = hi.maximum(transformedV2);

                // // 更新包围盒
                // lo = lo.minimum(vertices[v0]);
                // lo = lo.minimum(vertices[v1]);
                // lo = lo.minimum(vertices[v2]);
                //
                // hi = hi.maximum(vertices[v0]);
                // hi = hi.maximum(vertices[v1]);
                // hi = hi.maximum(vertices[v2]);

                // 填 shapeIds：把这几个顶点标记为当前 globalShapeId
                shapeIds[v0] = globalShapeId;
                shapeIds[v1] = globalShapeId;
                shapeIds[v2] = globalShapeId;
            }
        }
        mergedShape->vertexIndex.assign(vertexIndex);
        mergedShape->normalIndex.assign(normalIndex);
        mergedShape->texCoordIndex.assign(texCoordIndex);

        // 包围盒与中心
        auto shapeCenter = (lo + hi) * Real(0.5);
        mergedShape->boundingBox       = TAlignedBox3D<Real>(lo, hi);
        mergedShape->boundingTransform = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));

        reShapes.push_back(mergedShape);
        link.shapeId = globalShapeId;
        globalShapeId++;

        // p_world = T_world * link.meshTransform * p_mesh
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
        // 5+v: Perform point counter operation to assign shape ids
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
        // Vec3f lo = reduceBounding.minimum(targetPoints.begin(), targetPoints.size());
        // Vec3f hi = reduceBounding.maximum(targetPoints.begin(), targetPoints.size());

        // bounding.v0 = lo;
        // bounding.v1 = hi;
        // texMesh->shapes()[i]->boundingTransform.translation() = (lo + hi) / 2;

        Vec3f lo = bounding.v0;
        Vec3f hi = bounding.v1;

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

    // -------------------------------------------------------------------------
    // 新增：处理 Collision Mesh 并计算体积和转动惯量
    // -------------------------------------------------------------------------
    for (auto& link : links) {
        if (link.collisionMeshPath.empty()) {
            // 如果没有碰撞体，赋予默认值
            link.volume = 0;
            link.localInertia = Mat3f(0);
            continue;
        }

        // 构建路径
        auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.collisionMeshPath);
        std::string meshFile = meshFull.string();
        std::string meshFolder = meshFull.path().parent_path().string();

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // 加载 OBJ (不加载材质，以加快速度)
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshFile.c_str(), meshFolder.c_str());

        if (!ret) {
            std::cerr << "Failed to load collision obj: " << meshFile << std::endl;
            continue;
        }

        // 容器：存储合并后的所有三角形和变换后的顶点
        std::vector<Vec3f> colVertices;
        std::vector<TopologyModule::Triangle> colIndices;

        // 1. 提取所有顶点并应用局部变换 (link.meshTransform)
        // 注意：我们必须先应用变换，再计算惯量，这样惯量才是基于 Link 坐标系的分布计算的
        // 这里的 meshTransform 对应 URDF 中的 <collision><origin>
        colVertices.resize(attrib.vertices.size() / 3);
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            Vec3f p(attrib.vertices[i + 0], attrib.vertices[i + 1], attrib.vertices[i + 2]);
            // 变换到 Link 坐标系
            colVertices[i / 3] = link.meshTransform * p;
        }

        // 2. 提取所有 Shape 的面索引
        for (const auto& shape : shapes) {
            const auto& mesh = shape.mesh;
            for (size_t f = 0; f < mesh.indices.size(); f += 3) {
                // tinyobj 的索引
                int idx0 = mesh.indices[f + 0].vertex_index;
                int idx1 = mesh.indices[f + 1].vertex_index;
                int idx2 = mesh.indices[f + 2].vertex_index;

                colIndices.push_back(TopologyModule::Triangle(idx0, idx1, idx2));
            }
        }

        // 3. 计算物理属性
        Real vol = 0;
        Vec3f com(0);
        Mat3f inertia(0);

        computeMeshPhysicalProperties(colVertices, colIndices, vol, com, inertia);

        // 4. 存储结果
        link.volume = vol;
        link.localInertia = inertia;

        // 可选：打印调试信息
        std::cout << "Link: " << link.name << " | Vol: " << link.volume << std::endl;
        std::cout << "Inertia: \n" << link.localInertia << std::endl;
    }

    return true;
}

} // namespace dyno