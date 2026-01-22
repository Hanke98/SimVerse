#include "MeshPatching/KMeansPatcher.h"
#include "UrdfFunc.h"
#include "ImageLoader.h"
#include "GltfFunc.h"
#include "Vector/Vector3D.h"
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <filesystem>

#include "MeshPatching/MortonChunkPatcher.h"
#include "MeshPatching/PatchingTypes.h"
#include "MeshPatching/MeshTopologyBuilder.h"
#include "UrdfWriter.h"

// -----------------------------------------------------------------------------
// Triangle Morton ordering
// -----------------------------------------------------------------------------
namespace {

static inline uint32_t expandBits(uint32_t v)
{
    // Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint32_t morton3D(float x, float y, float z)
{
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    uint32_t xx = expandBits((unsigned int)x);
    uint32_t yy = expandBits((unsigned int)y);
    uint32_t zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

static inline float safeInv(float v, float eps = 1e-9f)
{
    return 1.0f / ((std::abs(v) < eps) ? eps : v);
}

// Build Morton-sorted order for triangles using their (already computed) centers.
// Output 'order' is a permutation of [0..numTris-1], sorted by Morton key.
static void mortonSortTrianglesByCenters(
    const std::vector<dyno::Vec3f>& triCenters,
    std::vector<uint32_t>&          order)
{
    const size_t n = triCenters.size();
    order.resize(n);
    if (n == 0) return;

    dyno::Vec3f cmin(dyno::REAL_MAX);
    dyno::Vec3f cmax(-dyno::REAL_MAX);
    for (const auto& c : triCenters)
    {
        cmin = cmin.minimum(c);
        cmax = cmax.maximum(c);
    }
    dyno::Vec3f extent = cmax - cmin;
    const float invX = safeInv(static_cast<float>(extent.x));
    const float invY = safeInv(static_cast<float>(extent.y));
    const float invZ = safeInv(static_cast<float>(extent.z));

    struct KeyTri { uint64_t key; uint32_t tri; }; //  key: sort key; tri: original triangle index
    std::vector<KeyTri> keys(n);

    for (uint32_t t = 0; t < static_cast<uint32_t>(n); ++t)
    {
        const auto& c = triCenters[t];
        const float nx = static_cast<float>(c.x - cmin.x) * invX;
        const float ny = static_cast<float>(c.y - cmin.y) * invY;
        const float nz = static_cast<float>(c.z - cmin.z) * invZ;
        const uint32_t m = morton3D(nx, ny, nz);

        // Stable tie-breaker with triangle index
        keys[t] = KeyTri{ (static_cast<uint64_t>(m) << 32) | static_cast<uint64_t>(t), t };
    }

    std::sort(keys.begin(), keys.end(), [](const KeyTri& a, const KeyTri& b) {
        return a.key < b.key;
    });

    for (size_t i = 0; i < n; ++i)
        order[i] = keys[i].tri;
}

} 

namespace dyno
{
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

        // 体积积分 (Integral of 1)
        integral[0] += detJ;

        // 一阶矩积分 (Center of Mass term) -> Integral of x, y, z
        Vec3f sum = v0 + v1 + v2;
        integral[1] += detJ * sum.x;
        integral[2] += detJ * sum.y;
        integral[3] += detJ * sum.z;

        // 二阶矩积分 (Covariance terms)
        // Integral(x^2)
        integral[4] += detJ * (v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + sum.x*sum.x);
        // Integral(y^2)
        integral[5] += detJ * (v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + sum.y*sum.y);
        // Integral(z^2)
        integral[6] += detJ * (v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + sum.z*sum.z);

        // Integral(xy) // there is a bug in the paper here
        integral[7] += detJ * (v0.x*v0.y + v1.x*v1.y + v2.x*v2.y + sum.x*sum.y);
        // Integral(yz)
        integral[8] += detJ * (v0.y*v0.z + v1.y*v1.z + v2.y*v2.z + sum.y*sum.z);
        // Integral(zx)
        integral[9] += detJ * (v0.z*v0.x + v1.z*v1.x + v2.z*v2.x + sum.z*sum.x);
    }

    // --- 归一化与后处理 ---

    // detJ 是 6*Vol
    outVolume = static_cast<Real>(integral[0] / 6.0);

    // 防止除以零（空网格）
    if (std::abs(outVolume) < Real(1e-9)) {
        outVolume = 0;
        outCenterOfMass = Vec3f(0);
        outInertiaTensor = Mat3f(0);
        return;
    }

    // 计算质心 (CoM)
    // integral[1] 是 int(x) * 24

    // 更正系数：
    // Volume = integral[0] / 6.0
    // CoM = (Integral 1st Moment) / (24.0) / Volume
    // CoM = (integral[1..3] / 24.0) / (integral[0] / 6.0) = integral[1..3] / (4.0 * integral[0])

    double inv_4_vol = 1.0 / (4.0 * integral[0]);
    outCenterOfMass = Vec3f(
        static_cast<Real>(integral[1] * inv_4_vol),
        static_cast<Real>(integral[2] * inv_4_vol),
        static_cast<Real>(integral[3] * inv_4_vol)
    );

    // 计算相对于原点的协方差矩阵对角线与非对角线
    // 系数：Integral(x^2) = sum(...) * detJ / 120
    // double i_xx = integral[4] / 60.0;

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

    // Output: patchBoundingBox[linkId][patchId] stores world-space AABBs for patches.
    std::vector<std::vector<TAlignedBox3D<Real>>> patchBoundingBox;
    patchBoundingBox.clear();
    patchBoundingBox.resize(links.size());
    const int facesPerPatch = 32;

    // Output: linkAABBs[linkId] stores world-space AABBs for links.
    urdfInfo.linkAABBs.clear();
    urdfInfo.linkAABBs.reserve(links.size());

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
    for (size_t linkId = 0; linkId < links.size(); ++linkId) {
        auto& link = links[linkId];
        if (!link.visualMeshPath.empty()) {
            // Construct the complete path to the mesh
            auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.visualMeshPath);
            std::string meshFile   = meshFull.string();
            std::string meshFolder = meshFull.path().parent_path().string();

            tinyobj::attrib_t                attrib;
            std::vector<tinyobj::shape_t>    shapes;
            std::vector<tinyobj::material_t> materials;
            std::string                      warn, err;

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

                // diffuse texture
                if (!mtl.diffuse_texname.empty())
                {
                    auto tex_path = (urdfRoot / mtl.diffuse_texname).string();
                    if (loader->loadImage(tex_path.c_str(), texture))
                    {
                        mat->texColor.assign(texture);
                    }
                }

                // bump / normal maps
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

            Transform3f T_world_mesh = composeTransform(link.T_world, link.T_mesh);
            Vec3f lo( REAL_MAX);
            Vec3f hi(-REAL_MAX);

            link.T_world = T_world_mesh;// need double check here!!!!!!

            for (const auto& tshape : shapes)
            {
                const auto& mesh = tshape.mesh;

                // auto shape = std::make_shared<Shape>();

                // std::vector<TopologyModule::Triangle> vertexIndex;
                // std::vector<TopologyModule::Triangle> normalIndex;
                // std::vector<TopologyModule::Triangle> texCoordIndex;

                // Bind materials (each shape in tinyobj can have material_ids)
                // if (!mesh.material_ids.empty() && mesh.material_ids[0] >= 0)
                // {
                //     int localMatId  = mesh.material_ids[0];
                //
                //     int globalMatId = static_cast<int>(matOffset) + localMatId;
                //
                //     if (globalMatId >= 0 && globalMatId < static_cast<int>(reMats.size()))
                //         shape->material = reMats[globalMatId];
                // }

                // In tinyobj, the indices are a list of triangles (each index contains the subscripts of v / n / t)
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

                    // Fill in shapeIds: mark these vertices as the current globalShapeId
                    shapeIds[v0] = globalShapeId;
                    shapeIds[v1] = globalShapeId;
                    shapeIds[v2] = globalShapeId;
                }
            }
            mergedShape->vertexIndex.assign(vertexIndex);
            mergedShape->normalIndex.assign(normalIndex);
            mergedShape->texCoordIndex.assign(texCoordIndex);

            // Bounding box and center
            auto shapeCenter = (lo + hi) * Real(0.5);
            mergedShape->boundingBox       = TAlignedBox3D<Real>(lo, hi);
            mergedShape->boundingTransform = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));
            urdfInfo.linkAABBs.push_back(TAlignedBox3D<Real>(lo, hi));

            // The posture of link under world space
            Transform3f T_w_link = link.T_world;
            Mat3f R_w_link = T_w_link.rotation();
            Vec3f t_w_link = T_w_link.translation();

            // Convert world coordinates back to link local: p_local = R^T (p_world - t)
            Vec3f center_local = R_w_link.transpose() * (shapeCenter - t_w_link);

            link.T_visual_bb_world = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));

            reShapes.push_back(mergedShape);
            link.visualShapeId = globalShapeId;
            globalShapeId++;

            // p_world = T_world * link.meshTransform * p_mesh
            auto R = T_world_mesh.rotation();
            for (size_t i = vOffset; i < vertices.size(); ++i)
            {
                vertices[i] = T_world_mesh * vertices[i];
            }

            // -----------------------------------------------------------------
            // Create patch shapes (world space) for this link (visual mesh).
            // -----------------------------------------------------------------
            if (!link.hasPatch) {
                PatchingParams     patchParams;
                PatchingResultHost patchResult;
                // MortonChunkPatcher patcher;
                KMeansPatcher      patcher;
                // MeshTopologyHost   topo;
                auto topo = MeshTopologyBuilder::BuildFromTriangles((int)vertexIndex.size(), vertexIndex);

                // topo.numFaces = static_cast<int>(vertexIndex.size());
                patchParams.targetFacesPerPatch = std::max(0, facesPerPatch);
                patcher.BuildPatches(topo, patchParams, patchResult);

                // Add assertions to verify patch allocation correctness
                assert(patchResult.patchOffsets.back() == topo.numFaces);
                assert(patchResult.patchFaces.size() == topo.numFaces);

                // Check that all facePatchId are assigned (not -1)
                for (int i = 0; i < topo.numFaces; ++i) {
                    assert(patchResult.facePatchId[i] != -1);
                }

                // Check patchFaces for no duplicates and no omissions (debug mode only)
                #ifndef NDEBUG
                std::vector<uint8_t> seen(topo.numFaces, 0);
                for (size_t i = 0; i < patchResult.patchFaces.size(); ++i) {
                    int face = patchResult.patchFaces[i];
                    assert(face >= 0 && face < topo.numFaces);
                    assert(seen[face] == 0); // no duplicate
                    seen[face] = 1;
                }
                for (int i = 0; i < topo.numFaces; ++i) {
                    assert(seen[i] == 1); // no omission
                }
                #endif

                link.patchFaces = patchResult.patchFaces;
                link.patchOffsets = patchResult.patchOffsets;

                link.patchAABBs.clear();
                link.patchAABBs.reserve(patchResult.numPatches);

                auto& bboxout = patchBoundingBox[linkId];
                bboxout.clear();
                bboxout.reserve(patchResult.numPatches);

                for (int p = 0; p < patchResult.numPatches; ++p)
                {
                    const int begin = patchResult.patchOffsets[p];
                    const int end   = patchResult.patchOffsets[p + 1];

                    Vec3f plo(REAL_MAX);
                    Vec3f phi(-REAL_MAX);

                    std::vector<TopologyModule::Triangle> patchVertexIndex;
                    std::vector<TopologyModule::Triangle> patchNormalIndex;
                    std::vector<TopologyModule::Triangle> patchTexCoordIndex;

                    patchVertexIndex.reserve(end - begin);
                    patchNormalIndex.reserve(end - begin);
                    patchTexCoordIndex.reserve(end - begin);

                    for (int t = begin; t < end; ++t)
                    {
                        auto faceIndex = patchResult.patchFaces[t];
                        const auto& tri = topo.faceVerts[faceIndex];
                        const Vec3f& a = vertices[tri[0]];
                        const Vec3f& b = vertices[tri[1]];
                        const Vec3f& c = vertices[tri[2]];

                        plo = plo.minimum(a).minimum(b).minimum(c);
                        phi = phi.maximum(a).maximum(b).maximum(c);

                        patchVertexIndex.push_back(vertexIndex[faceIndex]);
                        patchNormalIndex.push_back(normalIndex[faceIndex]);
                        patchTexCoordIndex.push_back(texCoordIndex[faceIndex]);

                        // // Set shapeIds for this patch
                        // shapeIds[tri[0]] = globalShapeId;
                        // shapeIds[tri[1]] = globalShapeId;
                        // shapeIds[tri[2]] = globalShapeId;
                    }

                    bboxout.emplace_back(TAlignedBox3D<Real>(plo, phi));
                    link.patchAABBs.push_back(TAlignedBox3D<Real>(plo, phi));

                    // // Create patchShape
                    // std::shared_ptr<Shape> patchShape = std::make_shared<Shape>();
                    // patchShape->vertexIndex.assign(patchVertexIndex);
                    // patchShape->normalIndex.assign(patchNormalIndex);
                    // patchShape->texCoordIndex.assign(patchTexCoordIndex);
                    // // patchShape->boundingBox = TAlignedBox3D<Real>(plo, phi);
                    // patchShape->boundingBox = reShapes[link.visualShapeId]->boundingBox; 

                    // auto patchCenter = (plo + phi) * Real(0.5);
                    // patchShape->boundingTransform = Transform3f(patchCenter, Mat3f::identityMatrix(), Vec3f(1));
                    // patchShape->boundingTransform = reShapes[link.visualShapeId]->boundingTransform;

                    // // Material with different color for adjacent patches
                    // auto mat = std::make_shared<Material>();
                    // Vec3f colors[] = {Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1), Vec3f(1,1,0), Vec3f(1,0,1), Vec3f(0,1,1)};
                    // mat->baseColor = colors[p % 6];
                    // reMats.push_back(mat);
                    // patchShape->material = reMats.back();

                    // reShapes.push_back(patchShape);
                    // link.patchShapeIds.push_back(globalShapeId);

                    // globalShapeId++;
                }
                // auto outputUrdfPath = urdfRoot.string() + "robotarm_with_patches.urdf";
                auto outputUrdfPath = urdfPath.string().substr(0, urdfPath.string().length() - 5) + "_with_patches.urdf";
                PatchWriteOptions writeOption;
                writeOption.overwriteExistingPatch = false;
                writeOption.writeMissingOnly = true;
                WriteUrdfWithPatches(urdfPath.string(), outputUrdfPath, urdfInfo, writeOption);
            }

            if (link.hasPatch) {
                // Divide the visual mesh into patches and compute their bounding boxes
                size_t numTriangles = reShapes[link.visualShapeId]->vertexIndex.size();

                size_t numPatches = link.patchFaces.size();
                patchBoundingBox[linkId].resize(numPatches);

                for (size_t p = 0; p < numPatches; ++p)
                {
                    size_t startTri = link.patchOffsets[p];
                    size_t endTri   = link.patchOffsets[p + 1];

                    Vec3f patchLo( REAL_MAX);
                    Vec3f patchHi(-REAL_MAX);

                    for (size_t t = startTri; t < endTri; ++t)
                    {
                        const auto indexTri = link.patchFaces[t];
                        auto tri = vertexIndex[indexTri];
                        for (int vi = 0; vi < 3; ++vi)
                        {
                            Vec3f v = vertices[tri[vi]];
                            patchLo = patchLo.minimum(v);
                            patchHi = patchHi.maximum(v);
                        }
                    }

                    patchBoundingBox[linkId][p] = TAlignedBox3D<Real>(patchLo, patchHi);
                }
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
        
        // avoid loading collision mesh for now
        // if (!link.collisionMeshPath.empty()) {
        //     // Construct the complete path to the mesh
        //     auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.collisionMeshPath);
        //     std::string meshFile   = meshFull.string();
        //     std::string meshFolder = meshFull.path().parent_path().string();

        //     tinyobj::attrib_t                attrib;
        //     std::vector<tinyobj::shape_t>    shapes;
        //     std::vector<tinyobj::material_t> materials;
        //     std::string                      warn, err;

        //     bool ret = tinyobj::LoadObj(
        //         &attrib,
        //         &shapes,
        //         &materials,
        //         &warn,
        //         &err,
        //         meshFile.c_str(),
        //         meshFolder.c_str()
        //     );

        //     if (!warn.empty())
        //         std::cerr << "tinyobj warn: " << warn << std::endl;
        //     if (!err.empty())
        //     {
        //         std::cerr << "tinyobj err: " << err << std::endl;
        //         continue;
        //     }
        //     if (!ret)
        //     {
        //         std::cerr << "Failed to load obj: " << meshFile << std::endl;
        //         continue;
        //     }

        //     size_t vOffset = vertices.size();
        //     size_t nOffset = normals.size();
        //     size_t tOffset = texCoords.size();

        //     bool hasNormals   = !attrib.normals.empty();
        //     bool hasTexcoords = !attrib.texcoords.empty();

        //     // Append the data from the obj file
        //     for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        //     {
        //         vertices.push_back(Vec3f(
        //             attrib.vertices[i + 0],
        //             attrib.vertices[i + 1],
        //             attrib.vertices[i + 2]
        //         ));
        //     }
        //     if (hasNormals) {
        //         for (size_t i = 0; i < attrib.normals.size(); i += 3)
        //         {
        //             normals.push_back(Vec3f(
        //                 attrib.normals[i + 0],
        //                 attrib.normals[i + 1],
        //                 attrib.normals[i + 2]
        //             ));
        //         }
        //     }

        //     if (hasTexcoords) {
        //         for (size_t i = 0; i < attrib.texcoords.size(); i += 2)
        //         {
        //             texCoords.push_back(Vec2f(
        //                 attrib.texcoords[i + 0],
        //                 attrib.texcoords[i + 1]
        //             ));
        //         }
        //     } else {
        //         if (texCoords.size() < vertices.size())
        //         {
        //             texCoords.resize(vertices.size());
        //         }

        //         for (size_t vi = vOffset; vi < vertices.size(); ++vi)
        //         {
        //             texCoords[vi] = Vec2f(0.0f, 0.0f);
        //         }
        //     }

        //     shapeIds.resize(vertices.size());

        //     // convert materials of tinyobj to engine, and add them into reMats
        //     uint matOffset = static_cast<uint>(reMats.size());
        //     reMats.resize(reMats.size() + materials.size());

        //     dyno::CArray2D<dyno::Vec4f> texture(1, 1);
        //     texture[0, 0] = dyno::Vec4f(1);

        //     for (size_t mId = 0; mId < materials.size(); ++mId)
        //     {
        //         const auto& mtl = materials[mId];
        //         reMats[matOffset + mId] = std::make_shared<Material>();
        //         auto& mat = reMats[matOffset + mId];

        //         mat->baseColor = Vec3f(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);

        //         std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();

        //         // diffuse 纹理
        //         if (!mtl.diffuse_texname.empty())
        //         {
        //             auto tex_path = (urdfRoot / mtl.diffuse_texname).string();
        //             if (loader->loadImage(tex_path.c_str(), texture))
        //             {
        //                 mat->texColor.assign(texture);
        //             }
        //         }

        //         // bump / normal 贴图
        //         if (!mtl.bump_texname.empty())
        //         {
        //             auto tex_path = (urdfRoot/ mtl.bump_texname).string();
        //             if (loader->loadImage(tex_path.c_str(), texture))
        //             {
        //                 mat->texBump.assign(texture);
        //                 auto texOpt = mtl.bump_texopt;
        //                 mat->bumpScale = texOpt.bump_multiplier;
        //             }
        //         }
        //     }

        //     // Merge all shapes into a single shape
        //     std::shared_ptr<Shape> mergedShape = std::make_shared<Shape>();
        //     std::vector<TopologyModule::Triangle> vertexIndex;
        //     std::vector<TopologyModule::Triangle> normalIndex;
        //     std::vector<TopologyModule::Triangle> texCoordIndex;

        //     Transform3f T_world_mesh = composeTransform(link.T_world, link.T_mesh);
        //     Vec3f lo( REAL_MAX);
        //     Vec3f hi(-REAL_MAX);

        //     for (const auto& tshape : shapes)
        //     {
        //         const auto& mesh = tshape.mesh;

        //         // tinyobj 里 indices 是三角形列表（每个 index 里有 v / n / t 下标）
        //         for (size_t i = 0; i < mesh.indices.size(); i += 3)
        //         {
        //             auto idx0 = mesh.indices[i + 0];
        //             auto idx1 = mesh.indices[i + 1];
        //             auto idx2 = mesh.indices[i + 2];

        //             // 加上 offset，把局部下标变成全局下标
        //             int v0 = idx0.vertex_index + static_cast<int>(vOffset);
        //             int v1 = idx1.vertex_index + static_cast<int>(vOffset);
        //             int v2 = idx2.vertex_index + static_cast<int>(vOffset);

        //             TopologyModule::Triangle tri(v0, v1, v2);

        //             vertexIndex.push_back(tri);

        //             if (hasNormals && idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
        //                 int n0 = (idx0.normal_index  >= 0) ? idx0.normal_index  + static_cast<int>(nOffset) : -1;
        //                 int n1 = (idx1.normal_index  >= 0) ? idx1.normal_index  + static_cast<int>(nOffset) : -1;
        //                 int n2 = (idx2.normal_index  >= 0) ? idx2.normal_index  + static_cast<int>(nOffset) : -1;
        //                 normalIndex.push_back(TopologyModule::Triangle(n0, n1, n2));
        //             } else {
        //                 normalIndex.push_back(tri);
        //             }

        //             if (hasTexcoords && idx0.texcoord_index >= 0 && idx1.texcoord_index >= 0 && idx2.texcoord_index >= 0) {
        //                 int t0 = (idx0.texcoord_index >= 0) ? idx0.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 int t1 = (idx1.texcoord_index >= 0) ? idx1.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 int t2 = (idx2.texcoord_index >= 0) ? idx2.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 texCoordIndex.push_back(TopologyModule::Triangle(t0, t1, t2));
        //             } else {
        //                 texCoordIndex.push_back(tri);
        //             }
        //             // TODO: Use oriented bounding box, and transform the bb later.
        //             Vec3f transformedV0 = T_world_mesh * vertices[v0];
        //             Vec3f transformedV1 = T_world_mesh * vertices[v1];
        //             Vec3f transformedV2 = T_world_mesh * vertices[v2];

        //             // Update the bounding box with transformed vertices
        //             lo = lo.minimum(transformedV0);
        //             lo = lo.minimum(transformedV1);
        //             lo = lo.minimum(transformedV2);

        //             hi = hi.maximum(transformedV0);
        //             hi = hi.maximum(transformedV1);
        //             hi = hi.maximum(transformedV2);

        //             // 填 shapeIds：把这几个顶点标记为当前 globalShapeId
        //             shapeIds[v0] = globalShapeId;
        //             shapeIds[v1] = globalShapeId;
        //             shapeIds[v2] = globalShapeId;
        //         }
        //     }
        //     mergedShape->vertexIndex.assign(vertexIndex);
        //     mergedShape->normalIndex.assign(normalIndex);
        //     mergedShape->texCoordIndex.assign(texCoordIndex);

        //     // 包围盒与中心
        //     auto shapeCenter = (lo + hi) * Real(0.5);
        //     mergedShape->boundingBox       = TAlignedBox3D<Real>(lo, hi);
        //     mergedShape->boundingTransform = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));

        //     link.T_collision_bb_world = Transform3f(shapeCenter, Mat3f::identityMatrix(), Vec3f(1));

        //     reShapes.push_back(mergedShape);
        //     link.collisionShapeId = globalShapeId;
        //     globalShapeId++;

        //     // p_world = T_world * link.meshTransform * p_mesh
        //     auto R = T_world_mesh.rotation();
        //     for (size_t i = vOffset; i < vertices.size(); ++i)
        //     {
        //         vertices[i] = T_world_mesh * vertices[i];
        //     }
        //     if (hasNormals)
        //     {
        //         for (size_t i = nOffset; i < normals.size(); ++i)
        //         {
        //             normals[i] = R * normals[i];
        //             Real len = normals[i].norm();
        //             if (len > Real(1e-8)) normals[i] /= len;
        //         }
        //     }
        // }

        // if (!link.visualMeshPath.empty()) {
        //     // Construct the complete path to the mesh
        //     auto meshFull = FilePath(getAssetPath() + "/../asset/" + link.visualMeshPath);
        //     std::string meshFile   = meshFull.string();
        //     std::string meshFolder = meshFull.path().parent_path().string();

        //     tinyobj::attrib_t                attrib;
        //     std::vector<tinyobj::shape_t>    shapes;
        //     std::vector<tinyobj::material_t> materials;
        //     std::string                      warn, err;

        //     bool ret = tinyobj::LoadObj(
        //         &attrib,
        //         &shapes,
        //         &materials,
        //         &warn,
        //         &err,
        //         meshFile.c_str(),
        //         meshFolder.c_str()
        //     );

        //     if (!warn.empty())
        //         std::cerr << "tinyobj warn: " << warn << std::endl;
        //     if (!err.empty())
        //     {
        //         std::cerr << "tinyobj err: " << err << std::endl;
        //         continue;
        //     }
        //     if (!ret)
        //     {
        //         std::cerr << "Failed to load obj: " << meshFile << std::endl;
        //         continue;
        //     }

        //     size_t vOffset = vertices.size();
        //     size_t nOffset = normals.size();
        //     size_t tOffset = texCoords.size();

        //     bool hasNormals   = !attrib.normals.empty();
        //     bool hasTexcoords = !attrib.texcoords.empty();

        //     // Append the data from the obj file
        //     for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        //     {
        //         vertices.push_back(Vec3f(
        //             attrib.vertices[i + 0],
        //             attrib.vertices[i + 1],
        //             attrib.vertices[i + 2]
        //         ));
        //     }
        //     if (hasNormals) {
        //         for (size_t i = 0; i < attrib.normals.size(); i += 3)
        //         {
        //             normals.push_back(Vec3f(
        //                 attrib.normals[i + 0],
        //                 attrib.normals[i + 1],
        //                 attrib.normals[i + 2]
        //             ));
        //         }
        //     }

        //     if (hasTexcoords) {
        //         for (size_t i = 0; i < attrib.texcoords.size(); i += 2)
        //         {
        //             texCoords.push_back(Vec2f(
        //                 attrib.texcoords[i + 0],
        //                 attrib.texcoords[i + 1]
        //             ));
        //         }
        //     } else {
        //         if (texCoords.size() < vertices.size())
        //         {
        //             texCoords.resize(vertices.size());
        //         }

        //         for (size_t vi = vOffset; vi < vertices.size(); ++vi)
        //         {
        //             texCoords[vi] = Vec2f(0.0f, 0.0f);
        //         }
        //     }

        //     shapeIds.resize(vertices.size());

        //     dyno::CArray2D<dyno::Vec4f> texture(1, 1);
        //     texture[0, 0] = dyno::Vec4f(1);

        //     std::vector<TopologyModule::Triangle> vertexIndex;
        //     std::vector<TopologyModule::Triangle> normalIndex;
        //     std::vector<TopologyModule::Triangle> texCoordIndex;
        //     // Triangle centers (for Morton ordering). One entry per triangle.
        //     std::vector<Vec3f> triCenters;

        //     Transform3f T_world_mesh = composeTransform(link.T_world, link.T_mesh);

        //     for (const auto& tshape : shapes)
        //     {
        //         const auto& mesh = tshape.mesh;

        //         // In tinyobj, indices are a list of triangles (each index contains v/n/t subscripts)
        //         for (size_t i = 0; i < mesh.indices.size(); i += 3)
        //         {
        //             auto idx0 = mesh.indices[i + 0];
        //             auto idx1 = mesh.indices[i + 1];
        //             auto idx2 = mesh.indices[i + 2];

        //             // Add the offset to convert the local subscript into a global subscript
        //             int v0 = idx0.vertex_index + static_cast<int>(vOffset);
        //             int v1 = idx1.vertex_index + static_cast<int>(vOffset);
        //             int v2 = idx2.vertex_index + static_cast<int>(vOffset);

        //             TopologyModule::Triangle tri(v0, v1, v2);

        //             vertexIndex.push_back(tri);

        //             if (hasNormals && idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
        //                 int n0 = (idx0.normal_index  >= 0) ? idx0.normal_index  + static_cast<int>(nOffset) : -1;
        //                 int n1 = (idx1.normal_index  >= 0) ? idx1.normal_index  + static_cast<int>(nOffset) : -1;
        //                 int n2 = (idx2.normal_index  >= 0) ? idx2.normal_index  + static_cast<int>(nOffset) : -1;
        //                 normalIndex.push_back(TopologyModule::Triangle(n0, n1, n2));
        //             } else {
        //                 normalIndex.push_back(tri);
        //             }

        //             if (hasTexcoords && idx0.texcoord_index >= 0 && idx1.texcoord_index >= 0 && idx2.texcoord_index >= 0) {
        //                 int t0 = (idx0.texcoord_index >= 0) ? idx0.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 int t1 = (idx1.texcoord_index >= 0) ? idx1.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 int t2 = (idx2.texcoord_index >= 0) ? idx2.texcoord_index + static_cast<int>(tOffset) : -1;
        //                 texCoordIndex.push_back(TopologyModule::Triangle(t0, t1, t2));
        //             } else {
        //                 texCoordIndex.push_back(tri);
        //             }
        //             // TODO: Use oriented bounding box, and transform the bb later.
        //             Vec3f transformedV0 = T_world_mesh * vertices[v0];
        //             Vec3f transformedV1 = T_world_mesh * vertices[v1];
        //             Vec3f transformedV2 = T_world_mesh * vertices[v2];

        //             // Store for later normalization + Morton sort
        //             triCenters.push_back((transformedV0 + transformedV1 + transformedV2) * (Real(1.0) / Real(3.0)));
        //         }
        //     }

        //     // -----------------------------------------------------------------
        //     // Morton sort triangles to improve spatial locality.
        //     // -----------------------------------------------------------------
        //     // if (triCenters.size() == vertexIndex.size() && !vertexIndex.empty())
        //     // {
        //     //     std::vector<uint32_t> triOrder;
        //     //     mortonSortTrianglesByCenters(triCenters, triOrder);

        //     //     // Reorder triangle index arrays consistently.
        //     //     std::vector<TopologyModule::Triangle> vSorted(vertexIndex.size());
        //     //     std::vector<TopologyModule::Triangle> nSorted(normalIndex.size());
        //     //     std::vector<TopologyModule::Triangle> tSorted(texCoordIndex.size());

        //     //     for (size_t k = 0; k < triOrder.size(); ++k)
        //     //     {
        //     //         const uint32_t src = triOrder[k];
        //     //         vSorted[k] = vertexIndex[src];
        //     //         nSorted[k] = normalIndex[src];
        //     //         tSorted[k] = texCoordIndex[src];
        //     //     }
        //     //     vertexIndex.swap(vSorted);
        //     //     normalIndex.swap(nSorted);
        //     //     texCoordIndex.swap(tSorted);
        //     // }

        //     // p_world = T_world * link.meshTransform * p_mesh
        //     for (size_t i = vOffset; i < vertices.size(); ++i)
        //     {
        //         vertices[i] = T_world_mesh * vertices[i];
        //     }
            
        //     // -----------------------------------------------------------------
        //     // Create patch shapes (world space) for this link (visual mesh).
        //     // -----------------------------------------------------------------
        //     if (!link.hasPatch) {
        //         PatchingParams     patchParams;
        //         PatchingResultHost patchResult;
        //         // MortonChunkPatcher patcher;
        //         KMeansPatcher      patcher;
        //         // MeshTopologyHost   topo;
        //         auto topo = MeshTopologyBuilder::BuildFromTriangles((int)vertexIndex.size(), vertexIndex);

        //         // topo.numFaces = static_cast<int>(vertexIndex.size());
        //         patchParams.targetFacesPerPatch = std::max(0, facesPerPatch);
        //         patcher.BuildPatches(topo, patchParams, patchResult);

        //         // Add assertions to verify patch allocation correctness
        //         assert(patchResult.patchOffsets.back() == topo.numFaces);
        //         assert(patchResult.patchFaces.size() == topo.numFaces);

        //         // Check that all facePatchId are assigned (not -1)
        //         for (int i = 0; i < topo.numFaces; ++i) {
        //             assert(patchResult.facePatchId[i] != -1);
        //         }

        //         // Check patchFaces for no duplicates and no omissions (debug mode only)
        //         #ifndef NDEBUG
        //         std::vector<uint8_t> seen(topo.numFaces, 0);
        //         for (size_t i = 0; i < patchResult.patchFaces.size(); ++i) {
        //             int face = patchResult.patchFaces[i];
        //             assert(face >= 0 && face < topo.numFaces);
        //             assert(seen[face] == 0); // no duplicate
        //             seen[face] = 1;
        //         }
        //         for (int i = 0; i < topo.numFaces; ++i) {
        //             assert(seen[i] == 1); // no omission
        //         }
        //         #endif

        //         link.patchFaces = patchResult.patchFaces;
        //         link.patchOffsets = patchResult.patchOffsets;

        //         link.patchAABBs.clear();
        //         link.patchAABBs.reserve(patchResult.numPatches);

        //         auto& bboxout = patchBoundingBox[linkId];
        //         bboxout.clear();
        //         bboxout.reserve(patchResult.numPatches);

        //         for (int p = 0; p < patchResult.numPatches; ++p)
        //         {
        //             const int begin = patchResult.patchOffsets[p];
        //             const int end   = patchResult.patchOffsets[p + 1];

        //             Vec3f plo(REAL_MAX);
        //             Vec3f phi(-REAL_MAX);

        //             std::vector<TopologyModule::Triangle> patchVertexIndex;
        //             std::vector<TopologyModule::Triangle> patchNormalIndex;
        //             std::vector<TopologyModule::Triangle> patchTexCoordIndex;

        //             patchVertexIndex.reserve(end - begin);
        //             patchNormalIndex.reserve(end - begin);
        //             patchTexCoordIndex.reserve(end - begin);

        //             for (int t = begin; t < end; ++t)
        //             {
        //                 auto faceIndex = patchResult.patchFaces[t];
        //                 const auto& tri = topo.faceVerts[faceIndex];
        //                 const Vec3f& a = vertices[tri[0]];
        //                 const Vec3f& b = vertices[tri[1]];
        //                 const Vec3f& c = vertices[tri[2]];

        //                 plo = plo.minimum(a).minimum(b).minimum(c);
        //                 phi = phi.maximum(a).maximum(b).maximum(c);

        //                 patchVertexIndex.push_back(vertexIndex[faceIndex]);
        //                 patchNormalIndex.push_back(normalIndex[faceIndex]);
        //                 patchTexCoordIndex.push_back(texCoordIndex[faceIndex]);

        //                 // Set shapeIds for this patch
        //                 shapeIds[tri[0]] = globalShapeId;
        //                 shapeIds[tri[1]] = globalShapeId;
        //                 shapeIds[tri[2]] = globalShapeId;
        //             }

        //             bboxout.emplace_back(TAlignedBox3D<Real>(plo, phi));
        //             link.patchAABBs.push_back(TAlignedBox3D<Real>(plo, phi));

        //             // Create patchShape
        //             std::shared_ptr<Shape> patchShape = std::make_shared<Shape>();
        //             patchShape->vertexIndex.assign(patchVertexIndex);
        //             patchShape->normalIndex.assign(patchNormalIndex);
        //             patchShape->texCoordIndex.assign(patchTexCoordIndex);
        //             // patchShape->boundingBox = TAlignedBox3D<Real>(plo, phi);
        //             patchShape->boundingBox = reShapes[link.visualShapeId]->boundingBox; 

        //             // auto patchCenter = (plo + phi) * Real(0.5);
        //             // patchShape->boundingTransform = Transform3f(patchCenter, Mat3f::identityMatrix(), Vec3f(1));
        //             patchShape->boundingTransform = reShapes[link.visualShapeId]->boundingTransform;

        //             // Material with different color for adjacent patches
        //             auto mat = std::make_shared<Material>();
        //             Vec3f colors[] = {Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1), Vec3f(1,1,0), Vec3f(1,0,1), Vec3f(0,1,1)};
        //             mat->baseColor = colors[p % 6];
        //             reMats.push_back(mat);
        //             patchShape->material = reMats.back();

        //             reShapes.push_back(patchShape);
        //             link.patchShapeIds.push_back(globalShapeId);

        //             globalShapeId++;
        //         }
        //         // auto outputUrdfPath = urdfRoot.string() + "robotarm_with_patches.urdf";
        //         auto outputUrdfPath = urdfPath.string().substr(0, urdfPath.string().length() - 5) + "_with_patches.urdf";
        //         PatchWriteOptions writeOption;
        //         writeOption.overwriteExistingPatch = false;
        //         writeOption.writeMissingOnly = true;
        //         WriteUrdfWithPatches(urdfPath.string(), outputUrdfPath, urdfInfo, writeOption);
        //     }

        //     if (link.hasPatch) {
        //         // Divide the visual mesh into patches and compute their bounding boxes
        //         size_t numTriangles = reShapes[link.visualShapeId]->vertexIndex.size();

        //         size_t numPatches = link.patchFaces.size();
        //         patchBoundingBox[linkId].resize(numPatches);

        //         for (size_t p = 0; p < numPatches; ++p)
        //         {
        //             size_t startTri = link.patchOffsets[p];
        //             size_t endTri   = link.patchOffsets[p + 1];

        //             Vec3f patchLo( REAL_MAX);
        //             Vec3f patchHi(-REAL_MAX);

        //             for (size_t t = startTri; t < endTri; ++t)
        //             {
        //                 const auto indexTri = link.patchFaces[t];
        //                 auto tri = vertexIndex[indexTri];
        //                 for (int vi = 0; vi < 3; ++vi)
        //                 {
        //                     Vec3f v = vertices[tri[vi]];
        //                     patchLo = patchLo.minimum(v);
        //                     patchHi = patchHi.maximum(v);
        //                 }
        //             }

        //             patchBoundingBox[linkId][p] = TAlignedBox3D<Real>(patchLo, patchHi);
        //         }
        //     }

        //     auto R = T_world_mesh.rotation();
        //     if (hasNormals)
        //     {
        //         for (size_t i = nOffset; i < normals.size(); ++i)
        //         {
        //             normals[i] = R * normals[i];
        //             Real len = normals[i].norm();
        //             if (len > Real(1e-8)) normals[i] /= len;
        //         }
        //     }
        // }
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
        // Perform point counter operation to assign shape ids
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
    // Process the Collision Mesh and calculate the volume and moment of inertia
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

        // 提取所有顶点并应用局部变换 (link.meshTransform)
        // TODO: 这里的meshTransform应该要对应URDF中的 <collision><origin>

        Transform3f T_world_mesh = composeTransform(link.T_world, link.T_mesh);
        colVertices.resize(attrib.vertices.size() / 3);
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            Vec3f p(attrib.vertices[i + 0], attrib.vertices[i + 1], attrib.vertices[i + 2]);
            // 变换到 Link 坐标系
            colVertices[i / 3] = T_world_mesh * p;
        }

        // 提取所有 Shape 的面索引
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

        // 计算物理属性
        Real vol = 0;
        Vec3f com(0);
        Mat3f inertia(0);

        computeMeshPhysicalProperties(colVertices, colIndices, vol, com, inertia);

        // 存储结果
        link.volume = vol;
        link.localInertia = inertia;

        // 打印调试信息
        // std::cout << "Link: " << link.name << " | Vol: " << link.volume << std::endl;
        // std::cout << "Inertia: \n" << link.localInertia << std::endl;
    }

    auto& joints = urdfInfo.joints;
    for (auto & joint : joints) {
        auto parentId = joint.parentLinkId;
        auto childId = joint.childLinkId;

        // Transform3f T_boundingbox_world;
        // if (!this->varVisualOrCollision()->getValue()) {
            Transform3f T_visual_bb_world = urdfInfo.links[childId].T_visual_bb_world;
        // } else {
            Transform3f T_collision_bb_world = urdfInfo.links[childId].T_collision_bb_world;
        // }

        // 获取 Parent Joint 的世界旋转矩阵 (R_PJ)
        Mat3f R_PJ = joint.originWorld.rotation();

        // 获取 Bounding Box 的世界旋转矩阵 (R_BB)
        // Mat3f R_BB = T_boundingbox_world.rotation();
        Mat3f R_BB_visual = T_visual_bb_world.rotation();
        Mat3f R_BB_collision = T_collision_bb_world.rotation();

        // 计算相对旋转 (R_PJ_to_BB = R_PJ_transpose * R_BB)
        // Mat3f relativeRotation_visual = R_PJ.transpose() * R_BB;
        Mat3f relativeRotation_visual = R_PJ.transpose() * R_BB_visual;
        Mat3f relativeRotation_collision = R_PJ.transpose() * R_BB_collision;

        // 获取世界坐标系下的相对平移向量 (t_BB - t_PJ)
        // Vec3f worldDeltaTranslation = T_boundingbox_world.translation()
        //                               - joints[j].originWorld.translation();
        Vec3f worldDeltaTranslation_visual = T_visual_bb_world.translation()
                                             - joint.originWorld.translation();
        Vec3f worldDeltaTranslation_collision = T_collision_bb_world.translation()
                                             - joint.originWorld.translation();


        // 获取 Parent Joint 的世界旋转矩阵转置 (R_PJ_transpose)
        Mat3f R_PJ_transpose = joint.originWorld.rotation().transpose();
        // 计算相对平移
        // Vec3f relativeTranslation = R_PJ_transpose * worldDeltaTranslation;
        Vec3f relativeTranslation_visual = R_PJ_transpose * worldDeltaTranslation_visual;
        Vec3f relativeTranslation_collision = R_PJ_transpose * worldDeltaTranslation_collision;
        // if (!visual_or_collision) {
            urdfInfo.links[childId].T_visual_bb_local.translation() = relativeTranslation_visual;
            urdfInfo.links[childId].T_visual_bb_local.rotation() = relativeRotation_visual;
        // }
        // else {
            urdfInfo.links[childId].T_collision_bb_local.translation() = relativeTranslation_collision;
            urdfInfo.links[childId].T_collision_bb_local.rotation() = relativeRotation_collision;
        // }

    }

    return true;
}

} // namespace dyno