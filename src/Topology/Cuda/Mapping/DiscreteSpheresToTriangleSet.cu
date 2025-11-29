//
// Created by wjv on 2025/11/28.
//
#include "DiscreteSpheresToTriangleSet.h"

namespace dyno
{
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	DiscreteSpheresToTriangleSet<TDataType>::DiscreteSpheresToTriangleSet()
		: TopologyMapping()
	{
		mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
	}

	__global__ void SetupVerticesForSphereInstances1(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<Sphere3D> sphereInstances,
		uint pointOffset,
		uint sphereOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInstances.size() * sphereVertices.size()) return;

		uint instanceId = tId / sphereVertices.size();
		uint vertexId = tId % sphereVertices.size();

		Sphere3D sphere = sphereInstances[instanceId];

		Vec3f v = sphereVertices[vertexId];
		vertices[pointOffset + tId] = sphere.center + sphere.radius * sphere.rotation.rotate(v);
	}

	template<typename Triangle>
	__global__ void SetupIndicesForSphereInstances1(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<Sphere3D> sphereInstances,
		uint vertexSize,						//vertex size of the instance sphere
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInstances.size() * sphereIndices.size()) return;

		uint instanceId = tId / sphereIndices.size();
		uint indexId = tId % sphereIndices.size();

		int vertexOffset = indexOffset + instanceId * vertexSize;

		Triangle tIndex = sphereIndices[indexId];
		indices[indexOffset + tId] = Triangle(tIndex[0] + vertexOffset, tIndex[1] + vertexOffset, tIndex[2] + vertexOffset);
	}

	template<typename TDataType>
	bool DiscreteSpheresToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
		}

		auto inTopo = this->inDiscreteElements()->constDataPtr();

		DArray<Sphere3D>& sphereInGlobal = inTopo->spheresInGlobal();

		ElementOffset elementOffset = inTopo->calculateElementOffset();

		int numOfSpheres = sphereInGlobal.size();

		auto triSet = this->outTriangleSet()->getDataPtr();

		auto& vertices = triSet->getPoints();
		auto& indices = triSet->triangleIndices();

		auto& sphereVertices = mStandardSphere.getPoints();
		auto& sphereIndices = mStandardSphere.triangleIndices();

		int numOfVertices = sphereVertices.size() * numOfSpheres;
		int numOfTriangles = sphereIndices.size() * numOfSpheres;

		vertices.resize(numOfVertices);
		indices.resize(numOfTriangles);

		uint vertexOffset = 0;
		uint indexOffset = 0;

		//Setup spheres
		cuExecute(numOfSpheres * sphereVertices.size(),
			SetupVerticesForSphereInstances1,
			vertices,
			sphereVertices,
			sphereInGlobal,
			vertexOffset,
			elementOffset.sphereIndex());

		cuExecute(numOfSpheres * sphereIndices.size(),
			SetupIndicesForSphereInstances1,
			indices,
			sphereIndices,
			sphereInGlobal,
			sphereVertices.size(),
			indexOffset);

		vertexOffset += numOfSpheres * sphereVertices.size();
		indexOffset += numOfSpheres * sphereIndices.size();

		this->outTriangleSet()->getDataPtr()->update();

		return true;
	}

	DEFINE_CLASS(DiscreteSpheresToTriangleSet);
}