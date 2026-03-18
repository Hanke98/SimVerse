#include "NeighborMeshLevelQuery.h"

#include "FCallbackFunc.h"
#include "Topology/TopologyConstants.h"

#include <cmath>
#include <limits>
#include <thrust/sort.h>
#include <vector>

namespace
{
	using dyno::EMPTY;

	enum NMLQRegionType
	{
		NMLQ_REGION_INVALID = 0,
		NMLQ_REGION_FACE = 1,
		NMLQ_REGION_EDGE = 2,
		NMLQ_REGION_VERTEX = 3,
	};

	enum NMLQPrimitivePassType
	{
		NMLQ_PASS_TRI0_VERTEX = 0,
		NMLQ_PASS_TRI0_EDGE = 1,
		NMLQ_PASS_TRI1_VERTEX = 2,
		NMLQ_PASS_TRI1_EDGE = 3,
		NMLQ_PASS_COUNT = 4,
	};

	static constexpr unsigned long long NMLQ_EdgePrimitiveKeyMask = 1ull << 63;

	template<typename Real, typename Coord, typename Matrix, typename Triangle, typename Edge, typename Tri2Edg, typename Edg2Tri, typename AABB>
	struct NMLQRuntimeView
	{
		using RealType = Real;
		using CoordType = Coord;
		using MatrixType = Matrix;
		using TriangleType = Triangle;
		using EdgeType = Edge;
		using Tri2EdgType = Tri2Edg;
		using Edg2TriType = Edg2Tri;
		using AABBType = AABB;

		dyno::DArray<Coord> vertices;
		dyno::DArray<Triangle> triangles;
		dyno::DArray<int> tri2Shape;
		dyno::DArray<Matrix> shapeRestR;
		dyno::DArray<Coord> shapeRestT;
		dyno::DArray<Tri2Edg> triangleEdges;
		dyno::DArray<Edge> edgeVertices;
		dyno::DArray<Edg2Tri> edgeAdjacentFaces;
		dyno::DArray<Coord> topoEdgeNormals;
		dyno::DArray<Coord> faceNormalsWorld;
		dyno::DArray<Coord> edgeNormalsWorld;
		dyno::DArray<AABB> triangleAabbsWorld;
		dyno::DArrayList<int> vertexIncidentEdges;
		dyno::DArray<int> faceAssignedVertexOffsets;
		dyno::DArray<int> faceAssignedVertexIndices;
		dyno::DArray<int> faceAssignedEdgeOffsets;
		dyno::DArray<int> faceAssignedEdgeIndices;
		dyno::DArray<dyno::Pair<uint, uint>> patchPairs;
		dyno::DArray<uint> patch2Shape;
		dyno::DArray<int> shape2RigidBody;
		Real dHat = Real(0);
		Real edgeEdgeActivationMargin = Real(0);
		bool verticesInRestWorld = false;
	};

	template<typename View>
	struct NMLQTriPairContext
	{
		using Real = typename View::RealType;

		int tri0 = EMPTY;
		int tri1 = EMPTY;
		int bodyId1 = INVALID;
		int bodyId2 = INVALID;
		int tri0Shape = EMPTY;
		int tri1Shape = EMPTY;
		dyno::TTriangle3D<Real> triangle0;
		dyno::TTriangle3D<Real> triangle1;
	};

	DYN_FUNC inline int NMLQ_GetPairPassSlot(int pairId, int passType)
	{
		return pairId * NMLQ_PASS_COUNT + passType;
	}

	DYN_FUNC inline unsigned long long NMLQ_EncodeVertexPrimitiveKey(int vertexId)
	{
		return static_cast<unsigned long long>(vertexId);
	}

	DYN_FUNC inline unsigned long long NMLQ_EncodeEdgePrimitiveKey(int edgeId)
	{
		return NMLQ_EdgePrimitiveKeyMask | static_cast<unsigned long long>(edgeId);
	}

	DYN_FUNC inline bool NMLQ_IsEdgePrimitiveKey(unsigned long long key)
	{
		return (key & NMLQ_EdgePrimitiveKeyMask) != 0;
	}

	DYN_FUNC inline bool NMLQ_IsPreferredEdgeContactType(
		bool edgePrimitive,
		bool preferEdgeFace,
		dyno::ContactType type)
	{
		if (!edgePrimitive)
			return true;
		return preferEdgeFace
			? type == dyno::ContactType::CT_EDGE_FACE
			: type == dyno::ContactType::CT_EDGE_EDGE;
	}

	template<typename Coord>
	DYN_FUNC Coord NLQ_StablePerpendicular(const Coord& direction)
	{
		using Real = typename Coord::VarType;

		const Real epsSqr = Real(1e-12);
		Coord dir = direction;
		if (dir.normSquared() <= epsSqr)
			return Coord(1, 0, 0);

		dir.normalize();

		const Real ax = dir[0] < Real(0) ? -dir[0] : dir[0];
		const Real ay = dir[1] < Real(0) ? -dir[1] : dir[1];
		const Real az = dir[2] < Real(0) ? -dir[2] : dir[2];

		Coord axis(1, 0, 0);
		if (ay <= ax && ay <= az)
			axis = Coord(0, 1, 0);
		else if (az <= ax && az <= ay)
			axis = Coord(0, 0, 1);

		Coord normal = dir.cross(axis);
		if (normal.normSquared() <= epsSqr)
		{
			axis = axis[0] == Real(1) ? Coord(0, 1, 0) : Coord(1, 0, 0);
			normal = dir.cross(axis);
		}

		if (normal.normSquared() <= epsSqr)
			return Coord(1, 0, 0);

		normal.normalize();
		return normal;
	}

	template<typename Real>
	DYN_FUNC Real NLQ_AbsValue(Real v)
	{
		return v < Real(0) ? -v : v;
	}

	template<typename Coord>
	DYN_FUNC Coord NLQ_BuildRobustFaceNormal(const Coord& p0, const Coord& p1, const Coord& p2)
	{
		using Real = typename Coord::VarType;

		const Real epsSqr = Real(1e-12);
		Coord normal = (p1 - p0).cross(p2 - p0);
		if (normal.normSquared() > epsSqr)
		{
			normal.normalize();
			return normal;
		}

		Coord longestEdge = p1 - p0;
		Real longestEdgeLen = longestEdge.normSquared();

		Coord edge1 = p2 - p1;
		Real edge1Len = edge1.normSquared();
		if (edge1Len > longestEdgeLen)
		{
			longestEdge = edge1;
			longestEdgeLen = edge1Len;
		}

		Coord edge2 = p0 - p2;
		if (edge2.normSquared() > longestEdgeLen)
			longestEdge = edge2;

		return NLQ_StablePerpendicular(longestEdge);
	}

	template<typename Coord>
	DYN_FUNC Coord NLQ_NormalizeOrFallback(const Coord& value, const Coord& fallback)
	{
		using Real = typename Coord::VarType;
		const Real epsSqr = Real(1e-12);

		Coord out = value;
		if (out.normSquared() > epsSqr)
		{
			out.normalize();
			return out;
		}

		out = fallback;
		if (out.normSquared() > epsSqr)
		{
			out.normalize();
			return out;
		}

		return Coord(1, 0, 0);
	}

	inline void NLQ_BuildCSR(
		const std::vector<std::vector<int>>& perFaceItems,
		dyno::DArray<int>& offsets,
		dyno::DArray<int>& indices)
	{
		std::vector<int> hOffsets(perFaceItems.size() + 1, 0);
		size_t totalCount = 0;
		for (size_t i = 0; i < perFaceItems.size(); ++i)
		{
			hOffsets[i] = static_cast<int>(totalCount);
			totalCount += perFaceItems[i].size();
		}
		hOffsets[perFaceItems.size()] = static_cast<int>(totalCount);

		std::vector<int> hIndices;
		hIndices.reserve(totalCount);
		for (const auto& items : perFaceItems)
			hIndices.insert(hIndices.end(), items.begin(), items.end());

		offsets.assign(hOffsets);
		indices.assign(hIndices);
	}

	template<typename View>
	DYN_FUNC bool NMLQ_TransformPoint(
		const View& view,
		const typename View::CoordType& input,
		int shapeId,
		typename View::CoordType& output)
	{
		output = input;
		if (!view.verticesInRestWorld)
			return true;

		if (shapeId < 0 || shapeId >= view.shapeRestR.size() || shapeId >= view.shapeRestT.size())
			return false;

		output = view.shapeRestR[shapeId] * input + view.shapeRestT[shapeId];
		return true;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_GetWorldVertex(
		const View& view,
		int vertexId,
		int shapeId,
		typename View::CoordType& p)
	{
		if (vertexId < 0 || vertexId >= view.vertices.size())
			return false;
		return NMLQ_TransformPoint(view, view.vertices[vertexId], shapeId, p);
	}

	template<typename View>
	DYN_FUNC bool NMLQ_GetWorldTriangle(
		const View& view,
		int triId,
		typename View::CoordType& p0,
		typename View::CoordType& p1,
		typename View::CoordType& p2,
		int* shapeIdOut = nullptr)
	{
		if (triId < 0 || triId >= view.triangles.size())
			return false;

		auto tri = view.triangles[triId];
		if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0
			|| tri[0] >= view.vertices.size() || tri[1] >= view.vertices.size() || tri[2] >= view.vertices.size())
			return false;

		int shapeId = EMPTY;
		if (view.verticesInRestWorld)
		{
			if (triId < 0 || triId >= view.tri2Shape.size())
				return false;
			shapeId = view.tri2Shape[triId];
			if (shapeId < 0 || shapeId >= view.shapeRestR.size() || shapeId >= view.shapeRestT.size())
				return false;
		}

		if (!NMLQ_GetWorldVertex(view, tri[0], shapeId, p0)
			|| !NMLQ_GetWorldVertex(view, tri[1], shapeId, p1)
			|| !NMLQ_GetWorldVertex(view, tri[2], shapeId, p2))
			return false;

		if (shapeIdOut != nullptr)
			*shapeIdOut = shapeId;
		return true;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_GetWorldEdge(
		const View& view,
		int edgeId,
		int shapeId,
		dyno::TSegment3D<typename View::RealType>& segment)
	{
		using Coord = typename View::CoordType;
		if (edgeId < 0 || edgeId >= view.edgeVertices.size())
			return false;

		auto edge = view.edgeVertices[edgeId];
		if (edge[0] < 0 || edge[1] < 0 || edge[0] >= view.vertices.size() || edge[1] >= view.vertices.size())
			return false;

		Coord p0;
		Coord p1;
		if (!NMLQ_GetWorldVertex(view, edge[0], shapeId, p0)
			|| !NMLQ_GetWorldVertex(view, edge[1], shapeId, p1))
			return false;

		segment = dyno::TSegment3D<typename View::RealType>(p0, p1);
		return true;
	}

	template<typename Coord>
	DYN_FUNC Coord NMLQ_TriangleCenter(const Coord& p0, const Coord& p1, const Coord& p2)
	{
		using Real = typename Coord::VarType;
		return (p0 + p1 + p2) / Real(3);
	}

	template<typename View>
	DYN_FUNC int NMLQ_FindLocalVertexInTriangle(
		const View& view,
		int triId,
		int vertexId)
	{
		if (triId < 0 || triId >= view.triangles.size())
			return EMPTY;
		auto tri = view.triangles[triId];
		for (int i = 0; i < 3; ++i)
		{
			if (tri[i] == vertexId)
				return i;
		}
		return EMPTY;
	}

	template<typename Tri2Edg>
	DYN_FUNC bool NMLQ_GetLocalIncidentEdges(
		const Tri2Edg& triEdges,
		int localVertexId,
		int& edge0,
		int& edge1)
	{
		edge0 = EMPTY;
		edge1 = EMPTY;

		switch (localVertexId)
		{
		case 0:
			edge0 = triEdges[0];
			edge1 = triEdges[2];
			return true;
		case 1:
			edge0 = triEdges[0];
			edge1 = triEdges[1];
			return true;
		case 2:
			edge0 = triEdges[1];
			edge1 = triEdges[2];
			return true;
		default:
			return false;
		}
	}

	template<typename Real>
	DYN_FUNC int NMLQ_LocalEdgeIdFromBarycentric(
		Real b0,
		Real b1,
		Real b2)
	{
		if (b0 <= b1 && b0 <= b2)
			return 1;
		if (b1 <= b0 && b1 <= b2)
			return 2;
		return 0;
	}

	template<typename Real>
	DYN_FUNC int NMLQ_LocalVertexIdFromBarycentric(
		Real b0,
		Real b1,
		Real b2)
	{
		if (b0 >= b1 && b0 >= b2)
			return 0;
		if (b1 >= b0 && b1 >= b2)
			return 1;
		return 2;
	}

	template<typename Real>
	DYN_FUNC bool NMLQ_ClassifyTriangleRegion(
		const dyno::TTriangle3D<Real>& triangle,
		const typename dyno::TTriangle3D<Real>::Coord3D& r,
		Real epsBary,
		int& regionType,
		int& localEdgeId,
		int& localVertexId,
		Real bary[3])
	{
		typename dyno::TTriangle3D<Real>::Param param;
		if (!triangle.computeBarycentrics(r, param))
			return false;

		bary[0] = param.u;
		bary[1] = param.v;
		bary[2] = param.w;

		int smallCount = 0;
		if (bary[0] <= epsBary) ++smallCount;
		if (bary[1] <= epsBary) ++smallCount;
		if (bary[2] <= epsBary) ++smallCount;

		localEdgeId = EMPTY;
		localVertexId = EMPTY;
		if (smallCount <= 0)
		{
			regionType = NMLQ_REGION_FACE;
			return true;
		}
		if (smallCount == 1)
		{
			regionType = NMLQ_REGION_EDGE;
			localEdgeId = NMLQ_LocalEdgeIdFromBarycentric(bary[0], bary[1], bary[2]);
			return true;
		}

		regionType = NMLQ_REGION_VERTEX;
		localVertexId = NMLQ_LocalVertexIdFromBarycentric(bary[0], bary[1], bary[2]);
		return true;
	}

	template<typename Real>
	DYN_FUNC bool NMLQ_IsBetterEdgeCandidate(
		Real dist2,
		Real bestDist2,
		Real align,
		Real bestAlign,
		int edgeId,
		int bestEdgeId)
	{
		const Real distEps = Real(1e-9);
		const Real alignEps = Real(1e-6);

		if (bestEdgeId == EMPTY)
			return true;
		if (dist2 + distEps < bestDist2)
			return true;
		if (NLQ_AbsValue(dist2 - bestDist2) <= distEps)
		{
			if (align > bestAlign + alignEps)
				return true;
			if (NLQ_AbsValue(align - bestAlign) <= alignEps && edgeId < bestEdgeId)
				return true;
		}
		return false;
	}

	template<typename Real>
	DYN_FUNC bool NMLQ_IsBetterEdgePairCandidate(
		Real dist2,
		Real bestDist2,
		Real align,
		Real bestAlign,
		int targetEdgeId,
		int bestTargetEdgeId,
		int sourceEdgeId,
		int bestSourceEdgeId)
	{
		const Real distEps = Real(1e-9);
		const Real alignEps = Real(1e-6);

		if (bestTargetEdgeId == EMPTY || bestSourceEdgeId == EMPTY)
			return true;
		if (dist2 + distEps < bestDist2)
			return true;
		if (NLQ_AbsValue(dist2 - bestDist2) <= distEps)
		{
			if (align > bestAlign + alignEps)
				return true;
			if (NLQ_AbsValue(align - bestAlign) <= alignEps)
			{
				if (targetEdgeId < bestTargetEdgeId)
					return true;
				if (targetEdgeId == bestTargetEdgeId && sourceEdgeId < bestSourceEdgeId)
					return true;
			}
		}
		return false;
	}

	template<typename View>
	DYN_FUNC void NMLQ_TryBestSourceEdgeCandidate(
		const View& view,
		int candidateEdgeId,
		int sourceShapeId,
		const dyno::TSegment3D<typename View::RealType>& targetSegment,
		int& bestEdge,
		typename View::RealType& bestDist2,
		typename View::RealType& bestAlign)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsSqr = Real(1e-12);
		if (candidateEdgeId < 0 || candidateEdgeId >= view.edgeVertices.size())
			return;

		dyno::TSegment3D<Real> sourceSegment;
		if (!NMLQ_GetWorldEdge(view, candidateEdgeId, sourceShapeId, sourceSegment))
			return;

		Coord tS = sourceSegment.direction();
		Coord tT = targetSegment.direction();
		if (tS.normSquared() <= epsSqr || tT.normSquared() <= epsSqr)
			return;
		tS.normalize();
		tT.normalize();

		auto pq = sourceSegment.proximity(targetSegment);
		Real dist2 = pq.lengthSquared();
		Real align = NLQ_AbsValue(tS.dot(tT));
		if (NMLQ_IsBetterEdgeCandidate(dist2, bestDist2, align, bestAlign, candidateEdgeId, bestEdge))
		{
			bestEdge = candidateEdgeId;
			bestDist2 = dist2;
			bestAlign = align;
		}
	}

	template<typename View>
	DYN_FUNC void NMLQ_TryBestEdgePairCandidate(
		const View& view,
		int sourceEdgeId,
		int sourceShapeId,
		int targetEdgeId,
		int targetShapeId,
		int& bestSourceEdge,
		int& bestTargetEdge,
		typename View::RealType& bestDist2,
		typename View::RealType& bestAlign)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsSqr = Real(1e-12);
		if (sourceEdgeId < 0 || sourceEdgeId >= view.edgeVertices.size()
			|| targetEdgeId < 0 || targetEdgeId >= view.edgeVertices.size())
			return;

		dyno::TSegment3D<Real> sourceSegment;
		dyno::TSegment3D<Real> targetSegment;
		if (!NMLQ_GetWorldEdge(view, sourceEdgeId, sourceShapeId, sourceSegment)
			|| !NMLQ_GetWorldEdge(view, targetEdgeId, targetShapeId, targetSegment))
			return;

		Coord tS = sourceSegment.direction();
		Coord tT = targetSegment.direction();
		if (tS.normSquared() <= epsSqr || tT.normSquared() <= epsSqr)
			return;
		tS.normalize();
		tT.normalize();

		auto pq = sourceSegment.proximity(targetSegment);
		Real dist2 = pq.lengthSquared();
		Real align = NLQ_AbsValue(tS.dot(tT));
		if (NMLQ_IsBetterEdgePairCandidate(
			dist2,
			bestDist2,
			align,
			bestAlign,
			targetEdgeId,
			bestTargetEdge,
			sourceEdgeId,
			bestSourceEdge))
		{
			bestDist2 = dist2;
			bestAlign = align;
			bestSourceEdge = sourceEdgeId;
			bestTargetEdge = targetEdgeId;
		}
	}

	template<typename View>
	DYN_FUNC void NMLQ_TryBestTargetEdgeCandidate(
		const View& view,
		int candidateTargetEdgeId,
		int targetShapeId,
		const dyno::TSegment3D<typename View::RealType>& sourceSegment,
		int& bestTargetEdge,
		typename View::RealType& bestDist2,
		typename View::RealType& bestAlign)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsSqr = Real(1e-12);
		if (candidateTargetEdgeId < 0 || candidateTargetEdgeId >= view.edgeVertices.size())
			return;

		dyno::TSegment3D<Real> targetSegment;
		if (!NMLQ_GetWorldEdge(view, candidateTargetEdgeId, targetShapeId, targetSegment))
			return;

		Coord tS = sourceSegment.direction();
		Coord tT = targetSegment.direction();
		if (tS.normSquared() <= epsSqr || tT.normSquared() <= epsSqr)
			return;
		tS.normalize();
		tT.normalize();

		auto edgePair = sourceSegment.proximity(targetSegment);
		Real dist2 = edgePair.lengthSquared();
		Real align = NLQ_AbsValue(tS.dot(tT));
		if (NMLQ_IsBetterEdgeCandidate(dist2, bestDist2, align, bestAlign, candidateTargetEdgeId, bestTargetEdge))
		{
			bestTargetEdge = candidateTargetEdgeId;
			bestDist2 = dist2;
			bestAlign = align;
		}
	}

	template<typename View>
	DYN_FUNC int NMLQ_SelectSourceEdgeForTargetEdge(
		const View& view,
		int sourceVertexId,
		int sourceTriId,
		int sourceShapeId,
		int targetEdgeId,
		const dyno::TSegment3D<typename View::RealType>& targetSegment)
	{
		using Real = typename View::RealType;
		int bestEdge = EMPTY;
		Real bestDist2 = std::numeric_limits<Real>::max();
		Real bestAlign = Real(-1);

		if (sourceTriId < 0 || sourceTriId >= view.triangleEdges.size())
			return EMPTY;

		int localVertexId = NMLQ_FindLocalVertexInTriangle(view, sourceTriId, sourceVertexId);
		int edge0 = EMPTY;
		int edge1 = EMPTY;
		if (localVertexId == EMPTY || !NMLQ_GetLocalIncidentEdges(view.triangleEdges[sourceTriId], localVertexId, edge0, edge1))
			return EMPTY;

		NMLQ_TryBestSourceEdgeCandidate(view, edge0, sourceShapeId, targetSegment, bestEdge, bestDist2, bestAlign);
		NMLQ_TryBestSourceEdgeCandidate(view, edge1, sourceShapeId, targetSegment, bestEdge, bestDist2, bestAlign);

		return bestEdge;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_SelectEdgePairFromVertices(
		const View& view,
		int sourceVertexId,
		int sourceTriId,
		int sourceShapeId,
		int targetVertexId,
		int targetTriId,
		int targetShapeId,
		int& bestSourceEdge,
		int& bestTargetEdge)
	{
		using Real = typename View::RealType;
		Real bestDist2 = std::numeric_limits<Real>::max();
		Real bestAlign = Real(-1);
		bestSourceEdge = EMPTY;
		bestTargetEdge = EMPTY;

		if (sourceTriId < 0 || sourceTriId >= view.triangleEdges.size()
			|| targetTriId < 0 || targetTriId >= view.triangleEdges.size())
			return false;

		int sourceLocalVertexId = NMLQ_FindLocalVertexInTriangle(view, sourceTriId, sourceVertexId);
		int targetLocalVertexId = NMLQ_FindLocalVertexInTriangle(view, targetTriId, targetVertexId);
		int e0 = EMPTY;
		int e1 = EMPTY;
		int te0 = EMPTY;
		int te1 = EMPTY;
		if (sourceLocalVertexId == EMPTY || targetLocalVertexId == EMPTY
			|| !NMLQ_GetLocalIncidentEdges(view.triangleEdges[sourceTriId], sourceLocalVertexId, e0, e1)
			|| !NMLQ_GetLocalIncidentEdges(view.triangleEdges[targetTriId], targetLocalVertexId, te0, te1))
			return false;

		NMLQ_TryBestEdgePairCandidate(view, e0, sourceShapeId, te0, targetShapeId, bestSourceEdge, bestTargetEdge, bestDist2, bestAlign);
		NMLQ_TryBestEdgePairCandidate(view, e0, sourceShapeId, te1, targetShapeId, bestSourceEdge, bestTargetEdge, bestDist2, bestAlign);
		NMLQ_TryBestEdgePairCandidate(view, e1, sourceShapeId, te0, targetShapeId, bestSourceEdge, bestTargetEdge, bestDist2, bestAlign);
		NMLQ_TryBestEdgePairCandidate(view, e1, sourceShapeId, te1, targetShapeId, bestSourceEdge, bestTargetEdge, bestDist2, bestAlign);

		return bestSourceEdge != EMPTY && bestTargetEdge != EMPTY;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_GetBodyIdsForPair(
		const View& view,
		int pairId,
		int& bodyId1,
		int& bodyId2,
		int& shape0,
		int& shape1)
	{
		if (pairId < 0 || pairId >= view.patchPairs.size())
			return false;

		auto pair = view.patchPairs[pairId];
		int patch0 = static_cast<int>(pair.first);
		int patch1 = static_cast<int>(pair.second);
		if (patch0 < 0 || patch1 < 0 || patch0 >= view.patch2Shape.size() || patch1 >= view.patch2Shape.size())
			return false;

		shape0 = static_cast<int>(view.patch2Shape[patch0]);
		shape1 = static_cast<int>(view.patch2Shape[patch1]);
		if (shape0 < 0 || shape1 < 0 || shape0 >= view.shape2RigidBody.size() || shape1 >= view.shape2RigidBody.size())
			return false;

		bodyId1 = view.shape2RigidBody[shape0];
		bodyId2 = view.shape2RigidBody[shape1];
		return true;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_BuildEdgeEdgeContact(
		const View& view,
		int sourceEdgeId,
		int sourceShapeId,
		int targetEdgeId,
		int targetShapeId,
		typename View::CoordType& contactPoint,
		typename View::CoordType& nTarget,
		typename View::RealType& depth)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsSqr = Real(1e-12);
		dyno::TSegment3D<Real> sourceSegment;
		dyno::TSegment3D<Real> targetSegment;
		if (!NMLQ_GetWorldEdge(view, sourceEdgeId, sourceShapeId, sourceSegment)
			|| !NMLQ_GetWorldEdge(view, targetEdgeId, targetShapeId, targetSegment))
			return false;

		Coord sourceDir = sourceSegment.direction();
		Coord targetDir = targetSegment.direction();
		if (sourceDir.normSquared() <= epsSqr || targetDir.normSquared() <= epsSqr)
			return false;

			auto pq = sourceSegment.proximity(targetSegment);
			Coord cSource = pq.startPoint();
			Coord cTarget = pq.endPoint();
			Coord pqVec = cTarget - cSource;
			Real gap = pqVec.norm();
			Real edgeEdgeActivationMargin = view.edgeEdgeActivationMargin + view.dHat;
			if (edgeEdgeActivationMargin < Real(0))
				edgeEdgeActivationMargin = Real(0);
			// Activate near edge-edge pairs using the dedicated margin plus the existing shell thickness.
			if (gap > edgeEdgeActivationMargin)
				return false;

		if (targetEdgeId < 0 || targetEdgeId >= view.edgeNormalsWorld.size())
			return false;

		Coord nTargetEdge = view.edgeNormalsWorld[targetEdgeId];
		if (nTargetEdge.normSquared() <= epsSqr)
			return false;
		nTargetEdge.normalize();

		// Only keep the pair when pq lies on the positive side of the target edge normal.
		if (nTargetEdge.dot(pqVec) <= Real(0))
			return false;

		sourceDir.normalize();
		targetDir.normalize();
		nTarget = sourceDir.cross(targetDir);
		if (nTarget.normSquared() <= epsSqr)
			return false;
		nTarget.normalize();

		// Orient the final normal to the same side as the target edge normal.
		if (nTarget.dot(nTargetEdge) <= Real(0))
			nTarget = -nTarget;

		contactPoint = Real(0.5) * (cSource + cTarget);
		// Use shell penetration so separated edge pairs do not inject a positive depth.
		depth = view.dHat - gap;
		if (depth < Real(0))
			depth = Real(0);
		return true;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_TryVertexTriangleContact(
		const View& view,
		int sourceTriId,
		int sourceShapeId,
		int sourceVertexId,
		const dyno::TTriangle3D<typename View::RealType>& sourceTriangle,
		int targetTriId,
		int targetShapeId,
		const dyno::TTriangle3D<typename View::RealType>& targetTriangle,
		typename View::CoordType& contactPoint,
		typename View::CoordType& nTarget,
		typename View::RealType& depth,
		dyno::ContactType& contactType)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsBary = Real(1e-5);
		Coord p;
		if (!NMLQ_GetWorldVertex(view, sourceVertexId, sourceShapeId, p))
			return false;

		Coord r = dyno::TPoint3D<Real>(p).project(targetTriangle).origin;
		int regionType = NMLQ_REGION_INVALID;
		int localEdgeId = EMPTY;
		int localVertexId = EMPTY;
		Real bary[3] = { Real(0), Real(0), Real(0) };
		if (!NMLQ_ClassifyTriangleRegion(targetTriangle, r, epsBary, regionType, localEdgeId, localVertexId, bary))
			return false;

		if (regionType == NMLQ_REGION_FACE)
		{
			Coord faceNormal = targetTriId >= 0 && targetTriId < view.faceNormalsWorld.size()
				? view.faceNormalsWorld[targetTriId]
				: NLQ_BuildRobustFaceNormal(targetTriangle.v[0], targetTriangle.v[1], targetTriangle.v[2]);
			nTarget = NLQ_NormalizeOrFallback(faceNormal, NLQ_StablePerpendicular(targetTriangle.v[1] - targetTriangle.v[0]));

			Real signedDistance = (p - targetTriangle.v[0]).dot(nTarget);
			if (signedDistance > view.dHat)
				return false;

			contactPoint = r;
			depth = signedDistance < Real(0) ? -signedDistance : Real(0);
			contactType = dyno::ContactType::CT_VERTEX_FACE;
			return true;
		}

		if (regionType == NMLQ_REGION_EDGE)
		{
			// Let the dedicated edge passes handle edge-origin contacts.
			return false;
		}

		if (regionType == NMLQ_REGION_VERTEX)
		{
			// Let the dedicated edge passes handle edge-origin contacts.
			return false;
		}

		return false;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_TryEdgeTriangleContact(
		const View& view,
		int sourceTriId,
		int sourceShapeId,
		int sourceEdgeId,
		int targetTriId,
		int targetShapeId,
		const dyno::TTriangle3D<typename View::RealType>& targetTriangle,
		typename View::CoordType& contactPoint,
		typename View::CoordType& nTarget,
		typename View::RealType& depth,
		dyno::ContactType& contactType)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		const Real epsBary = Real(1e-5);
		dyno::TSegment3D<Real> sourceSegment;
		if (!NMLQ_GetWorldEdge(view, sourceEdgeId, sourceShapeId, sourceSegment))
			return false;

		(void)sourceTriId;

		auto pq = sourceSegment.proximity(targetTriangle);
		Coord cTarget = pq.endPoint();
		int regionType = NMLQ_REGION_INVALID;
		int localEdgeId = EMPTY;
		int localVertexId = EMPTY;
		Real bary[3] = { Real(0), Real(0), Real(0) };
		if (!NMLQ_ClassifyTriangleRegion(targetTriangle, cTarget, epsBary, regionType, localEdgeId, localVertexId, bary))
			return false;

		if (regionType == NMLQ_REGION_FACE)
		{
			Coord faceNormal = targetTriId >= 0 && targetTriId < view.faceNormalsWorld.size()
				? view.faceNormalsWorld[targetTriId]
				: NLQ_BuildRobustFaceNormal(targetTriangle.v[0], targetTriangle.v[1], targetTriangle.v[2]);
			nTarget = NLQ_NormalizeOrFallback(faceNormal, NLQ_StablePerpendicular(targetTriangle.v[1] - targetTriangle.v[0]));

			Coord p0 = sourceSegment.startPoint();
			Coord p1 = sourceSegment.endPoint();
			Real d0 = (p0 - targetTriangle.v[0]).dot(nTarget);
			Real d1 = (p1 - targetTriangle.v[0]).dot(nTarget);
			Real minSignedDistance = d0 < d1 ? d0 : d1;
			Real edgeActivation = view.edgeEdgeActivationMargin + view.dHat;
			if (edgeActivation < Real(0))
				edgeActivation = Real(0);

			// Prefer a face-supported contact once the edge enters the face shell.
			if (minSignedDistance > edgeActivation)
				return false;

			contactPoint = cTarget;
			// Edge-face only contributes activation, normal, and point; vertex-face carries penetration depth.
			depth = Real(0);
			contactType = dyno::ContactType::CT_EDGE_FACE;
			return true;
		}

		if (targetTriId < 0 || targetTriId >= view.triangleEdges.size())
			return false;

		if (regionType == NMLQ_REGION_EDGE)
		{
			int targetEdgeId = view.triangleEdges[targetTriId][localEdgeId];
			if (targetEdgeId == EMPTY)
				return false;

			if (!NMLQ_BuildEdgeEdgeContact(view, sourceEdgeId, sourceShapeId, targetEdgeId, targetShapeId, contactPoint, nTarget, depth))
				return false;
			contactType = dyno::ContactType::CT_EDGE_EDGE;
			return true;
		}

		if (regionType == NMLQ_REGION_VERTEX)
		{
			int bestTargetEdge = EMPTY;
			Real bestDist2 = std::numeric_limits<Real>::max();
			Real bestAlign = Real(-1);
			int edge0 = EMPTY;
			int edge1 = EMPTY;
			if (!NMLQ_GetLocalIncidentEdges(view.triangleEdges[targetTriId], localVertexId, edge0, edge1))
				return false;

			NMLQ_TryBestTargetEdgeCandidate(view, edge0, targetShapeId, sourceSegment, bestTargetEdge, bestDist2, bestAlign);
			NMLQ_TryBestTargetEdgeCandidate(view, edge1, targetShapeId, sourceSegment, bestTargetEdge, bestDist2, bestAlign);

			if (bestTargetEdge == EMPTY)
				return false;

			if (!NMLQ_BuildEdgeEdgeContact(view, sourceEdgeId, sourceShapeId, bestTargetEdge, targetShapeId, contactPoint, nTarget, depth))
				return false;
			contactType = dyno::ContactType::CT_EDGE_EDGE;
			return true;
		}

		return false;
	}

	template<typename ContactPair, typename Coord, typename Real>
	DYN_FUNC void NMLQ_WriteContact(
		ContactPair& contact,
		int bodyId1,
		int bodyId2,
		int tri0,
		int tri1,
		const Coord& contactPoint,
		const Coord& nTarget,
		bool targetIsTri1,
		Real depth,
		dyno::ContactType type)
	{
		contact.bodyId1 = bodyId1;
		contact.bodyId2 = bodyId2;
		contact.localId1 = tri0;
		contact.localId2 = tri1;
		contact.pos1 = contactPoint;
		contact.pos2 = contactPoint;
		if (targetIsTri1)
		{
			contact.normal2 = -nTarget;
			contact.normal1 = nTarget;
		}
		else
		{
			contact.normal1 = -nTarget;
			contact.normal2 = nTarget;
		}
		contact.contactType = type;
		contact.interpenetration = depth < Real(0) ? Real(0) : depth;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_BuildTriPairContext(
		const View& view,
		int tri0,
		int tri1,
		int pairId,
		NMLQTriPairContext<View>& ctx)
	{
		using Real = typename View::RealType;
		using Coord = typename View::CoordType;

		int shape0 = EMPTY;
		int shape1 = EMPTY;
		if (!NMLQ_GetBodyIdsForPair(view, pairId, ctx.bodyId1, ctx.bodyId2, shape0, shape1))
			return false;

		Coord tri0p0, tri0p1, tri0p2;
		Coord tri1p0, tri1p1, tri1p2;
		if (!NMLQ_GetWorldTriangle(view, tri0, tri0p0, tri0p1, tri0p2, &ctx.tri0Shape)
			|| !NMLQ_GetWorldTriangle(view, tri1, tri1p0, tri1p1, tri1p2, &ctx.tri1Shape))
			return false;

		ctx.tri0 = tri0;
		ctx.tri1 = tri1;
		ctx.triangle0 = dyno::TTriangle3D<Real>(tri0p0, tri0p1, tri0p2);
		ctx.triangle1 = dyno::TTriangle3D<Real>(tri1p0, tri1p1, tri1p2);
		return true;
	}

	template<typename View>
	DYN_FUNC bool NMLQ_GetPrimitivePassContext(
		const NMLQTriPairContext<View>& ctx,
		int passType,
		int& sourceTriId,
		int& sourceShapeId,
		const dyno::TTriangle3D<typename View::RealType>*& sourceTriangle,
		int& targetTriId,
		int& targetShapeId,
		const dyno::TTriangle3D<typename View::RealType>*& targetTriangle,
		bool& targetIsTri1,
		bool& vertexPass)
	{
		switch (passType)
		{
		case NMLQ_PASS_TRI0_VERTEX:
			sourceTriId = ctx.tri0;
			sourceShapeId = ctx.tri0Shape;
			sourceTriangle = &ctx.triangle0;
			targetTriId = ctx.tri1;
			targetShapeId = ctx.tri1Shape;
			targetTriangle = &ctx.triangle1;
			targetIsTri1 = true;
			vertexPass = true;
			return true;
		case NMLQ_PASS_TRI0_EDGE:
			sourceTriId = ctx.tri0;
			sourceShapeId = ctx.tri0Shape;
			sourceTriangle = &ctx.triangle0;
			targetTriId = ctx.tri1;
			targetShapeId = ctx.tri1Shape;
			targetTriangle = &ctx.triangle1;
			targetIsTri1 = true;
			vertexPass = false;
			return true;
		case NMLQ_PASS_TRI1_VERTEX:
			sourceTriId = ctx.tri1;
			sourceShapeId = ctx.tri1Shape;
			sourceTriangle = &ctx.triangle1;
			targetTriId = ctx.tri0;
			targetShapeId = ctx.tri0Shape;
			targetTriangle = &ctx.triangle0;
			targetIsTri1 = false;
			vertexPass = true;
			return true;
		case NMLQ_PASS_TRI1_EDGE:
			sourceTriId = ctx.tri1;
			sourceShapeId = ctx.tri1Shape;
			sourceTriangle = &ctx.triangle1;
			targetTriId = ctx.tri0;
			targetShapeId = ctx.tri0Shape;
			targetTriangle = &ctx.triangle0;
			targetIsTri1 = false;
			vertexPass = false;
			return true;
		default:
			break;
		}

		sourceTriId = EMPTY;
		sourceShapeId = EMPTY;
		sourceTriangle = nullptr;
		targetTriId = EMPTY;
		targetShapeId = EMPTY;
		targetTriangle = nullptr;
		targetIsTri1 = true;
		vertexPass = true;
		return false;
	}

	template<typename View, typename ContactPair>
	DYN_FUNC int NMLQ_ProcessPrimitivePass(
		const View& view,
		const NMLQTriPairContext<View>& ctx,
		int passType,
		ContactPair* contacts,
		unsigned long long* primitiveKeys,
		int contactsSize,
		int writeBase,
		bool write)
	{
		int sourceTriId = EMPTY;
		int sourceShapeId = EMPTY;
		int targetTriId = EMPTY;
		int targetShapeId = EMPTY;
		const dyno::TTriangle3D<typename View::RealType>* sourceTriangle = nullptr;
		const dyno::TTriangle3D<typename View::RealType>* targetTriangle = nullptr;
		bool targetIsTri1 = true;
		bool vertexPass = true;
		if (!NMLQ_GetPrimitivePassContext(
			ctx,
			passType,
			sourceTriId,
			sourceShapeId,
			sourceTriangle,
			targetTriId,
			targetShapeId,
			targetTriangle,
			targetIsTri1,
			vertexPass))
			return 0;

		int count = 0;
		if (vertexPass)
		{
			if (sourceTriId < 0 || sourceTriId >= view.triangles.size())
				return 0;

			auto sourceTriIndices = view.triangles[sourceTriId];
			for (int localVertexId = 0; localVertexId < 3; ++localVertexId)
			{
				int vertexId = sourceTriIndices[localVertexId];
				if (vertexId < 0 || vertexId >= view.vertices.size())
					continue;

				typename View::CoordType contactPoint;
				typename View::CoordType nTarget;
				typename View::RealType depth = typename View::RealType(0);
				dyno::ContactType type = dyno::ContactType::CT_UNKNOWN;
				if (!NMLQ_TryVertexTriangleContact(
					view,
					sourceTriId,
					sourceShapeId,
					vertexId,
					*sourceTriangle,
					targetTriId,
					targetShapeId,
					*targetTriangle,
					contactPoint,
					nTarget,
					depth,
					type))
					continue;

				if (write && contacts != nullptr && primitiveKeys != nullptr)
				{
					int outIdx = writeBase + count;
					if (outIdx >= 0 && outIdx < contactsSize)
					{
						ContactPair cp;
						NMLQ_WriteContact(cp, ctx.bodyId1, ctx.bodyId2, ctx.tri0, ctx.tri1, contactPoint, nTarget, targetIsTri1, depth, type);
						contacts[outIdx] = cp;
						primitiveKeys[outIdx] = NMLQ_EncodeVertexPrimitiveKey(vertexId);
					}
				}

				++count;
			}
			return count;
		}

		if (sourceTriId < 0 || sourceTriId >= view.triangleEdges.size())
			return 0;

		auto sourceTriEdges = view.triangleEdges[sourceTriId];
		for (int localEdgeId = 0; localEdgeId < 3; ++localEdgeId)
		{
			int edgeId = sourceTriEdges[localEdgeId];
			if (edgeId == EMPTY)
				continue;

			typename View::CoordType contactPoint;
			typename View::CoordType nTarget;
			typename View::RealType depth = typename View::RealType(0);
			dyno::ContactType type = dyno::ContactType::CT_UNKNOWN;
			if (!NMLQ_TryEdgeTriangleContact(
				view,
				sourceTriId,
				sourceShapeId,
				edgeId,
				targetTriId,
				targetShapeId,
				*targetTriangle,
				contactPoint,
				nTarget,
				depth,
				type))
				continue;

			if (write && contacts != nullptr && primitiveKeys != nullptr)
			{
				int outIdx = writeBase + count;
				if (outIdx >= 0 && outIdx < contactsSize)
				{
					ContactPair cp;
					NMLQ_WriteContact(cp, ctx.bodyId1, ctx.bodyId2, ctx.tri0, ctx.tri1, contactPoint, nTarget, targetIsTri1, depth, type);
					contacts[outIdx] = cp;
					primitiveKeys[outIdx] = NMLQ_EncodeEdgePrimitiveKey(edgeId);
				}
			}

			++count;
		}
		return count;
	}

	template<typename View>
	__global__ void NMLQ_PrepareTriangleWorldData(View view)
	{
		using Coord = typename View::CoordType;
		using Real = typename View::RealType;

		int triId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (triId >= view.triangles.size() || triId >= view.triangleAabbsWorld.size() || triId >= view.faceNormalsWorld.size())
			return;

		Coord p0, p1, p2;
		if (!NMLQ_GetWorldTriangle(view, triId, p0, p1, p2, nullptr))
		{
			view.triangleAabbsWorld[triId] = typename View::AABBType(Coord(0), Coord(0));
			view.faceNormalsWorld[triId] = Coord(1, 0, 0);
			return;
		}

		typename View::AABBType box;
		box.v0 = p0.minimum(p1).minimum(p2);
		box.v1 = p0.maximum(p1).maximum(p2);
		view.triangleAabbsWorld[triId] = box;
		dyno::TTriangle3D<Real> tri(p0, p1, p2);
		view.faceNormalsWorld[triId] = tri.normal();
	}

	template<typename View>
	__global__ void NMLQ_PrepareEdgeNormalsWorld(View view)
	{
		using Coord = typename View::CoordType;
		using Real = typename View::RealType;

		int edgeId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (edgeId >= view.edgeVertices.size() || edgeId >= view.edgeNormalsWorld.size())
			return;

		const Real epsSqr = Real(1e-12);
		Coord edgeNormal(0);

		if (edgeId < view.edgeAdjacentFaces.size())
		{
			auto adjacentFaces = view.edgeAdjacentFaces[edgeId];
			int face0 = adjacentFaces[0];
			int face1 = adjacentFaces[1];
			if (face0 != EMPTY && face0 < view.faceNormalsWorld.size()
				&& face1 != EMPTY && face1 < view.faceNormalsWorld.size())
			{
				edgeNormal = view.faceNormalsWorld[face0] + view.faceNormalsWorld[face1];
				if (edgeNormal.normSquared() > epsSqr)
					edgeNormal.normalize();
				else
					edgeNormal = view.faceNormalsWorld[face0];
			}
			else if (face0 != EMPTY && face0 < view.faceNormalsWorld.size())
			{
				edgeNormal = view.faceNormalsWorld[face0];
			}
			else if (face1 != EMPTY && face1 < view.faceNormalsWorld.size())
			{
				edgeNormal = view.faceNormalsWorld[face1];
			}

			if (edgeNormal.normSquared() <= epsSqr)
			{
				int shapeHint = EMPTY;
				int faceHint = face0 != EMPTY ? face0 : face1;
				if (view.verticesInRestWorld && faceHint != EMPTY && faceHint >= 0 && faceHint < view.tri2Shape.size())
					shapeHint = view.tri2Shape[faceHint];

				dyno::TSegment3D<Real> edgeSegment;
				if (NMLQ_GetWorldEdge(view, edgeId, shapeHint, edgeSegment))
					edgeNormal = NLQ_StablePerpendicular(edgeSegment.direction());
			}
		}

		if (edgeNormal.normSquared() <= epsSqr && edgeId < view.topoEdgeNormals.size())
			edgeNormal = view.topoEdgeNormals[edgeId];

		view.edgeNormalsWorld[edgeId] = NLQ_NormalizeOrFallback(edgeNormal, Coord(1, 0, 0));
	}

	__global__ void NMLQ_CountTriPairsPerPatchPair(
		dyno::DArray<int> counts,
		dyno::DArray<dyno::Pair<uint, uint>> patchPairs,
		dyno::DArray<int> patch2TriOffsets)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= patchPairs.size() || pairId >= counts.size())
			return;

		auto pair = patchPairs[pairId];
		int patch0 = static_cast<int>(pair.first);
		int patch1 = static_cast<int>(pair.second);
		if (patch0 < 0 || patch1 < 0 || patch0 + 1 >= patch2TriOffsets.size() || patch1 + 1 >= patch2TriOffsets.size())
		{
			counts[pairId] = 0;
			return;
		}

		int count0 = patch2TriOffsets[patch0 + 1] - patch2TriOffsets[patch0];
		int count1 = patch2TriOffsets[patch1 + 1] - patch2TriOffsets[patch1];
		count0 = count0 > 0 ? count0 : 0;
		count1 = count1 > 0 ? count1 : 0;
		counts[pairId] = count0 * count1;
	}

	__global__ void NMLQ_SetTriPairs(
		dyno::DArray<int> tri0Out,
		dyno::DArray<int> tri1Out,
		dyno::DArray<int> patchPairIdOut,
		dyno::DArray<int> offsets,
		dyno::DArray<int> counts,
		dyno::DArray<dyno::Pair<uint, uint>> patchPairs,
		dyno::DArray<int> patch2TriOffsets,
		dyno::DArray<int> patch2TriIndices)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= patchPairs.size() || pairId >= offsets.size() || pairId >= counts.size())
			return;

		int count = counts[pairId];
		if (count <= 0)
			return;

		auto pair = patchPairs[pairId];
		int patch0 = static_cast<int>(pair.first);
		int patch1 = static_cast<int>(pair.second);
		if (patch0 < 0 || patch1 < 0 || patch0 + 1 >= patch2TriOffsets.size() || patch1 + 1 >= patch2TriOffsets.size())
			return;

		int begin0 = patch2TriOffsets[patch0];
		int end0 = patch2TriOffsets[patch0 + 1];
		int begin1 = patch2TriOffsets[patch1];
		int end1 = patch2TriOffsets[patch1 + 1];
		int count0 = end0 - begin0;
		int count1 = end1 - begin1;
		if (count0 <= 0 || count1 <= 0)
			return;

		int base = offsets[pairId];
		for (int i = 0; i < count0; ++i)
		{
			int tri0 = patch2TriIndices[begin0 + i];
			for (int j = 0; j < count1; ++j)
			{
				int outIdx = base + i * count1 + j;
				if (outIdx >= tri0Out.size() || outIdx >= tri1Out.size() || outIdx >= patchPairIdOut.size())
					return;

				tri0Out[outIdx] = tri0;
				tri1Out[outIdx] = patch2TriIndices[begin1 + j];
				patchPairIdOut[outIdx] = pairId;
			}
		}
	}

	template<typename AABB, typename Real>
	__global__ void NMLQ_CountCoarsePassedTriPairs(
		dyno::DArray<int> counts,
		dyno::DArray<int> tri0,
		dyno::DArray<int> tri1,
		dyno::DArray<AABB> triangleAabbs,
		Real dHat)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= counts.size() || pairId >= tri0.size() || pairId >= tri1.size())
			return;

		int t0 = tri0[pairId];
		int t1 = tri1[pairId];
		if (t0 < 0 || t1 < 0 || t0 >= triangleAabbs.size() || t1 >= triangleAabbs.size())
		{
			counts[pairId] = 0;
			return;
		}

		AABB box0 = triangleAabbs[t0];
		AABB box1 = triangleAabbs[t1];
		auto expandVec = typename AABB::Coord3D(dHat, dHat, dHat);
		box0.v0 -= expandVec;
		box0.v1 += expandVec;
		box1.v0 -= expandVec;
		box1.v1 += expandVec;

		counts[pairId] = box0.checkOverlap(box1) ? 1 : 0;
	}

	__global__ void NMLQ_SetCoarsePassedTriPairs(
		dyno::DArray<int> filteredTri0,
		dyno::DArray<int> filteredTri1,
		dyno::DArray<int> filteredPatchPairId,
		dyno::DArray<int> tri0,
		dyno::DArray<int> tri1,
		dyno::DArray<int> patchPairId,
		dyno::DArray<int> offsets,
		dyno::DArray<int> counts)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= counts.size() || pairId >= offsets.size())
			return;

		if (counts[pairId] <= 0)
			return;

		int outIdx = offsets[pairId];
		if (outIdx < 0 || outIdx >= filteredTri0.size() || outIdx >= filteredTri1.size() || outIdx >= filteredPatchPairId.size())
			return;

		filteredTri0[outIdx] = tri0[pairId];
		filteredTri1[outIdx] = tri1[pairId];
		filteredPatchPairId[outIdx] = patchPairId[pairId];
	}

	template<typename View>
	__global__ void NMLQ_CountPrimitiveCandidatesPerPass(
		dyno::DArray<int> primitivePassCounts,
		dyno::DArray<int> filteredTri0,
		dyno::DArray<int> filteredTri1,
		dyno::DArray<int> filteredPatchPairId,
		View view)
	{
		int slotId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (slotId >= primitivePassCounts.size())
			return;

			int pairId = slotId / NMLQ_PASS_COUNT;
			int passType = slotId % NMLQ_PASS_COUNT;

			if (pairId >= filteredTri0.size() || pairId >= filteredTri1.size() || pairId >= filteredPatchPairId.size())
			{
			primitivePassCounts[slotId] = 0;
			return;
		}

		NMLQTriPairContext<View> ctx;
		if (!NMLQ_BuildTriPairContext(view, filteredTri0[pairId], filteredTri1[pairId], filteredPatchPairId[pairId], ctx))
		{
			primitivePassCounts[slotId] = 0;
			return;
		}

		primitivePassCounts[slotId] = NMLQ_ProcessPrimitivePass(
			view,
			ctx,
			passType,
			(dyno::TContactPair<typename View::RealType>*)nullptr,
			nullptr,
			0,
			0,
			false);
	}

	template<typename View, typename ContactPair>
	__global__ void NMLQ_SetPrimitiveCandidatesPerPass(
		dyno::DArray<ContactPair> primitiveCandidateContacts,
		dyno::DArray<unsigned long long> primitiveCandidateKeys,
		dyno::DArray<int> primitivePassOffsets,
		dyno::DArray<int> primitivePassCounts,
		dyno::DArray<int> filteredTri0,
		dyno::DArray<int> filteredTri1,
		dyno::DArray<int> filteredPatchPairId,
		View view)
	{
		int slotId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (slotId >= primitivePassOffsets.size() || slotId >= primitivePassCounts.size())
			return;

			int count = primitivePassCounts[slotId];
			if (count <= 0)
				return;

			int pairId = slotId / NMLQ_PASS_COUNT;
			int passType = slotId % NMLQ_PASS_COUNT;

			if (pairId >= filteredTri0.size() || pairId >= filteredTri1.size() || pairId >= filteredPatchPairId.size())
				return;

		NMLQTriPairContext<View> ctx;
		if (!NMLQ_BuildTriPairContext(view, filteredTri0[pairId], filteredTri1[pairId], filteredPatchPairId[pairId], ctx))
			return;

		NMLQ_ProcessPrimitivePass(
			view,
			ctx,
			passType,
			primitiveCandidateContacts.begin(),
			primitiveCandidateKeys.begin(),
			primitiveCandidateContacts.size(),
			primitivePassOffsets[slotId],
			true);
	}

	__global__ void NMLQ_InitPrimitiveCandidateIndices(
		dyno::DArray<int> primitiveCandidateSortedIndices)
	{
		int idx = threadIdx.x + (blockIdx.x * blockDim.x);
		if (idx >= primitiveCandidateSortedIndices.size())
			return;
		primitiveCandidateSortedIndices[idx] = idx;
	}

	template<typename ContactPair, typename Real>
	__global__ void NMLQ_MarkMinDepthCandidatesPerPrimitiveKey(
		dyno::DArray<int> primitiveCandidateKeepFlags,
		dyno::DArray<unsigned long long> primitiveCandidateKeys,
		dyno::DArray<int> primitiveCandidateSortedIndices,
		dyno::DArray<ContactPair> primitiveCandidateContacts,
		Real depthTieEps,
		Real sameDirectionDotEps)
	{
		int sortedIdx = threadIdx.x + (blockIdx.x * blockDim.x);
		if (sortedIdx >= primitiveCandidateKeys.size() || sortedIdx >= primitiveCandidateSortedIndices.size())
			return;

		if (sortedIdx > 0 && primitiveCandidateKeys[sortedIdx - 1] == primitiveCandidateKeys[sortedIdx])
			return;

		unsigned long long key = primitiveCandidateKeys[sortedIdx];
		int groupEnd = sortedIdx;
		const bool edgePrimitive = NMLQ_IsEdgePrimitiveKey(key);
		bool preferEdgeFace = false;
		if (edgePrimitive)
		{
			for (int i = sortedIdx; i < primitiveCandidateKeys.size() && primitiveCandidateKeys[i] == key; ++i)
			{
				int rawIdx = primitiveCandidateSortedIndices[i];
				if (rawIdx >= 0 && rawIdx < primitiveCandidateContacts.size()
					&& primitiveCandidateContacts[rawIdx].contactType == dyno::ContactType::CT_EDGE_FACE)
				{
					preferEdgeFace = true;
					break;
				}
			}
		}

		Real minDepth = std::numeric_limits<Real>::max();
		while (groupEnd < primitiveCandidateKeys.size() && primitiveCandidateKeys[groupEnd] == key)
		{
			int rawIdx = primitiveCandidateSortedIndices[groupEnd];
			if (rawIdx >= 0 && rawIdx < primitiveCandidateContacts.size())
			{
				dyno::ContactType type = primitiveCandidateContacts[rawIdx].contactType;
				if (NMLQ_IsPreferredEdgeContactType(edgePrimitive, preferEdgeFace, type))
				{
					Real depth = primitiveCandidateContacts[rawIdx].interpenetration;
					if (depth < minDepth)
						minDepth = depth;
				}
			}
			++groupEnd;
		}

		if (minDepth == std::numeric_limits<Real>::max())
			return;

		for (int i = sortedIdx; i < groupEnd; ++i)
		{
			int rawIdx = primitiveCandidateSortedIndices[i];
			if (rawIdx < 0 || rawIdx >= primitiveCandidateKeepFlags.size() || rawIdx >= primitiveCandidateContacts.size())
				continue;

			dyno::ContactType type = primitiveCandidateContacts[rawIdx].contactType;
			if (!NMLQ_IsPreferredEdgeContactType(edgePrimitive, preferEdgeFace, type))
				continue;

			Real depth = primitiveCandidateContacts[rawIdx].interpenetration;
			if (depth > minDepth + depthTieEps)
				continue;

			auto direction = primitiveCandidateContacts[rawIdx].normal1;
			const Real dirNorm2 = direction.normSquared();
			if (dirNorm2 > Real(1e-12))
			{
				direction /= sqrt(dirNorm2);
				bool duplicateDirection = false;
				for (int j = sortedIdx; j < i; ++j)
				{
					int prevRawIdx = primitiveCandidateSortedIndices[j];
					if (prevRawIdx < 0 || prevRawIdx >= primitiveCandidateKeepFlags.size()
						|| prevRawIdx >= primitiveCandidateContacts.size()
						|| primitiveCandidateKeepFlags[prevRawIdx] <= 0)
						continue;

					dyno::ContactType prevType = primitiveCandidateContacts[prevRawIdx].contactType;
					if (!NMLQ_IsPreferredEdgeContactType(edgePrimitive, preferEdgeFace, prevType))
						continue;

					Real prevDepth = primitiveCandidateContacts[prevRawIdx].interpenetration;
					if (prevDepth > minDepth + depthTieEps)
						continue;

					auto prevDirection = primitiveCandidateContacts[prevRawIdx].normal1;
					const Real prevNorm2 = prevDirection.normSquared();
					if (prevNorm2 <= Real(1e-12))
						continue;

					prevDirection /= sqrt(prevNorm2);
					if (direction.dot(prevDirection) >= Real(1) - sameDirectionDotEps)
					{
						duplicateDirection = true;
						break;
					}
				}

				if (duplicateDirection)
					continue;
			}

			primitiveCandidateKeepFlags[rawIdx] = 1;
		}
	}

	__global__ void NMLQ_CountSelectedPrimitiveContactsPerTriPair(
		dyno::DArray<int> selectedPrimitiveCounts,
		dyno::DArray<int> primitivePassCounts,
		dyno::DArray<int> primitivePassOffsets,
		dyno::DArray<int> primitiveCandidateKeepFlags)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= selectedPrimitiveCounts.size())
			return;

		int slotBase = NMLQ_GetPairPassSlot(pairId, 0);
		if (slotBase < 0 || slotBase + (NMLQ_PASS_COUNT - 1) >= primitivePassCounts.size()
			|| slotBase >= primitivePassOffsets.size())
		{
			selectedPrimitiveCounts[pairId] = 0;
			return;
		}

		int rawCount = 0;
		for (int passType = 0; passType < NMLQ_PASS_COUNT; ++passType)
			rawCount += primitivePassCounts[slotBase + passType];
		if (rawCount <= 0)
		{
			selectedPrimitiveCounts[pairId] = 0;
			return;
		}

		int rawBegin = primitivePassOffsets[slotBase];
		int selectedCount = 0;
		for (int rawIdx = rawBegin; rawIdx < rawBegin + rawCount && rawIdx < primitiveCandidateKeepFlags.size(); ++rawIdx)
		{
			if (primitiveCandidateKeepFlags[rawIdx] > 0)
				++selectedCount;
		}
		selectedPrimitiveCounts[pairId] = selectedCount;
	}

	__global__ void NMLQ_SetFinalContactCounts(
		dyno::DArray<int> finalContactCounts,
		dyno::DArray<int> selectedPrimitiveCounts)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= finalContactCounts.size() || pairId >= selectedPrimitiveCounts.size())
			return;

		finalContactCounts[pairId] = selectedPrimitiveCounts[pairId];
	}

	template<typename ContactPair>
	__global__ void NMLQ_SetFinalContactsPerTriPair(
		dyno::DArray<ContactPair> contacts,
		dyno::DArray<int> offsets,
		dyno::DArray<int> primitivePassCounts,
		dyno::DArray<int> primitivePassOffsets,
		dyno::DArray<int> primitiveCandidateKeepFlags,
		dyno::DArray<ContactPair> primitiveCandidateContacts,
		dyno::DArray<int> selectedPrimitiveCounts)
	{
		int pairId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pairId >= offsets.size() || pairId >= selectedPrimitiveCounts.size())
			return;

		int writeBase = offsets[pairId];
		int selectedCount = selectedPrimitiveCounts[pairId];
		if (selectedCount > 0)
		{
			int slotBase = NMLQ_GetPairPassSlot(pairId, 0);
			if (slotBase < 0 || slotBase + (NMLQ_PASS_COUNT - 1) >= primitivePassCounts.size()
				|| slotBase >= primitivePassOffsets.size())
				return;

			int rawCount = 0;
			for (int passType = 0; passType < NMLQ_PASS_COUNT; ++passType)
				rawCount += primitivePassCounts[slotBase + passType];
			int rawBegin = primitivePassOffsets[slotBase];
			int written = 0;
			for (int rawIdx = rawBegin; rawIdx < rawBegin + rawCount && rawIdx < primitiveCandidateKeepFlags.size(); ++rawIdx)
			{
				if (primitiveCandidateKeepFlags[rawIdx] <= 0 || rawIdx >= primitiveCandidateContacts.size())
					continue;

				int outIdx = writeBase + written;
				if (outIdx >= 0 && outIdx < contacts.size())
					contacts[outIdx] = primitiveCandidateContacts[rawIdx];
				++written;
				}
				return;
			}
	}
}

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborMeshLevelQuery, TDataType)

	template<typename TDataType>
	NeighborMeshLevelQuery<TDataType>::NeighborMeshLevelQuery()
		: NeighborTriMeshQuery<TDataType>()
	{
		auto topologyInitCallback = std::make_shared<FCallBackFunc>(
			[this]()
			{
				this->initializeTopologyOwnershipCache(true);
			});
		this->inTriangleSet()->attach(topologyInitCallback);
	}

	template<typename TDataType>
	NeighborMeshLevelQuery<TDataType>::~NeighborMeshLevelQuery()
	{
		clearTopologyOwnershipCache();
	}

	template<typename TDataType>
	bool NeighborMeshLevelQuery<TDataType>::initializeImpl()
	{
		if (!NeighborTriMeshQuery<TDataType>::initializeImpl())
			return false;

		if (!mTopologyOwnershipReady && !this->inTriangleSet()->isEmpty())
			initializeTopologyOwnershipCache(false);

		return true;
	}

	template<typename TDataType>
	void NeighborMeshLevelQuery<TDataType>::clearTopologyOwnershipCache()
	{
		mVertexAdjacentFaces.clear();
		mVertexIncidentEdges.clear();
		mEdgeAdjacentFaces.clear();
		mTriangleEdges.clear();
		mEdgeVertexIndices.clear();

		mFaceAssignedVertexOffsets.clear();
		mFaceAssignedVertexIndices.clear();
		mFaceAssignedEdgeOffsets.clear();
		mFaceAssignedEdgeIndices.clear();

		mVertexAssignedFaces.clear();
		mEdgeAssignedFaces.clear();
		mEdgeNormals.clear();
		mTri2Shape.clear();

		mTriangleAabbsWorld.clear();
		mFaceNormalsWorld.clear();
		mEdgeNormalsWorld.clear();
		mPatchPairTriPairCounts.clear();
		mPatchPairTriPairOffsets.clear();
		mCandidateTri0.clear();
		mCandidateTri1.clear();
		mCandidatePatchPairId.clear();
		mCoarsePassCounts.clear();
		mCoarsePassOffsets.clear();
		mFilteredTri0.clear();
		mFilteredTri1.clear();
		mFilteredPatchPairId.clear();
		mPrimitivePassCounts.clear();
		mPrimitivePassOffsets.clear();
		mPrimitiveCandidateContacts.clear();
		mPrimitiveCandidateKeys.clear();
			mPrimitiveCandidateSortedIndices.clear();
			mPrimitiveCandidateKeepFlags.clear();
			mSelectedPrimitiveCounts.clear();
			mTriPairContactCounts.clear();
			mTriPairContactOffsets.clear();

		mTopologyOwnershipReady = false;
		mTriShapeReady = false;
	}

	template<typename TDataType>
	bool NeighborMeshLevelQuery<TDataType>::updateTriShapeLookup(bool forceRebuild)
	{
		if (mTriShapeReady && !forceRebuild)
			return true;

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr)
		{
			mTri2Shape.clear();
			mTriShapeReady = false;
			return false;
		}

		auto& shape2TriOffsets = this->inShape2TriOffsets()->getData();
		const int triCount = static_cast<int>(ts->triangleIndices().size());
		if (triCount <= 0 || shape2TriOffsets.size() < 2)
		{
			mTri2Shape.clear();
			mTriShapeReady = false;
			return false;
		}

		CArray<int> hShape2TriOffsets;
		hShape2TriOffsets.assign(shape2TriOffsets);
		const int shapeCount = static_cast<int>(hShape2TriOffsets.size()) - 1;
		if (shapeCount <= 0)
		{
			mTri2Shape.clear();
			mTriShapeReady = false;
			return false;
		}

		std::vector<int> tri2Shape(triCount, EMPTY);
		for (int shapeId = 0; shapeId < shapeCount; ++shapeId)
		{
			int begin = hShape2TriOffsets[shapeId];
			int end = hShape2TriOffsets[shapeId + 1];
			if (begin < 0) begin = 0;
			if (end > triCount) end = triCount;
			for (int triId = begin; triId < end; ++triId)
				tri2Shape[triId] = shapeId;
		}

		for (int triId = 0; triId < triCount; ++triId)
		{
			if (tri2Shape[triId] == EMPTY)
			{
				mTri2Shape.clear();
				mTriShapeReady = false;
				return false;
			}
		}

		mTri2Shape.assign(tri2Shape);
		mTriShapeReady = true;
		return true;
	}

	template<typename TDataType>
	void NeighborMeshLevelQuery<TDataType>::initializeTopologyOwnershipCache(bool forceRebuild)
	{
		if (mTopologyOwnershipReady && !forceRebuild)
			return;

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr)
		{
			clearTopologyOwnershipCache();
			return;
		}

		if (ts->triangleIndices().size() == 0 || ts->getPoints().size() == 0)
		{
			clearTopologyOwnershipCache();
			return;
		}

		ts->update();

		using Coord = typename NeighborMeshLevelQuery<TDataType>::Coord;
		using Triangle = typename NeighborMeshLevelQuery<TDataType>::Triangle;
		using Edge = typename NeighborMeshLevelQuery<TDataType>::Edge;
		using Tri2Edg = typename NeighborMeshLevelQuery<TDataType>::Tri2Edg;
		using Edg2Tri = typename NeighborMeshLevelQuery<TDataType>::Edg2Tri;

		CArray<Coord> hVertices;
		hVertices.assign(ts->getPoints());

		CArray<Triangle> hTriangles;
		hTriangles.assign(ts->triangleIndices());

		CArray<Edge> hEdges;
		hEdges.assign(ts->edgeIndices());

		CArrayList<int> hVertex2Tri;
		hVertex2Tri.assign(ts->vertex2Triangle());

		CArrayList<int> hVertex2Edge;
		hVertex2Edge.assign(ts->vertex2Edge());

		CArray<Tri2Edg> hTri2Edg;
		hTri2Edg.assign(ts->triangle2Edge());

		CArray<Edg2Tri> hEdge2Tri;
		hEdge2Tri.assign(ts->edge2Triangle());

		const int vertexCount = static_cast<int>(hVertices.size());
		const int faceCount = static_cast<int>(hTriangles.size());
		const int edgeCount = static_cast<int>(hEdges.size());

		if (faceCount <= 0 || vertexCount <= 0)
		{
			clearTopologyOwnershipCache();
			return;
		}

		if (static_cast<int>(hVertex2Tri.size()) != vertexCount
			|| static_cast<int>(hVertex2Edge.size()) != vertexCount
			|| static_cast<int>(hTri2Edg.size()) != faceCount
			|| static_cast<int>(hEdge2Tri.size()) != edgeCount)
		{
			clearTopologyOwnershipCache();
			return;
		}

		mVertexAdjacentFaces.assign(hVertex2Tri);
		mVertexIncidentEdges.assign(hVertex2Edge);
		mEdgeAdjacentFaces.assign(hEdge2Tri);
		mTriangleEdges.assign(hTri2Edg);
		mEdgeVertexIndices.assign(hEdges);

		std::vector<Coord> faceNormals(faceCount, Coord(1, 0, 0));
		for (int faceId = 0; faceId < faceCount; ++faceId)
		{
			const Triangle tri = hTriangles[faceId];
			if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0
				|| tri[0] >= vertexCount || tri[1] >= vertexCount || tri[2] >= vertexCount)
			{
				faceNormals[faceId] = Coord(1, 0, 0);
				continue;
			}

			faceNormals[faceId] = NLQ_BuildRobustFaceNormal(
				hVertices[tri[0]],
				hVertices[tri[1]],
				hVertices[tri[2]]);
		}

		std::vector<int> vertexAssignedFaces(vertexCount, EMPTY);
		std::vector<int> edgeAssignedFaces(edgeCount, EMPTY);
		std::vector<std::vector<int>> faceAssignedVertices(faceCount);
		std::vector<std::vector<int>> faceAssignedEdges(faceCount);
		std::vector<int> faceVertexLoads(faceCount, 0);
		std::vector<int> faceEdgeLoads(faceCount, 0);

		for (int vertexId = 0; vertexId < vertexCount; ++vertexId)
		{
			auto& adjacentFaces = hVertex2Tri[vertexId];
			int bestFace = EMPTY;
			int bestLoad = std::numeric_limits<int>::max();
			for (int i = 0; i < adjacentFaces.size(); ++i)
			{
				const int faceId = adjacentFaces[i];
				if (faceId < 0 || faceId >= faceCount)
					continue;

				const int currentLoad = faceVertexLoads[faceId];
				if (bestFace == EMPTY || currentLoad < bestLoad || (currentLoad == bestLoad && faceId < bestFace))
				{
					bestFace = faceId;
					bestLoad = currentLoad;
				}
			}

			if (bestFace != EMPTY)
			{
				vertexAssignedFaces[vertexId] = bestFace;
				faceAssignedVertices[bestFace].push_back(vertexId);
				faceVertexLoads[bestFace] += 1;
			}
		}

		for (int edgeId = 0; edgeId < edgeCount; ++edgeId)
		{
			const Edg2Tri adjacentFaces = hEdge2Tri[edgeId];
			int bestFace = EMPTY;
			int bestLoad = std::numeric_limits<int>::max();
			for (int i = 0; i < 2; ++i)
			{
				const int faceId = adjacentFaces[i];
				if (faceId < 0 || faceId >= faceCount)
					continue;

				const int currentLoad = faceEdgeLoads[faceId];
				if (bestFace == EMPTY || currentLoad < bestLoad || (currentLoad == bestLoad && faceId < bestFace))
				{
					bestFace = faceId;
					bestLoad = currentLoad;
				}
			}

			if (bestFace != EMPTY)
			{
				edgeAssignedFaces[edgeId] = bestFace;
				faceAssignedEdges[bestFace].push_back(edgeId);
				faceEdgeLoads[bestFace] += 1;
			}
		}

		std::vector<Coord> edgeNormals(edgeCount, Coord(1, 0, 0));
		for (int edgeId = 0; edgeId < edgeCount; ++edgeId)
		{
			const auto adjacentFaces = hEdge2Tri[edgeId];
			const int face0 = adjacentFaces[0];
			const int face1 = adjacentFaces[1];

			Coord edgeNormal(0);
			if (face0 != EMPTY && face1 != EMPTY)
			{
				edgeNormal = faceNormals[face0] + faceNormals[face1];
				if (edgeNormal.normSquared() > static_cast<typename Coord::VarType>(1e-12))
					edgeNormal.normalize();
				else
					edgeNormal = faceNormals[face0];
			}
			else if (face0 != EMPTY)
			{
				edgeNormal = faceNormals[face0];
			}
			else if (face1 != EMPTY)
			{
				edgeNormal = faceNormals[face1];
			}

			if (edgeNormal.normSquared() <= static_cast<typename Coord::VarType>(1e-12))
			{
				const Edge edge = hEdges[edgeId];
				if (edge[0] >= 0 && edge[1] >= 0 && edge[0] < vertexCount && edge[1] < vertexCount)
					edgeNormal = NLQ_StablePerpendicular(hVertices[edge[1]] - hVertices[edge[0]]);
				else
					edgeNormal = Coord(1, 0, 0);
			}

			edgeNormals[edgeId] = edgeNormal;
		}

		NLQ_BuildCSR(faceAssignedVertices, mFaceAssignedVertexOffsets, mFaceAssignedVertexIndices);
		NLQ_BuildCSR(faceAssignedEdges, mFaceAssignedEdgeOffsets, mFaceAssignedEdgeIndices);

		mVertexAssignedFaces.assign(vertexAssignedFaces);
		mEdgeAssignedFaces.assign(edgeAssignedFaces);
		mEdgeNormals.assign(edgeNormals);

		mTopologyOwnershipReady = true;
		mTriShapeReady = false;
	}

	template<typename TDataType>
	void NeighborMeshLevelQuery<TDataType>::buildCollisionTriSet()
	{
		if (!this->inEnableVisualizeCollisionTriSet()->getValue() || this->outContacts()->size() == 0)
		{
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr)
		{
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto& vertices = ts->getPoints();
		auto& triIndices = ts->triangleIndices();
		CArray<ContactPair> hContacts;
		hContacts.assign(this->outContacts()->getData());
		CArray<Coord> hVertices;
		hVertices.assign(vertices);
		CArray<Triangle> hTriangles;
		hTriangles.assign(triIndices);

		CArray<int> hTri2Shape;
		if (this->varInputVerticesInRestWorld()->getValue())
		{
			if (!mTriShapeReady)
				updateTriShapeLookup(false);
			hTri2Shape.assign(mTri2Shape);
		}

		CArray<Coord> hShapeRestT;
		CArray<Matrix> hShapeRestR;
		if (this->varInputVerticesInRestWorld()->getValue())
		{
			hShapeRestT.assign(this->shapeRestTranslationsData());
			hShapeRestR.assign(this->shapeRestRotationsData());
		}

		std::vector<Coord> contactVertices;
		std::vector<Triangle> contactTriangles;
		contactVertices.reserve(hContacts.size() * 6);
		contactTriangles.reserve(hContacts.size() * 2);

		for (uint i = 0; i < hContacts.size(); ++i)
		{
			const int triId0 = hContacts[i].localId1;
			const int triId1 = hContacts[i].localId2;
			if (triId0 < 0 || triId0 >= static_cast<int>(hTriangles.size())
				|| triId1 < 0 || triId1 >= static_cast<int>(hTriangles.size()))
				continue;

			auto appendTriangle = [&](int triId)
			{
				Triangle tri = hTriangles[triId];
				Coord p0 = hVertices[tri[0]];
				Coord p1 = hVertices[tri[1]];
				Coord p2 = hVertices[tri[2]];
				if (this->varInputVerticesInRestWorld()->getValue() && triId < static_cast<int>(hTri2Shape.size()))
				{
					int shapeId = hTri2Shape[triId];
					if (shapeId >= 0 && shapeId < static_cast<int>(hShapeRestR.size()) && shapeId < static_cast<int>(hShapeRestT.size()))
					{
						p0 = hShapeRestR[shapeId] * p0 + hShapeRestT[shapeId];
						p1 = hShapeRestR[shapeId] * p1 + hShapeRestT[shapeId];
						p2 = hShapeRestR[shapeId] * p2 + hShapeRestT[shapeId];
					}
				}

				int base = static_cast<int>(contactVertices.size());
				contactVertices.push_back(p0);
				contactVertices.push_back(p1);
				contactVertices.push_back(p2);
				contactTriangles.push_back(Triangle(base, base + 1, base + 2));
			};

			appendTriangle(triId0);
			appendTriangle(triId1);
		}

		if (contactTriangles.empty())
		{
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		this->triSet->setPoints(contactVertices);
		this->triSet->setTriangles(contactTriangles);
		this->triSet->update();
		this->outPotentialTriSet()->setDataPtr(this->triSet);
	}

	template<typename TDataType>
	void NeighborMeshLevelQuery<TDataType>::narrowPhase()
	{
		auto& patchPairs = this->outPotentialPatchPairs()->getData();
		if (patchPairs.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		if (!mTopologyOwnershipReady)
			initializeTopologyOwnershipCache(false);

		auto ts = this->inTriangleSet()->constDataPtr();
		if (ts == nullptr || !mTopologyOwnershipReady)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		auto& vertices = ts->getPoints();
		auto& triangles = ts->triangleIndices();
		auto& patch2TriOffsets = this->inPatch2TriOffsets()->getData();
		auto& patch2TriIndices = this->inPatch2TriIndices()->getData();

		const int triCount = static_cast<int>(triangles.size());
		const int edgeCount = static_cast<int>(mEdgeVertexIndices.size());
		if (triCount <= 0 || patch2TriOffsets.size() < 2 || patch2TriIndices.size() == 0)
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		if (this->varInputVerticesInRestWorld()->getValue() && !updateTriShapeLookup(false))
		{
			this->outContacts()->resize(0);
			this->triSet->clear();
			this->outPotentialTriSet()->setDataPtr(this->triSet);
			return;
		}

		if (mTriangleAabbsWorld.size() != static_cast<uint>(triCount))
			mTriangleAabbsWorld.resize(triCount);
		if (mFaceNormalsWorld.size() != static_cast<uint>(triCount))
			mFaceNormalsWorld.resize(triCount);
		if (mEdgeNormalsWorld.size() != static_cast<uint>(edgeCount))
			mEdgeNormalsWorld.resize(edgeCount);

		NMLQRuntimeView<Real, Coord, Matrix, Triangle, Edge, Tri2Edg, Edg2Tri, AABB> view{
			vertices,
			triangles,
			mTri2Shape,
			this->shapeRestRotationsData(),
			this->shapeRestTranslationsData(),
			mTriangleEdges,
			mEdgeVertexIndices,
			mEdgeAdjacentFaces,
			mEdgeNormals,
			mFaceNormalsWorld,
			mEdgeNormalsWorld,
			mTriangleAabbsWorld,
				mVertexIncidentEdges,
				mFaceAssignedVertexOffsets,
				mFaceAssignedVertexIndices,
				mFaceAssignedEdgeOffsets,
				mFaceAssignedEdgeIndices,
				patchPairs,
				this->patch2ShapeData(),
				this->inShape2RigidBodyIds()->getData(),
				this->varDHead()->getValue(),
				this->varEdgeEdgeActivationMargin()->getValue(),
				this->varInputVerticesInRestWorld()->getValue()
			};

		// Precompute world-space triangle AABBs, face normals, edge normals for coarse culling and narrow-phase reuse.
		cuExecute(triCount, NMLQ_PrepareTriangleWorldData, view);
		if (edgeCount > 0)
			cuExecute(edgeCount, NMLQ_PrepareEdgeNormalsWorld, view);

		if (mPatchPairTriPairCounts.size() != patchPairs.size())
			mPatchPairTriPairCounts.resize(patchPairs.size());
		mPatchPairTriPairCounts.reset();

		cuExecute(patchPairs.size(),
			NMLQ_CountTriPairsPerPatchPair,
			mPatchPairTriPairCounts,
			patchPairs,
			patch2TriOffsets);

		int totalCandidateTriPairs = mReduce.accumulate(mPatchPairTriPairCounts.begin(), mPatchPairTriPairCounts.size());
		if (totalCandidateTriPairs <= 0)
		{
			this->outContacts()->resize(0);
			buildCollisionTriSet();
			return;
		}

		if (mPatchPairTriPairOffsets.size() != mPatchPairTriPairCounts.size())
			mPatchPairTriPairOffsets.resize(mPatchPairTriPairCounts.size());
		mPatchPairTriPairOffsets.assign(mPatchPairTriPairCounts);
		mScan.exclusive(mPatchPairTriPairOffsets, true);

		if (mCandidateTri0.size() != static_cast<uint>(totalCandidateTriPairs))
			mCandidateTri0.resize(totalCandidateTriPairs);
		if (mCandidateTri1.size() != static_cast<uint>(totalCandidateTriPairs))
			mCandidateTri1.resize(totalCandidateTriPairs);
		if (mCandidatePatchPairId.size() != static_cast<uint>(totalCandidateTriPairs))
			mCandidatePatchPairId.resize(totalCandidateTriPairs);

		cuExecute(patchPairs.size(),
			NMLQ_SetTriPairs,
			mCandidateTri0,
			mCandidateTri1,
			mCandidatePatchPairId,
			mPatchPairTriPairOffsets,
			mPatchPairTriPairCounts,
			patchPairs,
			patch2TriOffsets,
			patch2TriIndices);

		if (mCoarsePassCounts.size() != static_cast<uint>(totalCandidateTriPairs))
			mCoarsePassCounts.resize(totalCandidateTriPairs);
		mCoarsePassCounts.reset();

		// count how many aabbs of triangle pair are overlapped
		cuExecute(totalCandidateTriPairs,
			NMLQ_CountCoarsePassedTriPairs,
			mCoarsePassCounts,
			mCandidateTri0,
			mCandidateTri1,
			mTriangleAabbsWorld,
			view.dHat);

		int totalFilteredTriPairs = mReduce.accumulate(mCoarsePassCounts.begin(), mCoarsePassCounts.size());
		if (totalFilteredTriPairs <= 0)
		{
			this->outContacts()->resize(0);
			buildCollisionTriSet();
			return;
		}

		if (mCoarsePassOffsets.size() != mCoarsePassCounts.size())
			mCoarsePassOffsets.resize(mCoarsePassCounts.size());
		mCoarsePassOffsets.assign(mCoarsePassCounts);
		mScan.exclusive(mCoarsePassOffsets, true);

		if (mFilteredTri0.size() != static_cast<uint>(totalFilteredTriPairs))
			mFilteredTri0.resize(totalFilteredTriPairs);
		if (mFilteredTri1.size() != static_cast<uint>(totalFilteredTriPairs))
			mFilteredTri1.resize(totalFilteredTriPairs);
		if (mFilteredPatchPairId.size() != static_cast<uint>(totalFilteredTriPairs))
			mFilteredPatchPairId.resize(totalFilteredTriPairs);

		cuExecute(totalCandidateTriPairs,
			NMLQ_SetCoarsePassedTriPairs,
			mFilteredTri0,
			mFilteredTri1,
			mFilteredPatchPairId,
			mCandidateTri0,
			mCandidateTri1,
			mCandidatePatchPairId,
			mCoarsePassOffsets,
			mCoarsePassCounts);

		const int primitivePassSlotCount = totalFilteredTriPairs * NMLQ_PASS_COUNT;
		if (mPrimitivePassCounts.size() != static_cast<uint>(primitivePassSlotCount))
			mPrimitivePassCounts.resize(primitivePassSlotCount);
		mPrimitivePassCounts.reset();

		cuExecute(primitivePassSlotCount,
			NMLQ_CountPrimitiveCandidatesPerPass,
			mPrimitivePassCounts,
			mFilteredTri0,
			mFilteredTri1,
			mFilteredPatchPairId,
			view);

		int totalPrimitiveCandidates = primitivePassSlotCount > 0
			? mReduce.accumulate(mPrimitivePassCounts.begin(), mPrimitivePassCounts.size())
			: 0;

		if (mPrimitivePassOffsets.size() != mPrimitivePassCounts.size())
			mPrimitivePassOffsets.resize(mPrimitivePassCounts.size());
		mPrimitivePassOffsets.assign(mPrimitivePassCounts);
		if (mPrimitivePassOffsets.size() > 0)
			mScan.exclusive(mPrimitivePassOffsets, true);

		if (mPrimitiveCandidateContacts.size() != static_cast<uint>(totalPrimitiveCandidates))
			mPrimitiveCandidateContacts.resize(totalPrimitiveCandidates);
		if (mPrimitiveCandidateKeys.size() != static_cast<uint>(totalPrimitiveCandidates))
			mPrimitiveCandidateKeys.resize(totalPrimitiveCandidates);
		if (mPrimitiveCandidateSortedIndices.size() != static_cast<uint>(totalPrimitiveCandidates))
			mPrimitiveCandidateSortedIndices.resize(totalPrimitiveCandidates);
		if (mPrimitiveCandidateKeepFlags.size() != static_cast<uint>(totalPrimitiveCandidates))
			mPrimitiveCandidateKeepFlags.resize(totalPrimitiveCandidates);
		mPrimitiveCandidateKeepFlags.reset();

		if (totalPrimitiveCandidates > 0)
		{
			cuExecute(primitivePassSlotCount,
				NMLQ_SetPrimitiveCandidatesPerPass,
				mPrimitiveCandidateContacts,
				mPrimitiveCandidateKeys,
				mPrimitivePassOffsets,
				mPrimitivePassCounts,
				mFilteredTri0,
				mFilteredTri1,
				mFilteredPatchPairId,
				view);

			cuExecute(totalPrimitiveCandidates,
				NMLQ_InitPrimitiveCandidateIndices,
				mPrimitiveCandidateSortedIndices);

			thrust::stable_sort_by_key(
				thrust::device,
				mPrimitiveCandidateKeys.begin(),
				mPrimitiveCandidateKeys.begin() + mPrimitiveCandidateKeys.size(),
				mPrimitiveCandidateSortedIndices.begin());

			{
				uint pDims = cudaGridSize((uint)totalPrimitiveCandidates, BLOCK_SIZE);
					NMLQ_MarkMinDepthCandidatesPerPrimitiveKey<ContactPair, Real><<<pDims, BLOCK_SIZE>>>(
						mPrimitiveCandidateKeepFlags,
						mPrimitiveCandidateKeys,
						mPrimitiveCandidateSortedIndices,
						mPrimitiveCandidateContacts,
						Real(1e-6),
						Real(1e-4));
				cuSynchronize();
			}
		}

		if (mSelectedPrimitiveCounts.size() != static_cast<uint>(totalFilteredTriPairs))
			mSelectedPrimitiveCounts.resize(totalFilteredTriPairs);
		mSelectedPrimitiveCounts.reset();
		if (totalPrimitiveCandidates > 0)
		{
			cuExecute(totalFilteredTriPairs,
				NMLQ_CountSelectedPrimitiveContactsPerTriPair,
				mSelectedPrimitiveCounts,
				mPrimitivePassCounts,
				mPrimitivePassOffsets,
				mPrimitiveCandidateKeepFlags);
		}

		if (mTriPairContactCounts.size() != static_cast<uint>(totalFilteredTriPairs))
			mTriPairContactCounts.resize(totalFilteredTriPairs);
		mTriPairContactCounts.reset();
		cuExecute(totalFilteredTriPairs,
			NMLQ_SetFinalContactCounts,
			mTriPairContactCounts,
			mSelectedPrimitiveCounts);

		int totalContacts = mReduce.accumulate(mTriPairContactCounts.begin(), mTriPairContactCounts.size());
		if (totalContacts <= 0)
		{
			this->outContacts()->resize(0);
			buildCollisionTriSet();
			return;
		}

		if (mTriPairContactOffsets.size() != mTriPairContactCounts.size())
			mTriPairContactOffsets.resize(mTriPairContactCounts.size());
		mTriPairContactOffsets.assign(mTriPairContactCounts);
		mScan.exclusive(mTriPairContactOffsets, true);

		this->outContacts()->resize(totalContacts);
		cuExecute(totalFilteredTriPairs,
			NMLQ_SetFinalContactsPerTriPair,
			this->outContacts()->getData(),
			mTriPairContactOffsets,
			mPrimitivePassCounts,
			mPrimitivePassOffsets,
			mPrimitiveCandidateKeepFlags,
			mPrimitiveCandidateContacts,
			mSelectedPrimitiveCounts);

		buildCollisionTriSet();
	}

	DEFINE_CLASS(NeighborMeshLevelQuery);
}
