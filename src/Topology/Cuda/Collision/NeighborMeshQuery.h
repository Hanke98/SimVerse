#pragma once
#include "NeighborTriMeshQuery.h"

namespace dyno
{
	template<typename TDataType>
	class NeighborMeshQuery : public NeighborTriMeshQuery<TDataType>
	{
		DECLARE_TCLASS(NeighborMeshQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename ::dyno::Pair<uint, uint> PairUU;

		NeighborMeshQuery();
		~NeighborMeshQuery() override;

    public:
        

	protected:
		void compute() override;

	private:
		bool broadPhase();
		void narrowPhase();
		bool updateShape2ElementIds(int shapeCount);

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<int> mShape2ElementIds;
		DArray<int> mShape2RigidBodyIds;

		bool mMeshWarnedEmptyMapping = false;
		bool mMeshWarnedEmptyElementMapping = false;
	};
}
