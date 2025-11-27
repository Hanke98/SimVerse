#include "TJConstraintSolver.h"
#include "SharedFuncsForRigidBody.h"
#include "Profiler.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TJConstraintSolver, TDataType)

	template<typename TDataType>
	TJConstraintSolver<TDataType>::TJConstraintSolver()
		:ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
	}

	template<typename TDataType>
	TJConstraintSolver<TDataType>::~TJConstraintSolver()
	{
	}

	template<typename TDataType>
	void TJConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		int ballAndSocketJoint_size = topo->ballAndSocketJoints().size();
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		int pointJoint_size = topo->pointJoints().size();

		if (this->varFrictionEnabled()->getData())
		{
			constraint_size += 3 * contact_size;
		}
		else
		{
			constraint_size = contact_size;
		}

		if (ballAndSocketJoint_size != 0)
		{
			constraint_size += 3 * ballAndSocketJoint_size;
		}

		if (sliderJoint_size != 0)
		{
			constraint_size += 8 * sliderJoint_size;
		}

		if (hingeJoint_size != 0)
		{
			constraint_size += 8 * hingeJoint_size;
		}

		if (fixedJoint_size != 0)
		{
			constraint_size += 6 * fixedJoint_size;
		}

		if (pointJoint_size != 0)
		{
			constraint_size += 3 * pointJoint_size;
		}

		if (constraint_size == 0)
		{
			return;
		}

		mVelocityConstraints.resize(constraint_size);

		if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			setUpContactAndFrictionConstraints(
				mVelocityConstraints,
				mContactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->varFrictionEnabled()->getData()
			);
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = topo->ballAndSocketJoints();
			int begin_index = contact_size;

			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}

			setUpBallAndSocketJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index
			);
		}

		if (sliderJoint_size != 0)
		{
			auto& joints = topo->sliderJoints();
			int begin_index = contact_size;

			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			begin_index += 3 * ballAndSocketJoint_size;
			setUpSliderJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpHingeJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpFixedJointConstraints(
				mVelocityConstraints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);;
		}

		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size + 6 * fixedJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpPointJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				begin_index
			);
		}

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mJ.resize(4 * constraint_size);
		mB.resize(4 * constraint_size);
		mK_1.resize(constraint_size);
		mK_2.resize(constraint_size);
		mK_3.resize(constraint_size);
		mEta.resize(constraint_size);
		mLambda.resize(constraint_size);

		mJ.reset();
		mB.reset();
		mK_1.reset();
		mK_2.reset();
		mK_3.reset();
		mEta.reset();
		mLambda.reset();

		mContactNumber.reset();

		calculateJacobianMatrix(
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			this->inRotationMatrix()->getData(),
			mVelocityConstraints
		);

		calculateK(
			mVelocityConstraints,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mK_1,
			mK_2,
			mK_3
		);

		mErrors.resize(constraint_size);
		mErrors.reset();


		calculateEtaVectorForPJSBaumgarte(
			mEta,
			mJ,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			mVelocityConstraints,
			mErrors,
			this->varSlop()->getValue(),
			this->varBaumgarteBias()->getValue(),
			this->varSubStepping()->getValue(),
			dt
		);

		if (contact_size != 0)
		{
			calculateContactPoints(
				this->inContacts()->getData(),
				mContactNumber);
		}
	}


#define LEI_BLOCK_SIZE 128
	namespace LeiModify
	{
		template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
		__global__ void SF_setUpSliderJointConstraintsCompress(
			DArray<Constraint> constraints,
			DArray<Joint> joints,
			DArray<Coord> pos,
			DArray<Matrix> rotMat,
			DArray<Quat> rotQuat,
			int begin_index
		)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);

			if (tId >= joints.size())
				return;

			int constraint_size = 8;

			int idx1 = joints[tId].bodyId1;
			int idx2 = joints[tId].bodyId2;

			Coord r1 = rotMat[idx1] * joints[tId].r1;
			Coord r2 = rotMat[idx2] * joints[tId].r2;

			Coord n = rotMat[idx1] * joints[tId].sliderAxis;
			n = n.normalize();
			Coord n1, n2;
			if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
			{
				n1 = Coord(0, n[2], -n[1]);
				n1 = n1.normalize();
			}
			else if (abs(n[0]) > EPSILON)
			{
				n1 = Coord(n[2], 0, -n[0]);
				n1 = n1.normalize();
			}
			n2 = n1.cross(n);
			n2 = n2.normalize();

			int baseIndex = constraint_size * tId + begin_index;

			for (int i = 0; i < 5; i++)
			{
				constraints[baseIndex + i].isValid = true;
			}


			bool useRange = joints[tId].useRange;
			Real C_min = 0.0;
			Real C_max = 0.0;
			if (joints[tId].useRange)
			{
				Coord u = pos[idx2] + r2 - pos[idx1] - r1;
				C_min = u.dot(n) - joints[tId].d_min;
				C_max = joints[tId].d_max - u.dot(n);
				if (C_min < 0)
					constraints[baseIndex + 5].isValid = true;
				else
					constraints[baseIndex + 5].isValid = false;
				if (C_max < 0)
					constraints[baseIndex + 6].isValid = true;
				else
					constraints[baseIndex + 6].isValid = false;
			}
			else
			{
				constraints[baseIndex + 5].isValid = false;
				constraints[baseIndex + 6].isValid = false;
			}

			Real v_moter = 0.0;
			bool useMoter = joints[tId].useMoter;
			if (useMoter)
			{
				v_moter = joints[tId].v_moter;
				constraints[baseIndex + 7].isValid = true;
			}
			else
			{
				constraints[baseIndex + 7].isValid = false;
			}

			for (int i = 0; i < constraint_size; i++)
			{
				auto& constraint = constraints[baseIndex + i];
				constraint.bodyId1 = idx1;
				constraint.bodyId2 = idx2;
				constraint.pos1 = r1;
				constraint.pos2 = r2;
				constraint.normal1 = n1;
				constraint.normal2 = n2;
				constraint.axis = n;
				constraint.interpenetration = v_moter;
				constraint.d_min = C_min;
				constraint.d_max = C_max;
				constraint.rotQuat = joints[tId].q_init;
			}

			constraints[baseIndex].type = ConstraintType::CN_ANCHOR_TRANS_1;
			constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_TRANS_2;
			constraints[baseIndex + 2].type = ConstraintType::CN_BAN_ROT_1;
			constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_2;
			constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_3;
			constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_SLIDER_MIN;
			constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_SLIDER_MAX;
			constraints[baseIndex + 7].type = ConstraintType::CN_JOINT_SLIDER_MOTER;
		}

		template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
		__global__ void SF_setUpHingeJointConstraintsCompress(
			DArray<Constraint> constraints,
			DArray<Joint> joints,
			DArray<Coord> pos,
			DArray<Matrix> rotMat,
			DArray<Quat> rotation_q,
			int begin_index
		)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (tId >= joints.size())
				return;

			int constraint_size = 8;

			int idx1 = joints[tId].bodyId1;
			int idx2 = joints[tId].bodyId2;

			Matrix rotMat1 = rotMat[idx1];
			Matrix rotMat2 = rotMat[idx2];


			Coord r1 = rotMat1 * joints[tId].r1;
			Coord r2 = rotMat2 * joints[tId].r2;

			Coord a1 = rotMat1 * joints[tId].hingeAxisBody1;
			Coord a2 = rotMat2 * joints[tId].hingeAxisBody2;




			// two vector orthogonal to the a2
			Coord b2, c2;
			if (abs(a2[1]) > EPSILON || abs(a2[2]) > EPSILON)
			{
				b2 = Coord(0, a2[2], -a2[1]);
				b2 = b2.normalize();
			}
			else if (abs(a2[0]) > EPSILON)
			{
				b2 = Coord(a2[2], 0, -a2[0]);
				b2 = b2.normalize();
			}
			c2 = b2.cross(a2);
			c2 = c2.normalize();

			Real C_min = 0.0;
			Real C_max = 0.0;
			int baseIndex = tId * constraint_size + begin_index;

			if (joints[tId].useRange)
			{
				Real theta = rotation_q[idx2].angle(rotation_q[idx1]);

				Quat q_rot = rotation_q[idx2] * rotation_q[idx1].inverse();

				if (a1.dot(Coord(q_rot.x, q_rot.y, q_rot.z)) < 0)
				{
					theta = -theta;
				}


				C_min = theta - joints[tId].d_min;
				C_max = joints[tId].d_max - theta;

				if (C_min < 0)
				{
					constraints[baseIndex + 5].isValid = true;
				}
				else
					constraints[baseIndex + 5].isValid = false;

				if (C_max < 0)
				{
					constraints[baseIndex + 6].isValid = true;
				}
				else
					constraints[baseIndex + 6].isValid = false;
			}

			else
			{
				constraints[baseIndex + 5].isValid = false;
				constraints[baseIndex + 6].isValid = false;
			}

			Real v_moter = 0.0;
			if (joints[tId].useMoter)
			{
				v_moter = joints[tId].v_moter;
				constraints[baseIndex + 7].isValid = true;
			}
			else
				constraints[baseIndex + 7].isValid = false;

			for (int i = 0; i < constraint_size; i++)
			{
				constraints[baseIndex + i].bodyId1 = idx1;
				constraints[baseIndex + i].bodyId2 = idx2;
				constraints[baseIndex + i].axis = a1;
				constraints[baseIndex + i].normal1 = r1;
				constraints[baseIndex + i].normal2 = r2;
				constraints[baseIndex + i].pos1 = b2;
				constraints[baseIndex + i].pos2 = c2;
				constraints[baseIndex + i].d_min = C_min > 0 ? 0 : C_min;
				constraints[baseIndex + i].d_max = C_max > 0 ? 0 : C_max;
				constraints[baseIndex + i].interpenetration = v_moter;
			}

			for (int i = 0; i < 5; i++)
			{
				constraints[baseIndex + i].isValid = true;
			}

			constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
			constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
			constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;

			constraints[baseIndex + 3].type = ConstraintType::CN_ALLOW_ROT1D_1;
			constraints[baseIndex + 4].type = ConstraintType::CN_ALLOW_ROT1D_2;

			constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_HINGE_MIN;
			constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_HINGE_MAX;
			constraints[baseIndex + 7].type = ConstraintType::CN_JOINT_HINGE_MOTER;
		}


		template<typename Joint, typename Constraint, typename Matrix, typename Quat>
		__global__ void SF_setUpFixedJointConstraintsCompress(
			DArray<Constraint> constraints,
			DArray<Joint> joints,
			DArray<Matrix> rotMat,
			DArray<Quat> rotQuat,
			int begin_index
		)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);

			if (tId >= joints.size())
				return;

			int idx1 = joints[tId].bodyId1;
			int idx2 = joints[tId].bodyId2;
			Vector<Real, 3> r1 = rotMat[idx1] * joints[tId].r1;
			Vector<Real, 3> r2;
			if (idx2 != INVALID)
			{
				r2 = rotMat[idx2] * joints[tId].r2;
			}

			int baseIndex = 6 * tId + begin_index;
			for (int i = 0; i < 6; i++)
			{
				constraints[baseIndex + i].bodyId1 = idx1;
				constraints[baseIndex + i].bodyId2 = idx2;
				constraints[baseIndex + i].normal1 = r1;
				constraints[baseIndex + i].normal2 = r2;
				constraints[baseIndex + i].pos1 = joints[tId].w;
				constraints[baseIndex + i].rotQuat = joints[tId].q_init;
				constraints[baseIndex + i].isValid = true;
			}

			constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
			constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
			constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
			constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_1;
			constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_2;
			constraints[baseIndex + 5].type = ConstraintType::CN_BAN_ROT_3;
		}

		__device__ void SF_AnchorEqualConstraint123
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			Vec3f tmp(eta[baseId], eta[baseId + 1], eta[baseId + 2]);
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]) + J[4 * (baseId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (baseId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]);
				}
			}

			Vec3f delta_lambda = omega * (K_3[baseId] * tmp);

			const int offset0 = gId % 3;
			const int offset1 = (gId + 1) % 3;
			const int offset2 = (gId + 2) % 3;

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * (baseId + i)][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * (baseId + i)][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * (baseId + i)][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * (baseId + i) + 1][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * (baseId + i) + 1][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * (baseId + i) + 1][offset2] * delta_lambda[i]);

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][offset0], B[4 * (baseId + i) + 2][offset0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][offset1], B[4 * (baseId + i) + 2][offset1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][offset2], B[4 * (baseId + i) + 2][offset2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * (baseId + i) + 3][offset0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * (baseId + i) + 3][offset1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * (baseId + i) + 3][offset2] * delta_lambda[i]);
				}
			}

		}

		__device__ void SF_BanRotConstraint123
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			Vec3f tmp(eta[baseId], eta[baseId + 1], eta[baseId + 2]);
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]) + J[4 * (baseId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (baseId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]);
				}
			}

			Vec3f delta_lambda = omega * (K_3[baseId] * tmp);

			const int offset0 = gId % 3;
			const int offset1 = (gId + 1) % 3;
			const int offset2 = (gId + 2) % 3;

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * (baseId + i)][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * (baseId + i)][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * (baseId + i)][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * (baseId + i) + 1][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * (baseId + i) + 1][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * (baseId + i) + 1][offset2] * delta_lambda[i]);

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][offset0], B[4 * (baseId + i) + 2][offset0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][offset1], B[4 * (baseId + i) + 2][offset1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][offset2], B[4 * (baseId + i) + 2][offset2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * (baseId + i) + 3][offset0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * (baseId + i) + 3][offset1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * (baseId + i) + 3][offset2] * delta_lambda[i]);
				}
			}

		}

		__device__ void SF_AllowRotConstraint12
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			Vec2f tmp(eta[baseId], eta[baseId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]) + J[4 * (baseId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (baseId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Vec2f delta_lambda = omega * (K_2[baseId] * tmp);
			const int offset0 = gId % 3;
			const int offset1 = (gId + 1) % 3;
			const int offset2 = (gId + 2) % 3;
			for (int i = 0; i < 2; i++)
			{
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * (baseId + i)][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * (baseId + i)][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * (baseId + i)][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * (baseId + i) + 1][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * (baseId + i) + 1][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * (baseId + i) + 1][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][offset0], B[4 * (baseId + i) + 2][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][offset1], B[4 * (baseId + i) + 2][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][offset2], B[4 * (baseId + i) + 2][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * (baseId + i) + 3][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * (baseId + i) + 3][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * (baseId + i) + 3][offset2] * delta_lambda[i]);
			}

		}

		__device__ void SF_AnchorTRANSConstraint12
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			Vec2f tmp(eta[baseId], eta[baseId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (baseId + i)].dot(impulse[idx1 * 2]) + J[4 * (baseId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (baseId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (baseId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Vec2f delta_lambda = omega * (K_2[baseId] * tmp);
			const int offset0 = gId % 3;
			const int offset1 = (gId + 1) % 3;
			const int offset2 = (gId + 2) % 3;
			for (int i = 0; i < 2; i++)
			{
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * (baseId + i)][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * (baseId + i)][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * (baseId + i)][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * (baseId + i) + 1][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * (baseId + i) + 1][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * (baseId + i) + 1][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][offset0], B[4 * (baseId + i) + 2][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][offset1], B[4 * (baseId + i) + 2][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][offset2], B[4 * (baseId + i) + 2][offset2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * (baseId + i) + 3][offset0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * (baseId + i) + 3][offset1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * (baseId + i) + 3][offset2] * delta_lambda[i]);
			}

		}

		__device__ void SF_JointHingeSlideMinConstraint
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			float tmp = eta[baseId];
			tmp -= J[4 * baseId].dot(impulse[idx1 * 2]) + J[4 * baseId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * baseId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * baseId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[baseId] > 0)
			{
				const int offset0 = gId % 3;
				const int offset1 = (gId + 1) % 3;
				const int offset2 = (gId + 2) % 3;

				float delta_lambda = tmp * K_1[baseId] * omega;
				lambda[baseId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * baseId][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * baseId][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * baseId][offset2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * baseId + 1][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * baseId + 1][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * baseId + 1][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][offset0], B[4 * baseId + 2][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset1], B[4 * baseId + 2][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset2], B[4 * baseId + 2][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * baseId + 3][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * baseId + 3][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * baseId + 3][offset2] * delta_lambda);
			}

		}

		__device__ void SF_JointHingeSlideMaxConstraint
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			float tmp = eta[baseId];
			tmp -= J[4 * baseId].dot(impulse[idx1 * 2]) + J[4 * baseId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * baseId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * baseId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[baseId] > 0)
			{
				const int offset0 = gId % 3;
				const int offset1 = (gId + 1) % 3;
				const int offset2 = (gId + 2) % 3;

				float delta_lambda = tmp * K_1[baseId] * omega;
				lambda[baseId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * baseId][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * baseId][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * baseId][offset2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * baseId + 1][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * baseId + 1][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * baseId + 1][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][offset0], B[4 * baseId + 2][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset1], B[4 * baseId + 2][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset2], B[4 * baseId + 2][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * baseId + 3][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * baseId + 3][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * baseId + 3][offset2] * delta_lambda);
			}

		}

		__device__ void SF_JointHingeSlideMoterConstraint
		(
			const int idx1,
			const int idx2,
			const int baseId,
			const int gId,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			float tmp = eta[baseId];
			tmp -= J[4 * baseId].dot(impulse[idx1 * 2]) + J[4 * baseId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * baseId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * baseId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[baseId] > 0)
			{
				const int offset0 = gId % 3;
				const int offset1 = (gId + 1) % 3;
				const int offset2 = (gId + 2) % 3;

				float delta_lambda = tmp * K_1[baseId] * omega;
				lambda[baseId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][offset0], B[4 * baseId][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset1], B[4 * baseId][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][offset2], B[4 * baseId][offset2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][offset0], B[4 * baseId + 1][offset0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset1], B[4 * baseId + 1][offset1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][offset2], B[4 * baseId + 1][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][offset0], B[4 * baseId + 2][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset1], B[4 * baseId + 2][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][offset2], B[4 * baseId + 2][offset2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][offset0], B[4 * baseId + 3][offset0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset1], B[4 * baseId + 3][offset1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][offset2], B[4 * baseId + 3][offset2] * delta_lambda);
			}

		}

		__device__  void SF_SliderJointJacobiIteration
		(
			const int blockId,
			const int blockDim,
			const int threadIdx,
			const int gId,
			const int size,
			const int base,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			const int tId = blockId * blockDim + threadIdx;
			const int cId = tId / 4;
			if (cId >= size)
				return;
			const int pId = tId - 4 * cId;
			const int idx1 = constraints[base + 8 * cId + pId].bodyId1;
			const int idx2 = constraints[base + 8 * cId + pId].bodyId2;

			if (pId == 0)
			{
				SF_AnchorTRANSConstraint12
				(
					idx1,
					idx2,
					base + 8 * cId,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else if (pId == 1)
			{
				SF_BanRotConstraint123
				(
					idx1,
					idx2,
					base + 8 * cId + 2,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);

			}
			else if (pId == 2)
			{
				SF_JointHingeSlideMinConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 5,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);

				SF_JointHingeSlideMaxConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 6,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else
			{
				SF_JointHingeSlideMoterConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 7,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);

			}

		}

		__device__  void SF_HingeJointJacobiIteration
		(
			const int blockId,
			const int blockDim,
			const int threadIdx,
			const int gId,
			const int size,
			const int base,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			const int tId = blockId * blockDim + threadIdx;
			const int cId = tId / 4;
			if (cId >= size)
				return;
			const int pId = tId - 4 * cId;
			const int idx1 = constraints[base + 8 * cId + pId].bodyId1;
			const int idx2 = constraints[base + 8 * cId + pId].bodyId2;

			if (pId == 0)
			{
				SF_AnchorEqualConstraint123
				(
					idx1,
					idx2,
					base + 8 * cId,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else if (pId == 1)
			{
				SF_AllowRotConstraint12
				(
					idx1,
					idx2,
					base + 8 * cId + 3,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else if (pId == 2)
			{
				SF_JointHingeSlideMinConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 5,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);

				SF_JointHingeSlideMaxConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 6,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else
			{
				SF_JointHingeSlideMoterConstraint
				(
					idx1,
					idx2,
					base + 8 * cId + 7,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
		}

		__device__  void SF_FixedJointJacobiIteration
		(
			const int blockId,
			const int blockDim,
			const int threadIdx,
			const int gId,
			const int size,
			const int base,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt,
			const float omega
		)
		{
			const int tId = blockId * blockDim + threadIdx;
			const int cId = tId / 2;
			if (cId >= size)
				return;
			const int pId = tId - 2 * cId;

			const int idx1 = constraints[base + 6 * cId + pId].bodyId1;
			const int idx2 = constraints[base + 6 * cId + pId].bodyId2;

			if (pId == 0)
			{
				SF_AnchorEqualConstraint123
				(
					idx1,
					idx2,
					base + 6 * cId,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);

			}
			else if (pId == 1)
			{
				SF_BanRotConstraint123
				(
					idx1,
					idx2,
					base + 6 * cId + 3,
					gId,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
		}

		__global__ void SF_JacobiIterationCompress(
			const int SliderJointBlockOffset,
			const int  SliderJointNum,
			const int SliderJointBase,
			const int HingeJointBlockOffset,
			const int  HingeJointNum,
			const int HingeJointBase,
			const int FixedJointBlockOffset,
			const int  FixedJointNum,
			const int FixedJointBase,
			float* lambda,
			Vec3f* impulse,
			Vec3f* J,
			Vec3f* B,
			float* eta,
			TConstraintPair<float>* constraints,
			int* nbq,
			float* K_1,
			Mat2f* K_2,
			Mat3f* K_3,
			float* mass,
			float* fricCoeffs,
			float mu,
			float g,
			float dt
		)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			const float omega = Real(1) / 5;  // 0.2
			if (blockIdx.x <= SliderJointBlockOffset && SliderJointNum > 0)
			{
				SF_SliderJointJacobiIteration
				(
					blockIdx.x,
					blockDim.x,
					threadIdx.x,
					tId,
					SliderJointNum,
					SliderJointBase,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else if (blockIdx.x <= HingeJointBlockOffset && HingeJointNum > 0)
			{
				SF_HingeJointJacobiIteration
				(
					blockIdx.x - SliderJointBlockOffset,
					blockDim.x,
					threadIdx.x,
					tId,
					HingeJointNum,
					HingeJointBase,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
			else if (blockIdx.x <= FixedJointBlockOffset && FixedJointNum > 0)
			{

				SF_FixedJointJacobiIteration
				(
					blockIdx.x - HingeJointBlockOffset,
					blockDim.x,
					threadIdx.x,
					tId,
					FixedJointNum,
					FixedJointBase,
					lambda,
					impulse,
					J,
					B,
					eta,
					constraints,
					nbq,
					K_1,
					K_2,
					K_3,
					mass,
					fricCoeffs,
					mu,
					g,
					dt,
					omega
				);
			}
		}
	}

	template<typename TDataType>
	void TJConstraintSolver<TDataType>::initializeJacobianCompress(Real dt)
	{
		auto topo = this->inDiscreteElements()->constDataPtr();
		int constraint_size = 0;
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		std::vector<int> baseIndex(3);
		baseIndex[0] = constraint_size; constraint_size += 8 * sliderJoint_size;
		baseIndex[1] = constraint_size; constraint_size += 8 * hingeJoint_size;
		baseIndex[2] = constraint_size; constraint_size += 6 * fixedJoint_size;
		if (constraint_size == 0)
		{
			return;
		}
	
		/// //////////////////////////
		mVelocityConstraints.resize(constraint_size);
		if (sliderJoint_size != 0)
		{
			auto& joints = topo->sliderJoints();
			dim3 blockSize(LEI_BLOCK_SIZE);
			uint32_t blockNum = (joints.size() + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
			dim3 gridSize(blockNum);
			LeiModify::SF_setUpSliderJointConstraintsCompress << <gridSize, blockSize >> >
				(
					mVelocityConstraints,
					joints,
					this->inCenter()->getData(),
					this->inRotationMatrix()->getData(),
					this->inQuaternion()->getData(),
					baseIndex[0]
					);
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();
			dim3 blockSize(LEI_BLOCK_SIZE);
			uint32_t blockNum = (joints.size() + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
			dim3 gridSize(blockNum);
			LeiModify::SF_setUpHingeJointConstraintsCompress << <gridSize, blockSize >> >
				(
					mVelocityConstraints,
					joints,
					this->inCenter()->getData(),
					this->inRotationMatrix()->getData(),
					this->inQuaternion()->getData(),
					baseIndex[1]
					);
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			dim3 blockSize(LEI_BLOCK_SIZE);
			uint32_t blockNum = (joints.size() + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
			dim3 gridSize(blockNum);
			LeiModify::SF_setUpFixedJointConstraintsCompress << <gridSize, blockSize >> >
				(
					mVelocityConstraints,
					joints,
					this->inRotationMatrix()->getData(),
					this->inQuaternion()->getData(),
					baseIndex[2]
					);

		}
		//////////////////////
		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mJ.resize(4 * constraint_size); // Vec3
		mB.resize(4 * constraint_size); // Vec3
		mK_1.resize(constraint_size); // float
		mK_2.resize(constraint_size); // Matrix2
		mK_3.resize(constraint_size); // MatrixX
		mEta.resize(constraint_size); // float
		mLambda.resize(constraint_size); // float

		mJ.reset();
		mB.reset();
		mK_1.reset();
		mK_2.reset();
		mK_3.reset();
		mEta.reset();
		mLambda.reset();

		mContactNumber.reset();

		calculateJacobianMatrix(
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			this->inRotationMatrix()->getData(),
			mVelocityConstraints
		);

		calculateK(
			mVelocityConstraints,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mK_1,
			mK_2,
			mK_3
		);

		mErrors.resize(constraint_size);
		mErrors.reset();

		calculateEtaVectorForPJSBaumgarte(
			mEta,
			mJ,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			mVelocityConstraints,
			mErrors,
			this->varSlop()->getValue(),
			this->varBaumgarteBias()->getValue(),
			this->varSubStepping()->getValue(),
			dt
		);
	}

	template<typename TDataType>
	void TJConstraintSolver<TDataType>::constrainCompress()
	{
		uint bodyNum = this->inCenter()->size();
		auto topo = this->inDiscreteElements()->constDataPtr();
		mImpulseC.resize(bodyNum * 2);
		mImpulseExt.resize(bodyNum * 2);
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();

		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0)
		{
			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}


			setUpContactsInLocalFrame(
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData()
			);


			Real dh = dt / this->varSubStepping()->getValue();

			for (int i = 0; i < this->varSubStepping()->getValue(); i++)
			{
				mImpulseExt.reset();
				if (this->varGravityEnabled()->getValue())
				{
					setUpGravity(
						mImpulseExt,
						this->varGravityValue()->getValue(),
						dh
					);
				}

				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseExt,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
				);

				// 清空约束冲量
				mImpulseC.reset();

				initializeJacobianCompress(dh);
				//
				int block_size = 0;
				int sliderJoint_size = topo->sliderJoints().size();
				int hingeJoint_size = topo->hingeJoints().size();
				int fixedJoint_size = topo->fixedJoints().size();
				std::vector<int> blockOffset(3);
				blockOffset[0] = block_size;  block_size += (4 * sliderJoint_size + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
				blockOffset[1] = block_size;  block_size += (4 * hingeJoint_size + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
				blockOffset[2] = block_size;  block_size += (2 * fixedJoint_size + (LEI_BLOCK_SIZE - 1)) / LEI_BLOCK_SIZE;
				dim3 blockSize(LEI_BLOCK_SIZE);
				dim3 gridSize(block_size);
				std::vector<int> baseIndex(3);
				int constraint_size = 0;
				baseIndex[0] = constraint_size; constraint_size += 8 * sliderJoint_size;
				baseIndex[1] = constraint_size; constraint_size += 8 * hingeJoint_size;
				baseIndex[2] = constraint_size; constraint_size += 6 * fixedJoint_size;
				
				Mat2f* K2_ptr = mK_2.begin();
				Mat3f* K3_ptr = mK_3.begin();
				for (int j = 0; j < this->varIterationNumberForVelocitySolver()->getValue(); j++)
				{
					LeiModify::SF_JacobiIterationCompress << <gridSize, blockSize >> >
					(
						blockOffset[0],
						sliderJoint_size,
						baseIndex[0],
						blockOffset[1],
						hingeJoint_size,
						baseIndex[1],
						blockOffset[2],
						fixedJoint_size,
						baseIndex[2],
						mLambda.begin(),
						mImpulseC.begin(),
						mJ.begin(),
						mB.begin(),
						mEta.begin(),
						mVelocityConstraints.begin(),
						mContactNumber.begin(),
						mK_1.begin(),
						K2_ptr,
						K3_ptr,
						this->inMass()->getData().begin(),
						this->inFrictionCoefficients()->getData().begin(),
						this->varFrictionCoefficient()->getData(),
						this->varGravityValue()->getData(),
						dh
					);

				}

				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseC,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
				);

				updateGesture(
					this->inAttribute()->getData(),
					this->inCenter()->getData(),
					this->inQuaternion()->getData(),
					this->inRotationMatrix()->getData(),
					this->inInertia()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					this->inInitialInertia()->getData(),
					dh
				);
			}
		}

		else
		{
			if (this->varGravityEnabled()->getValue())
			{
				setUpGravity(
					mImpulseExt,
					this->varGravityValue()->getValue(),
					dt
				);
			}


			updateVelocity(
				this->inAttribute()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				mImpulseExt,
				this->varLinearDamping()->getValue(),
				this->varAngularDamping()->getValue(),
				dt
			);

			updateGesture(
				this->inAttribute()->getData(),
				this->inCenter()->getData(),
				this->inQuaternion()->getData(),
				this->inRotationMatrix()->getData(),
				this->inInertia()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				this->inInitialInertia()->getData(),
				dt
			);
		}

	}


	template<typename TDataType>
	void TJConstraintSolver<TDataType>::constrain()
	{
		return constrainCompress();

		PROFILE_SCOPE("TJConstraintSolver::constrain");
			
		uint bodyNum = this->inCenter()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		mImpulseC.resize(bodyNum * 2);
		mImpulseExt.resize(bodyNum * 2);
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();


		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0)
		{
			this->inContacts()->clear();
			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			setUpContactsInLocalFrame(
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData()
			);

			Real dh = dt / this->varSubStepping()->getValue();

			for (int i = 0; i < this->varSubStepping()->getValue(); i++)
			{
				if (this->varGravityEnabled()->getValue())
				{
					setUpGravity(
						mImpulseExt,
						this->varGravityValue()->getValue(),
						dh
					);
				}

				setUpExternalForce(
					mImpulseExt,
					this->inExternalForce()->getData(),
					this->inExternalTorque()->getData(),
					this->inMass()->getData(),
					this->inInertia()->getData(),
					this->inAngularVelocity()->getData(),
					this->inRotationMatrix()->getData(),
					dh
				);

				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseExt,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
				);

				mImpulseC.reset();
				initializeJacobian(dh);
				// auto error = checkOutPositionError(
				// 	this->inCenter()->getData(),
				// 	mVelocityConstraints
				// );
				// printf(" Substep %d, Position Error = %f\n", i, error);

				{
					PROFILE_SCOPE("JacobiIterationLoop");
					// auto error0 = checkOutError(
					// 	mJ,
					// 	mImpulseC,
					// 	mVelocityConstraints,
					// 	mEta
					// );
					for (int j = 0; j < this->varIterationNumberForVelocitySolver()->getValue(); j++)
					{
						JacobiIteration(
							mLambda,
							mImpulseC,
							mJ,
							mB,
							mEta,
							mVelocityConstraints,
							mContactNumber,
							mK_1,
							mK_2,
							mK_3,
							this->inMass()->getData(),
							this->inFrictionCoefficients()->getData(),
							this->varFrictionCoefficient()->getData(),
							this->varGravityValue()->getData(),
							dh
						);

					}

				}


				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseC,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
				);

				updateGesture(
					this->inAttribute()->getData(),
					this->inCenter()->getData(),
					this->inQuaternion()->getData(),
					this->inRotationMatrix()->getData(),
					this->inInertia()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					this->inInitialInertia()->getData(),
					dh
				);

			}
		}

		else
		{
			if (this->varGravityEnabled()->getValue())
			{
				setUpGravity(
					mImpulseExt,
					this->varGravityValue()->getValue(),
					dt
				);
			}

			setUpExternalForce(
				mImpulseExt,
				this->inExternalForce()->getData(),
				this->inExternalTorque()->getData(),
				this->inMass()->getData(),
				this->inInertia()->getData(),
				this->inAngularVelocity()->getData(),
				this->inRotationMatrix()->getData(),
				dt
			);

			updateVelocity(
				this->inAttribute()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				mImpulseExt,
				this->varLinearDamping()->getValue(),
				this->varAngularDamping()->getValue(),
				dt
			);

			updateGesture(
				this->inAttribute()->getData(),
				this->inCenter()->getData(),
				this->inQuaternion()->getData(),
				this->inRotationMatrix()->getData(),
				this->inInertia()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				this->inInitialInertia()->getData(),
				dt
			);
		}

	}

	DEFINE_CLASS(TJConstraintSolver);
}
