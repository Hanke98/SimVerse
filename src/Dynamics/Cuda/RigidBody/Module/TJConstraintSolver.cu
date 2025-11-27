#include "TJConstraintSolver.h"
#include "SharedFuncsForRigidBody.h"
#include "Profiler.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TJConstraintSolver, TDataType)

	template<typename TDataType>
	TJConstraintSolver<TDataType>::TJConstraintSolver()
	    : ConstraintModule()
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
			    this->varFrictionEnabled()->getData());
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
			    mVelocityConstraints, joints, this->inCenter()->getData(), this->inRotationMatrix()->getData(), begin_index);
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
			    begin_index);
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
			    begin_index);
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
			    mVelocityConstraints, joints, this->inRotationMatrix()->getData(), this->inQuaternion()->getData(), begin_index);
			;
		}

		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size + 6 * fixedJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpPointJointConstraints(mVelocityConstraints, joints, this->inCenter()->getData(), begin_index);
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
		    mVelocityConstraints);

		calculateK(
		    mVelocityConstraints,
		    mJ,
		    mB,
		    this->inCenter()->getData(),
		    this->inInertia()->getData(),
		    this->inMass()->getData(),
		    mK_1,
		    mK_2,
		    mK_3);

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
		    dt);

		if (contact_size != 0)
		{
			calculateContactPoints(this->inContacts()->getData(), mContactNumber);
		}
	}

	template<typename TDataType>
	void TJConstraintSolver<TDataType>::constrain()
	{
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
			if (mContactsInLocalFrame.size() != this->inContacts()->size())
			{
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			setUpContactsInLocalFrame(
			    mContactsInLocalFrame, this->inContacts()->getData(), this->inCenter()->getData(), this->inRotationMatrix()->getData());

			Real dh = dt / this->varSubStepping()->getValue();

			for (int i = 0; i < this->varSubStepping()->getValue(); i++)
			{
				if (this->varGravityEnabled()->getValue())
				{
					setUpGravity(mImpulseExt, this->varGravityValue()->getValue(), dh);
				}

				setUpExternalForce(
				    mImpulseExt,
				    this->inExternalForce()->getData(),
				    this->inExternalTorque()->getData(),
				    this->inMass()->getData(),
				    this->inInertia()->getData(),
				    this->inAngularVelocity()->getData(),
				    this->inRotationMatrix()->getData(),
				    dh);

				updateVelocity(
				    this->inAttribute()->getData(),
				    this->inVelocity()->getData(),
				    this->inAngularVelocity()->getData(),
				    mImpulseExt,
				    this->varLinearDamping()->getValue(),
				    this->varAngularDamping()->getValue(),
				    dh);

				mImpulseC.reset();
				initializeJacobian(dh);
				// auto all_err = checkOutErrors(mErrors);
				// printf("Substep %d, constraint error before solve: %.12f \n", i, all_err);

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
						    dh);
					}
				}

				// {
				// 	CArray<Coord> mImpulseC_host;
				// 	mImpulseC_host.assign(mImpulseC);
				// 	for (int i = 0; i < mImpulseC_host.size(); i++)
				// 	{
				// 		printf(
				// 		    "impulseC[%d]: (%.12f, %.12f, %.12f) \n", i, mImpulseC_host[i][0], mImpulseC_host[i][1], mImpulseC_host[i][2]);
				// 	}
				// }
				updateVelocity(
				    this->inAttribute()->getData(),
				    this->inVelocity()->getData(),
				    this->inAngularVelocity()->getData(),
				    mImpulseC,
				    this->varLinearDamping()->getValue(),
				    this->varAngularDamping()->getValue(),
				    dh);

				updateGesture(
				    this->inAttribute()->getData(),
				    this->inCenter()->getData(),
				    this->inQuaternion()->getData(),
				    this->inRotationMatrix()->getData(),
				    this->inInertia()->getData(),
				    this->inVelocity()->getData(),
				    this->inAngularVelocity()->getData(),
				    this->inInitialInertia()->getData(),
				    dh);
			}
		}
		else
		{
			if (this->varGravityEnabled()->getValue())
			{
				setUpGravity(mImpulseExt, this->varGravityValue()->getValue(), dt);
			}

			setUpExternalForce(
			    mImpulseExt,
			    this->inExternalForce()->getData(),
			    this->inExternalTorque()->getData(),
			    this->inMass()->getData(),
			    this->inInertia()->getData(),
			    this->inAngularVelocity()->getData(),
			    this->inRotationMatrix()->getData(),
			    dt);

			updateVelocity(
			    this->inAttribute()->getData(),
			    this->inVelocity()->getData(),
			    this->inAngularVelocity()->getData(),
			    mImpulseExt,
			    this->varLinearDamping()->getValue(),
			    this->varAngularDamping()->getValue(),
			    dt);

			updateGesture(
			    this->inAttribute()->getData(),
			    this->inCenter()->getData(),
			    this->inQuaternion()->getData(),
			    this->inRotationMatrix()->getData(),
			    this->inInertia()->getData(),
			    this->inVelocity()->getData(),
			    this->inAngularVelocity()->getData(),
			    this->inInitialInertia()->getData(),
			    dt);
		}

		const int ENABLE_POST_STAB = 0;
		if constexpr (ENABLE_POST_STAB)
		{
			mImpulseC.reset();
			initializeJacobian(dt);
			DArray<Real> tempErrors;
			auto before_err = PostStablizationErrorValidate(
			    /**/
			    tempErrors,
			    mImpulseC,
			    mJ,
			    mErrors,
			    mVelocityConstraints);
			printf("before post stab, err0: %.12f\n", before_err);

			// auto all_err = checkOutErrors(mErrors);
			// printf("All constraint error before post-stabilization: %.12f \n", all_err);
			// mEta.reset();
			// mErrors.reset();
			// calculateEtaVectorForPJSBaumgarte(
			//     mEta,
			//     mJ,
			//     this->inVelocity()->getData(),
			//     this->inAngularVelocity()->getData(),
			//     this->inCenter()->getData(),
			//     this->inQuaternion()->getData(),
			//     mVelocityConstraints,
			//     mErrors,
			//     this->varSlop()->getValue(),
			//     1.0f,
			//     1,
			//     1.0f);
			// all_err = checkOutErrors(mErrors);
			// printf("All constraint error before post-stabilization (recalc eta): %.12f \n", all_err);

			{
				PROFILE_SCOPE("FinalJacobiIterationLoop");
				for (int j = 0; j < 30; j++)
				{
					PostStablization(
					    mLambda,
					    mImpulseC,
					    mJ,
					    mB,
					    mErrors,
					    mVelocityConstraints,
					    mContactNumber,
					    mK_1,
					    mK_2,
					    mK_3,
					    this->inMass()->getData(),
					    this->inFrictionCoefficients()->getData(),
					    this->varFrictionCoefficient()->getData(),
					    this->varGravityValue()->getData());
				}
				// validate
				DArray<Real> tempErrors;
				tempErrors.resize(mErrors.size());
				tempErrors.reset();
				auto after_err = PostStablizationErrorValidate(
				    /**/
				    tempErrors,
				    mImpulseC,
				    mJ,
				    mErrors,
				    mVelocityConstraints);
				printf("after post stab, err1: %.12f\n", after_err);
				// CArray<Real> tempErrors_host;
				// tempErrors_host.assign(tempErrors);
				// for (int i = 0; i < tempErrors_host.size(); i++)
				// {
				// 	printf("constraint %d, error: %.12f \n", i, tempErrors_host[i]);
				// }
			}

			DArray<Coord> dpVel;
			dpVel.resize(this->inCenter()->size());
			dpVel.reset();
			DArray<Coord> dpAngularVel;
			dpAngularVel.resize(this->inAngularVelocity()->size());
			dpAngularVel.reset();
			updateVelocity(
			    this->inAttribute()->getData(),
			    dpVel,
			    dpAngularVel,
			    mImpulseC,
			    this->varLinearDamping()->getValue(),
			    this->varAngularDamping()->getValue(),
			    1.0);

			{
				CArray<Coord> dpVel_host;
				CArray<Coord> dpAngularVel_host;
				CArray<Coord> mImpulseC_host;
				dpVel_host.assign(dpVel);
				dpAngularVel_host.assign(dpAngularVel);
				mImpulseC_host.assign(mImpulseC);
				for (int i = 0; i < dpVel_host.size(); i++)
				{
					// printf("dpVel[%d]: (%15.12f, %15.12f, %15.12f) \n", i, dpVel_host[i][0], dpVel_host[i][1], dpVel_host[i][2]);
					// printf(
					//     "dpOme[%d]: (%15.12f, %15.12f, %15.12f) \n",
					//     i,
					//     dpAngularVel_host[i][0],
					//     dpAngularVel_host[i][1],
					//     dpAngularVel_host[i][2]);
					printf(
					    "dp[%d]: (%15.12f %15.12f, %15.12f), (%15.12f, %15.12f, %15.12f)\n",
					    i,
					    mImpulseC_host[2 * i][0],
					    mImpulseC_host[2 * i][1],
					    mImpulseC_host[2 * i][2],
					    mImpulseC_host[2 * i + 1][0],
					    mImpulseC_host[2 * i + 1][1],
					    mImpulseC_host[2 * i + 1][2]);
				}
			}

			updateGesture(
			    this->inAttribute()->getData(),
			    this->inCenter()->getData(),
			    this->inQuaternion()->getData(),
			    this->inRotationMatrix()->getData(),
			    this->inInertia()->getData(),
			    dpVel,
			    dpAngularVel,
			    this->inInitialInertia()->getData(),
			    1.0);

			initializeJacobian(dt);
			auto err = checkOutErrors(mErrors);
			printf("Final constraint error after post-stabilization, err3= %.12f \n", err);

			// getchar();
		}
	}

	DEFINE_CLASS(TJConstraintSolver);
} // namespace dyno
