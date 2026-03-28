#include "Array/Array.h"
#include "cholesky.h"
#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <random>

namespace dyno
{
	void TestI()
	{
		spdlog::info("====================   Test Basic Cholesky   ===================");

		// 最大 block size
		int n = 350;

		// block 个数也是 n，size 分别为 1,2,3,...,n
		int num_blocks = n;
		CArray<int> hBlockSizes;
		CArray<int> hBlockOffsets;
		CArray<int> hXOffsets;
		hBlockSizes.resize(num_blocks);
		hBlockOffsets.resize(num_blocks);
		hXOffsets.resize(num_blocks);

		// fill sizes and offsets
		int total_size = 0;
		int total_x_size = 0;
		for (int b = 0; b < num_blocks; b++)
		{
			int block_size = b + 1;
			hBlockSizes[b] = block_size;
			hBlockOffsets[b] = total_size;
			hXOffsets[b] = total_x_size;
			total_size += block_size * block_size;
			total_x_size += block_size;
		}

		CArray<double> hA;
		CArray<double> hLRef;
		CArray<double> hLInit;
		CArray<double> hLGpu;
		CArray<double> hX;

		hA.resize(total_size);
		hLRef.resize(total_size);
		hLInit.resize(total_size);
		hLGpu.resize(total_size);
		hX.resize(total_x_size);

		hA.reset();
		hLRef.reset();
		hLInit.reset();
		hLGpu.reset();
		hX.reset();

		// construct Ax = b with known L, where A = L * L^T, x = [1, 1, ..., 1], b = A * x
		for (int b = 0; b < num_blocks; b++)
		{
			int m = hBlockSizes[b];
			int offset = hBlockOffsets[b];
			int Xoffset = hXOffsets[b];

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(m, m);
			Eigen::VectorXd x = Eigen::VectorXd::Ones(m);

			double val = 1.0f;
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j <= i; j++)
				{
					// Lmat(i, j) = val;
					// val += 0.01f;
					if (i == j)
						Lmat(i, j) = 5.0 + 0.01 * i;
					else
						Lmat(i, j) = val + 0.001 * (i + j + 1);
				}
			}
			x = Lmat * Lmat.transpose() * x;

			Eigen::MatrixXd Amat = Lmat * Lmat.transpose();

			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < m; j++)
				{
					hA[offset + i * m + j] = Amat(i, j);
					hLRef[offset + i * m + j] = Lmat(i, j);
				}
				hX[Xoffset + i] = x(i);
			}
		}

		DArray<double> dA;
		DArray<double> dL;
		DArray<int> dBlockSizes;
		DArray<int> dBlockOffsets;
		DArray<int> dXOffsets;
		DArray<double> dX;

		dA.assign(hA);
		dL.assign(hLInit);
		dX.assign(hX);
		dBlockSizes.assign(hBlockSizes);
		dBlockOffsets.assign(hBlockOffsets);
		dXOffsets.assign(hXOffsets);

		BatchCholeskyFactorizeHost(dA.begin(), dL.begin(), dBlockSizes.begin(), dBlockOffsets.begin(), num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		BatchCholeskySolveHost(dL.begin(), dX.begin(), dBlockSizes.begin(), dBlockOffsets.begin(), dXOffsets.begin(), num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		hLGpu.assign(dL);
		hX.assign(dX);

		double max_err = 0.0f;
		double max_x_err = 0.0f;

		for (int b = 0; b < num_blocks; b++)
		{
			int m = hBlockSizes[b];
			int offset = hBlockOffsets[b];
			int Xoffset = hXOffsets[b];

			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j <= i; j++)
				{
					double err = std::abs(hLGpu[offset + i * m + j] - hLRef[offset + i * m + j]);
					max_err = std::max(max_err, err) / (std::abs(hLRef[offset + i * m + j]) + 1e-8f);
				}
				double x_err = std::abs(hX[Xoffset + i] - 1.0f);
				max_x_err = std::max(max_x_err, x_err);
			}
		}

		spdlog::info("Max lower-triangle error = {}", max_err);
		spdlog::info("Max X error = {}", max_x_err);

		double tol = 1e-4f;
		if (max_err < tol)
			spdlog::info("Cholesky test PASSED");
		else
			spdlog::error("Cholesky test FAILED");
		spdlog::info("====================   End Test Basic Cholesky   ===================");
	}

	void TestII()
	{
		spdlog::info("====================   Test Block Cholesky   ===================");

		constexpr int MAX_N = 96;

		// 20 个 64, 20 个 80, 20 个 96
		std::vector<int> block_size_list;
		block_size_list.insert(block_size_list.end(), 100, 64);
		block_size_list.insert(block_size_list.end(), 100, 80);
		block_size_list.insert(block_size_list.end(), 100, 96);

		const int num_blocks = static_cast<int>(block_size_list.size());

		CArray<int> hBlockSizes;
		CArray<int> hBlockOffsets;
		CArray<int> hXOffsets;
		hBlockSizes.resize(num_blocks);
		hBlockOffsets.resize(num_blocks);
		hXOffsets.resize(num_blocks);

		int total_size = 0;
		int total_x_size = 0;
		for (int b = 0; b < num_blocks; ++b)
		{
			int m = block_size_list[b];
			hBlockSizes[b] = m;
			hBlockOffsets[b] = total_size;
			hXOffsets[b] = total_x_size;
			total_size += m * m;
			total_x_size += m;
		}

		CArray<double> hA;
		CArray<double> hLRef;
		CArray<double> hLInit;
		CArray<double> hLGpu;
		CArray<double> hX;

		hA.resize(total_size);
		hLRef.resize(total_size);
		hLInit.resize(total_size);
		hLGpu.resize(total_size);
		hX.resize(total_x_size);

		hA.reset();
		hLRef.reset();
		hLInit.reset();
		hLGpu.reset();
		hX.reset();

		// 构造 A = L * L^T，其中 LRef 是已知下三角
		for (int b = 0; b < num_blocks; ++b)
		{
			int m = hBlockSizes[b];
			int offset = hBlockOffsets[b];

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(m, m);
			Eigen::VectorXd x = Eigen::VectorXd::Ones(m);

			// 构造一个数值稳定、严格正定的下三角 L
			// 对角给大一些，非对角给较小扰动
			for (int i = 0; i < m; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					if (i == j)
					{
						Lmat(i, j) = 8.0 + 0.02 * i + 0.001 * b;
					}
					else
					{
						Lmat(i, j) = 0.01 * (i - j + 1) + 0.0001 * (i + j + b + 1);
					}
				}
			}

			Eigen::MatrixXd Amat = Lmat * Lmat.transpose();
			x = Amat * x; // 生成对应的 x

			for (int i = 0; i < m; ++i)
			{
				for (int j = 0; j < m; ++j)
				{
					hA[offset + i * m + j] = Amat(i, j);
					hLRef[offset + i * m + j] = Lmat(i, j);
					hX[hXOffsets[b] + i] = x(i);
				}
			}
		}

		DArray<double> dA;
		DArray<double> dL;
		DArray<int> dBlockSizes;
		DArray<int> dBlockOffsets;
		DArray<int> dXOffsets;
		DArray<double> dX;
		dA.assign(hA);
		dL.assign(hLInit);
		dBlockSizes.assign(hBlockSizes);
		dBlockOffsets.assign(hBlockOffsets);
		dXOffsets.assign(hXOffsets);
		dX.assign(hX);

		BlockCholeskySingleTileHost(dA, dL, dBlockSizes, dBlockOffsets, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		BlockCholeskySolveSingleTileHost(dL, dX, dBlockSizes, dBlockOffsets, dXOffsets, num_blocks);

		hLGpu.assign(dL);
		hX.assign(dX);

		double max_abs_err = 0.0;
		double max_rel_err = 0.0;
		double max_x_err = 0.0;
		for (int b = 0; b < num_blocks; ++b)
		{
			int m = hBlockSizes[b];
			int offset = hBlockOffsets[b];

			for (int i = 0; i < m; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					double ref = hLRef[offset + i * m + j];
					double got = hLGpu[offset + i * m + j];
					double abs_err = std::abs(got - ref);
					double rel_err = abs_err / (std::abs(ref) + 1e-12);

					max_abs_err = std::max(max_abs_err, abs_err);
					max_rel_err = std::max(max_rel_err, rel_err);
				}
				double x_ref = 1.0; // 因为我们构造的 x 是全 1
				double x_got = hX[hXOffsets[b] + i];
				double x_err = std::abs(x_got - x_ref);
				max_x_err = std::max(max_x_err, x_err);
				// 如果你希望顺手确认上三角被清零，也可以保留这段
				for (int j = i + 1; j < m; ++j)
				{
					double upper_val = hLGpu[offset + i * m + j];
					max_abs_err = std::max(max_abs_err, std::abs(upper_val));
				}
			}
		}

		spdlog::info("MAX_N = {}", MAX_N);
		spdlog::info("num_blocks = {}", num_blocks);
		spdlog::info("total_size = {}", total_size);
		spdlog::info("Max abs lower-triangle error = {}", max_abs_err);
		spdlog::info("Max rel lower-triangle error = {}", max_rel_err);
		spdlog::info("Max X error = {}", max_x_err);

		const double tol = 1e-8;
		if (max_rel_err < tol)
			spdlog::info("Block Cholesky test PASSED");
		else
			spdlog::error("Block Cholesky test FAILED");

		spdlog::info("====================   End Test Block Cholesky   ===================");
	}

	void TestIII()
	{
		spdlog::info("====================   Test Uniform Blocked Cholesky 512x512   ====================");

		constexpr int block_size = 512;
		constexpr int num_blocks = 100; // 先测 10 个 uniform blocks
		constexpr int total_size = num_blocks * block_size * block_size;

		CArray<double> hA;
		CArray<double> hLRef;
		CArray<double> hAGpu;

		hA.resize(total_size);
		hLRef.resize(total_size);
		hAGpu.resize(total_size);

		hA.reset();
		hLRef.reset();
		hAGpu.reset();

		// 构造 A = L * L^T，其中 LRef 是已知下三角
		for (int b = 0; b < num_blocks; ++b)
		{
			const int offset = b * block_size * block_size;

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(block_size, block_size);

			// 构造一个稳定的下三角 L
			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					if (i == j)
					{
						// 对角稍大一些，保证 SPD 且条件数别太差
						Lmat(i, j) = 10.0 + 0.01 * i + 0.001 * b;
					}
					else
					{
						// 非对角较小
						Lmat(i, j) = 0.02 * (i - j + 1) + 0.0001 * (i + j + b + 1);
					}
				}
			}

			Eigen::MatrixXd Amat = Lmat * Lmat.transpose();

			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j < block_size; ++j)
				{
					hA[offset + i * block_size + j] = Amat(i, j);
					hLRef[offset + i * block_size + j] = Lmat(i, j);
				}
			}
		}

		DArray<double> dA;
		dA.assign(hA);

		// 原地 blocked Cholesky
		UniformBlockCholeskyFactorizeWithTileHost(dA.begin(), block_size, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		hAGpu.assign(dA);

		double max_abs_err = 0.0;
		double max_rel_err = 0.0;

		for (int b = 0; b < num_blocks; ++b)
		{
			const int offset = b * block_size * block_size;

			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					double ref = hLRef[offset + i * block_size + j];
					double got = hAGpu[offset + i * block_size + j];

					double abs_err = std::abs(got - ref);
					double rel_err = abs_err / (std::abs(ref) + 1e-12);

					max_abs_err = std::max(max_abs_err, abs_err);
					max_rel_err = std::max(max_rel_err, rel_err);
				}

				// // 检查上三角是否接近 0（如果你的对角 tile store 只写下三角，这里通常应为 0）
				// for (int j = i + 1; j < block_size; ++j)
				// {
				//     double upper_val = hAGpu[offset + i * block_size + j];
				//     max_abs_err = std::max(max_abs_err, std::abs(upper_val));
				// }
			}
		}

		spdlog::info("block_size = {}", block_size);
		spdlog::info("num_blocks = {}", num_blocks);
		spdlog::info("total_size = {}", total_size);
		spdlog::info("Max abs lower-triangle error = {}", max_abs_err);
		spdlog::info("Max rel lower-triangle error = {}", max_rel_err);

		const double tol = 1e-8;
		if (max_rel_err < tol)
			spdlog::info("Uniform blocked Cholesky test PASSED");
		else
			spdlog::error("Uniform blocked Cholesky test FAILED");

		spdlog::info("====================   End Test Uniform Blocked Cholesky 512x512   ===================");
	}

	void TestIV()
	{
		spdlog::info("====================   Test Uniform Blocked Cholesky Random Dense L + Solve   ====================");

		constexpr int block_size = 512;
		constexpr int num_blocks = 4;
		constexpr int total_mat_size = num_blocks * block_size * block_size;
		constexpr int total_vec_size = num_blocks * block_size;

		CArray<double> hA;
		CArray<double> hARef;
		CArray<double> hLRef;
		CArray<double> hAGpu;

		CArray<double> hXRef; // 真解
		CArray<double> hB; // 右端 b = A * x_ref
		CArray<double> hXGpu; // GPU solve 得到的解

		hA.resize(total_mat_size);
		hARef.resize(total_mat_size);
		hLRef.resize(total_mat_size);
		hAGpu.resize(total_mat_size);

		hXRef.resize(total_vec_size);
		hB.resize(total_vec_size);
		hXGpu.resize(total_vec_size);

		hA.reset();
		hARef.reset();
		hLRef.reset();
		hAGpu.reset();
		hXRef.reset();
		hB.reset();
		hXGpu.reset();

		std::mt19937 rng(20260317);
		std::uniform_real_distribution<double> offdiag_dist(-0.5, 0.5);
		std::uniform_real_distribution<double> diag_dist(1.0, 15.0);
		std::uniform_real_distribution<double> x_dist(-1.0, 1.0);

		for (int b = 0; b < num_blocks; ++b)
		{
			const int mat_offset = b * block_size * block_size;
			const int vec_offset = b * block_size;

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(block_size, block_size);
			Eigen::VectorXd xref = Eigen::VectorXd::Zero(block_size);

			// 1) 随机构造 dense lower-triangular L
			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					if (i == j)
					{
						Lmat(i, j) = diag_dist(rng) + 0.001 * i + 0.0001 * b;
					}
					else
					{
						Lmat(i, j) = offdiag_dist(rng);
					}
				}
			}

			// 2) 构造 A = L L^T
			Eigen::MatrixXd Amat = Lmat * Lmat.transpose();

			// 3) 构造参考真解 x_ref
			for (int i = 0; i < block_size; ++i)
			{
				xref(i) = x_dist(rng);
			}

			// 4) 构造右端 b = A * x_ref
			Eigen::VectorXd bvec = Amat * xref;

			// 写回 host arrays
			for (int i = 0; i < block_size; ++i)
			{
				hXRef[vec_offset + i] = xref(i);
				hB[vec_offset + i] = bvec(i);

				for (int j = 0; j < block_size; ++j)
				{
					hA[mat_offset + i * block_size + j] = Amat(i, j);
					hARef[mat_offset + i * block_size + j] = Amat(i, j);
					hLRef[mat_offset + i * block_size + j] = Lmat(i, j);
				}
			}
		}

		DArray<double> dA;
		DArray<double> dX;

		dA.assign(hA);
		dX.assign(hB); // solve 输入是 b

		// ------------------------------------------------------------
		// 1) GPU blocked Cholesky, in-place: A -> L
		// ------------------------------------------------------------
		UniformBlockCholeskyFactorizeWithTileHost(dA.begin(), block_size, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		// 保存 factorize 结果用于检查
		hAGpu.assign(dA);

		// ------------------------------------------------------------
		// 2) GPU solve: x = A^{-1} b
		//    dA 中现在已经是 L
		//    dX 输入时是 b，输出时变成 x
		// ------------------------------------------------------------
		UniformBlockCholeskySolveWithTileHost(dA.begin(), dX.begin(), block_size, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		hXGpu.assign(dX);

		// ------------------------------------------------------------
		// 3) 检查 factorize 误差
		// ------------------------------------------------------------
		double max_abs_lower_err = 0.0;
		double max_rel_lower_err = 0.0;
		double max_upper_abs = 0.0;

		double max_recon_abs_err = 0.0;
		double max_recon_rel_err = 0.0;

		for (int b = 0; b < num_blocks; ++b)
		{
			const int mat_offset = b * block_size * block_size;

			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					double ref = hLRef[mat_offset + i * block_size + j];
					double got = hAGpu[mat_offset + i * block_size + j];

					double abs_err = std::abs(got - ref);
					double rel_err = abs_err / (std::abs(ref) + 1e-12);

					max_abs_lower_err = std::max(max_abs_lower_err, abs_err);
					max_rel_lower_err = std::max(max_rel_lower_err, rel_err);
				}

				// 上三角只做观测；为了后续重构，直接清零 host 侧缓存里的上三角
				for (int j = i + 1; j < block_size; ++j)
				{
					max_upper_abs = std::max(max_upper_abs, std::abs(hAGpu[mat_offset + i * block_size + j]));
					hAGpu[mat_offset + i * block_size + j] = 0.0;
				}
			}

			// reconstruction check: A_ref ?= L_gpu * L_gpu^T
			Eigen::MatrixXd Lgpu = Eigen::MatrixXd::Zero(block_size, block_size);
			Eigen::MatrixXd Aref = Eigen::MatrixXd::Zero(block_size, block_size);

			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j < block_size; ++j)
				{
					Aref(i, j) = hARef[mat_offset + i * block_size + j];
					if (i >= j)
						Lgpu(i, j) = hAGpu[mat_offset + i * block_size + j];
				}
			}

			Eigen::MatrixXd Arecon = Lgpu * Lgpu.transpose();
			Eigen::MatrixXd Diff = Arecon - Aref;

			double recon_abs_err = Diff.cwiseAbs().maxCoeff();
			double recon_rel_err = Diff.norm() / (Aref.norm() + 1e-12);

			max_recon_abs_err = std::max(max_recon_abs_err, recon_abs_err);
			max_recon_rel_err = std::max(max_recon_rel_err, recon_rel_err);
		}

		// ------------------------------------------------------------
		// 4) 检查 solve 误差
		// ------------------------------------------------------------
		double max_abs_x_err = 0.0;
		double max_rel_x_err = 0.0;

		for (int i = 0; i < total_vec_size; ++i)
		{
			double ref = hXRef[i];
			double got = hXGpu[i];

			double abs_err = std::abs(got - ref);
			double rel_err = abs_err / (std::abs(ref) + 1e-12);

			max_abs_x_err = std::max(max_abs_x_err, abs_err);
			max_rel_x_err = std::max(max_rel_x_err, rel_err);
		}

		// ------------------------------------------------------------
		// 5) 打印结果
		// ------------------------------------------------------------
		spdlog::info("block_size = {}", block_size);
		spdlog::info("num_blocks = {}", num_blocks);
		spdlog::info("total_mat_size = {}", total_mat_size);
		spdlog::info("total_vec_size = {}", total_vec_size);

		spdlog::info("Max abs lower-triangle error = {}", max_abs_lower_err);
		spdlog::info("Max rel lower-triangle error = {}", max_rel_lower_err);
		spdlog::info("Max abs upper-triangle residual = {}", max_upper_abs);

		spdlog::info("Max abs reconstruction error = {}", max_recon_abs_err);
		spdlog::info("Max rel reconstruction error = {}", max_recon_rel_err);

		spdlog::info("Max abs solve-x error = {}", max_abs_x_err);
		spdlog::info("Max rel solve-x error = {}", max_rel_x_err);

		const double tol_lower = 1e-8;
		const double tol_recon = 1e-8;
		const double tol_solve = 1e-8;

		if (max_rel_lower_err < tol_lower && max_recon_rel_err < tol_recon && max_rel_x_err < tol_solve)
		{
			spdlog::info("Random dense L blocked Cholesky + solve test PASSED");
		}
		else
		{
			spdlog::error("Random dense L blocked Cholesky + solve test FAILED");
		}

		spdlog::info("====================   End Test Uniform Blocked Cholesky Random Dense L + Solve   ===================");
	}

	void TestLowerSolve()
	{
		spdlog::info("====================   Test Lower Solve Only   ====================");

		constexpr int block_size = 512;
		constexpr int num_blocks = 4;
		constexpr int total_mat_size = num_blocks * block_size * block_size;
		constexpr int total_vec_size = num_blocks * block_size;

		CArray<double> hL;
		CArray<double> hYRef;
		CArray<double> hB;
		CArray<double> hYGpu;

		hL.resize(total_mat_size);
		hYRef.resize(total_vec_size);
		hB.resize(total_vec_size);
		hYGpu.resize(total_vec_size);

		hL.reset();
		hYRef.reset();
		hB.reset();
		hYGpu.reset();

		std::mt19937 rng(20260318);
		std::uniform_real_distribution<double> offdiag_dist(-0.5, 0.5);
		std::uniform_real_distribution<double> diag_dist(1.0, 15.0);
		std::uniform_real_distribution<double> y_dist(-1.0, 1.0);

		for (int b = 0; b < num_blocks; ++b)
		{
			const int mat_offset = b * block_size * block_size;
			const int vec_offset = b * block_size;

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(block_size, block_size);
			Eigen::VectorXd yref = Eigen::VectorXd::Zero(block_size);

			// 构造随机 dense lower triangular L
			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					if (i == j)
						Lmat(i, j) = diag_dist(rng) + 0.001 * i + 0.0001 * b;
					else
						Lmat(i, j) = offdiag_dist(rng);
				}
			}

			// 构造参考解 y_ref
			for (int i = 0; i < block_size; ++i)
			{
				yref(i) = y_dist(rng);
			}

			// 构造右端 b = L * y_ref
			Eigen::VectorXd bvec = Lmat * yref;

			for (int i = 0; i < block_size; ++i)
			{
				hYRef[vec_offset + i] = yref(i);
				hB[vec_offset + i] = bvec(i);

				for (int j = 0; j < block_size; ++j)
				{
					hL[mat_offset + i * block_size + j] = Lmat(i, j);
				}
			}
		}

		DArray<double> dL;
		DArray<double> dX;

		dL.assign(hL);
		dX.assign(hB); // solve 输入是 b

		// 只跑 lower solve: L y = b
		UniformBlockCholeskySolveWithTileHost(dL.begin(), dX.begin(), block_size, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		hYGpu.assign(dX);

		double max_abs_y_err = 0.0;
		double max_rel_y_err = 0.0;

		// 额外加一个整体范数误差
		double max_vec_rel_err = 0.0;

		for (int b = 0; b < num_blocks; ++b)
		{
			const int vec_offset = b * block_size;

			Eigen::VectorXd yref(block_size);
			Eigen::VectorXd ygpu(block_size);

			for (int i = 0; i < block_size; ++i)
			{
				double ref = hYRef[vec_offset + i];
				double got = hYGpu[vec_offset + i];

				yref(i) = ref;
				ygpu(i) = got;

				double abs_err = std::abs(got - ref);
				double rel_err = abs_err / (std::abs(ref) + 1e-12);

				max_abs_y_err = std::max(max_abs_y_err, abs_err);
				max_rel_y_err = std::max(max_rel_y_err, rel_err);
			}

			double vec_rel_err = (ygpu - yref).norm() / (yref.norm() + 1e-12);
			max_vec_rel_err = std::max(max_vec_rel_err, vec_rel_err);
		}

		spdlog::info("block_size = {}", block_size);
		spdlog::info("num_blocks = {}", num_blocks);
		spdlog::info("Max abs lower-solve y error = {}", max_abs_y_err);
		spdlog::info("Max rel lower-solve y error = {}", max_rel_y_err);
		spdlog::info("Max vector rel lower-solve y error = {}", max_vec_rel_err);

		const double tol_rel = 1e-8;
		const double tol_vec = 1e-8;

		if (max_rel_y_err < tol_rel && max_vec_rel_err < tol_vec)
			spdlog::info("Lower solve only test PASSED");
		else
			spdlog::error("Lower solve only test FAILED");

		spdlog::info("====================   End Test Lower Solve Only   ===================");
	}

	void TestUpperSolve()
	{
		spdlog::info("====================   Test Upper Solve Only   ====================");

		constexpr int block_size = 512;
		constexpr int num_blocks = 4;
		constexpr int total_mat_size = num_blocks * block_size * block_size;
		constexpr int total_vec_size = num_blocks * block_size;

		CArray<double> hL;
		CArray<double> hXRef;
		CArray<double> hY;
		CArray<double> hXGpu;

		hL.resize(total_mat_size);
		hXRef.resize(total_vec_size);
		hY.resize(total_vec_size);
		hXGpu.resize(total_vec_size);

		hL.reset();
		hXRef.reset();
		hY.reset();
		hXGpu.reset();

		std::mt19937 rng(20260318);
		std::uniform_real_distribution<double> offdiag_dist(-0.5, 0.5);
		std::uniform_real_distribution<double> diag_dist(1.0, 15.0);
		std::uniform_real_distribution<double> x_dist(-1.0, 1.0);

		for (int b = 0; b < num_blocks; ++b)
		{
			const int mat_offset = b * block_size * block_size;
			const int vec_offset = b * block_size;

			Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(block_size, block_size);
			Eigen::VectorXd xref = Eigen::VectorXd::Zero(block_size);

			// 构造随机 dense lower triangular L
			for (int i = 0; i < block_size; ++i)
			{
				for (int j = 0; j <= i; ++j)
				{
					if (i == j)
						Lmat(i, j) = diag_dist(rng) + 0.001 * i + 0.0001 * b;
					else
						Lmat(i, j) = offdiag_dist(rng);
				}
			}

			// 构造参考解 x_ref
			for (int i = 0; i < block_size; ++i)
			{
				xref(i) = x_dist(rng);
			}

			// 构造右端 y = L^T * x_ref
			Eigen::VectorXd yvec = Lmat.transpose() * xref;

			for (int i = 0; i < block_size; ++i)
			{
				hXRef[vec_offset + i] = xref(i);
				hY[vec_offset + i] = yvec(i);

				for (int j = 0; j < block_size; ++j)
				{
					hL[mat_offset + i * block_size + j] = Lmat(i, j);
				}
			}
		}

		DArray<double> dL;
		DArray<double> dX;

		dL.assign(hL);
		dX.assign(hY); // upper solve 输入是 y

		// 只跑 upper solve: L^T x = y
		UniformBlockCholeskySolveWithTileHost(dL.begin(), dX.begin(), block_size, num_blocks);

		cuSafeCall(cudaDeviceSynchronize());

		hXGpu.assign(dX);

		double max_abs_x_err = 0.0;
		double max_rel_x_err = 0.0;
		double max_vec_rel_err = 0.0;

		for (int b = 0; b < num_blocks; ++b)
		{
			const int vec_offset = b * block_size;

			Eigen::VectorXd xref(block_size);
			Eigen::VectorXd xgpu(block_size);

			for (int i = 0; i < block_size; ++i)
			{
				double ref = hXRef[vec_offset + i];
				double got = hXGpu[vec_offset + i];

				xref(i) = ref;
				xgpu(i) = got;

				double abs_err = std::abs(got - ref);
				double rel_err = abs_err / (std::abs(ref) + 1e-12);

				max_abs_x_err = std::max(max_abs_x_err, abs_err);
				max_rel_x_err = std::max(max_rel_x_err, rel_err);
			}

			double vec_rel_err = (xgpu - xref).norm() / (xref.norm() + 1e-12);
			max_vec_rel_err = std::max(max_vec_rel_err, vec_rel_err);
		}

		spdlog::info("block_size = {}", block_size);
		spdlog::info("num_blocks = {}", num_blocks);
		spdlog::info("Max abs upper-solve x error = {}", max_abs_x_err);
		spdlog::info("Max rel upper-solve x error = {}", max_rel_x_err);
		spdlog::info("Max vector rel upper-solve x error = {}", max_vec_rel_err);

		const double tol_rel = 1e-8;
		const double tol_vec = 1e-8;

		if (max_rel_x_err < tol_rel && max_vec_rel_err < tol_vec)
			spdlog::info("Upper solve only test PASSED");
		else
			spdlog::error("Upper solve only test FAILED");

		spdlog::info("====================   End Test Upper Solve Only   ===================");
	}

} // namespace dyno