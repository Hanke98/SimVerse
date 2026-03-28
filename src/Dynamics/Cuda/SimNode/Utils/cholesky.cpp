#include "Array/Array.h"
#include "cholesky.h"
#include "spdlog/spdlog.h"
#include <Eigen/Dense>

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

        BatchCholeskyFactorizeHost(
            dA,
            dL,
            dBlockSizes,
            dBlockOffsets,
            num_blocks);

        cuSafeCall(cudaDeviceSynchronize());

        BatchCholeskySolveHost(
            dL,
            dX,
            dBlockSizes,
            dBlockOffsets,
            dXOffsets,
            num_blocks);
        
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
        block_size_list.insert(block_size_list.end(), 10, 64);
        block_size_list.insert(block_size_list.end(), 10, 80);
        block_size_list.insert(block_size_list.end(), 10, 96);

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

        BlockCholeskySingleTileHost(
            dA,
            dL,
            dBlockSizes,
            dBlockOffsets,
            num_blocks);

        cuSafeCall(cudaDeviceSynchronize());

        BlockCholeskySolveSingleTileHost(
            dL,
            dX,
            dBlockSizes,
            dBlockOffsets,
            dXOffsets,
            num_blocks);

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
}