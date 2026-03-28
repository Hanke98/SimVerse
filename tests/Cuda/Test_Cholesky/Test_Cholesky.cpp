#include "gtest/gtest.h"

#include "Array/Array.h"
#include "SimNode/Utils/cholesky.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace dyno
{
namespace
{
struct DispatchResult
{
    double max_rel_lower_err = 0.0;
    double max_rel_x_err = 0.0;
};

DispatchResult RunDispatchCase(CholeskyMethod method, const std::vector<int>& sizes, bool inplace_required)
{
    const int num_blocks = static_cast<int>(sizes.size());

    CArray<int> h_block_sizes;
    CArray<int> h_block_offsets;
    CArray<int> h_x_offsets;
    h_block_sizes.resize(num_blocks);
    h_block_offsets.resize(num_blocks);
    h_x_offsets.resize(num_blocks);

    int total_mat_size = 0;
    int total_vec_size = 0;
    for (int b = 0; b < num_blocks; ++b)
    {
        h_block_sizes[b] = sizes[b];
        h_block_offsets[b] = total_mat_size;
        h_x_offsets[b] = total_vec_size;
        total_mat_size += sizes[b] * sizes[b];
        total_vec_size += sizes[b];
    }

    CArray<double> hA;
    CArray<double> hL_ref;
    CArray<double> hX_ref;
    CArray<double> hB;
    hA.resize(total_mat_size);
    hL_ref.resize(total_mat_size);
    hX_ref.resize(total_vec_size);
    hB.resize(total_vec_size);
    hA.reset();
    hL_ref.reset();
    hX_ref.reset();
    hB.reset();

    std::mt19937 rng(20260320 + static_cast<int>(method));
    std::uniform_real_distribution<double> offdiag_dist(-0.2, 0.2);
    std::uniform_real_distribution<double> diag_dist(5.0, 20.0);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);

    for (int b = 0; b < num_blocks; ++b)
    {
        const int m = h_block_sizes[b];
        const int mat_offset = h_block_offsets[b];
        const int vec_offset = h_x_offsets[b];

        Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(m, m);
        Eigen::VectorXd xref = Eigen::VectorXd::Zero(m);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                if (i == j) Lmat(i, j) = diag_dist(rng) + 1e-3 * i;
                else Lmat(i, j) = offdiag_dist(rng);
            }
            xref(i) = x_dist(rng);
        }

        const Eigen::MatrixXd Amat = Lmat * Lmat.transpose();
        const Eigen::VectorXd bvec = Amat * xref;

        for (int i = 0; i < m; ++i)
        {
            hX_ref[vec_offset + i] = xref(i);
            hB[vec_offset + i] = bvec(i);
            for (int j = 0; j < m; ++j)
            {
                hA[mat_offset + i * m + j] = Amat(i, j);
                hL_ref[mat_offset + i * m + j] = (i >= j) ? Lmat(i, j) : 0.0;
            }
        }
    }

    DArray<double> dA;
    DArray<double> dL;
    DArray<double> dX;
    DArray<int> d_block_sizes;
    DArray<int> d_block_offsets;
    DArray<int> d_x_offsets;

    dA.assign(hA);
    dL.resize(total_mat_size);
    dL.reset();
    dX.assign(hB);
    d_block_sizes.assign(h_block_sizes);
    d_block_offsets.assign(h_block_offsets);
    d_x_offsets.assign(h_x_offsets);

    if (inplace_required)
    {
        dL.assign(hA);
        CholeskyFactorizeHost(
            dL.begin(), dL.begin(),
            d_block_sizes.begin(), d_block_offsets.begin(),
            num_blocks, method);
    }
    else
    {
        CholeskyFactorizeHost(
            dA.begin(), dL.begin(),
            d_block_sizes.begin(), d_block_offsets.begin(),
            num_blocks, method);
    }
    cuSafeCall(cudaDeviceSynchronize());

    CholeskySolveHost(
        dL.begin(), dX.begin(),
        d_block_sizes.begin(), d_block_offsets.begin(), d_x_offsets.begin(),
        num_blocks, method);
    cuSafeCall(cudaDeviceSynchronize());

    CArray<double> hL_gpu;
    CArray<double> hX_gpu;
    hL_gpu.assign(dL);
    hX_gpu.assign(dX);

    DispatchResult result;
    for (int b = 0; b < num_blocks; ++b)
    {
        const int m = h_block_sizes[b];
        const int mat_offset = h_block_offsets[b];
        const int vec_offset = h_x_offsets[b];

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                const double ref = hL_ref[mat_offset + i * m + j];
                const double got = hL_gpu[mat_offset + i * m + j];
                const double rel = std::abs(got - ref) / (std::abs(ref) + 1e-12);
                result.max_rel_lower_err = std::max(result.max_rel_lower_err, rel);
            }

            const double x_ref = hX_ref[vec_offset + i];
            const double x_got = hX_gpu[vec_offset + i];
            const double x_rel = std::abs(x_got - x_ref) / (std::abs(x_ref) + 1e-12);
            result.max_rel_x_err = std::max(result.max_rel_x_err, x_rel);
        }
    }
    return result;
}
} // namespace

TEST(CholeskyDispatch, Simplest)
{
    const DispatchResult r = RunDispatchCase(CholeskyMethod::Simplest, {64, 80, 96}, false);
    EXPECT_LT(r.max_rel_lower_err, 1e-8);
    EXPECT_LT(r.max_rel_x_err, 1e-8);
}

TEST(CholeskyDispatch, SingleTiled)
{
    const DispatchResult r = RunDispatchCase(CholeskyMethod::SingleTiled, {64, 80, 96}, false);
    EXPECT_LT(r.max_rel_lower_err, 1e-8);
    EXPECT_LT(r.max_rel_x_err, 1e-8);
}

TEST(CholeskyDispatch, UniformTiled)
{
    const DispatchResult r = RunDispatchCase(CholeskyMethod::UniformTiled, {192, 192, 192, 192}, true);
    EXPECT_LT(r.max_rel_lower_err, 1e-8);
    EXPECT_LT(r.max_rel_x_err, 1e-8);
}

TEST(CholeskyDispatch, PaddedTiled)
{
    const DispatchResult r = RunDispatchCase(CholeskyMethod::PaddedTiled, {128, 173, 211, 256}, true);
    EXPECT_LT(r.max_rel_lower_err, 1e-8);
    EXPECT_LT(r.max_rel_x_err, 1e-8);
}

TEST(CholeskyDispatch, WavefrontTiled)
{
    const DispatchResult r = RunDispatchCase(CholeskyMethod::WavefrontTiled, {192, 192, 192, 192}, true);
    EXPECT_LT(r.max_rel_lower_err, 1e-8);
    EXPECT_LT(r.max_rel_x_err, 1e-8);
}

} // namespace dyno
