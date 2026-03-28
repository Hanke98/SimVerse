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
TEST(CholeskyUniform, CompareWithCuSolverCuBlas)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int n = 192;
    constexpr int batch = 4;
    constexpr double tol = 1e-10;

    CArray<int> h_sizes;
    CArray<int> h_offsets;
    CArray<int> h_x_offsets;
    h_sizes.resize(batch);
    h_offsets.resize(batch);
    h_x_offsets.resize(batch);
    for (int b = 0; b < batch; ++b)
    {
        h_sizes[b] = n;
        h_offsets[b] = b * n * n;
        h_x_offsets[b] = b * n;
    }

    const int total_mat = batch * n * n;
    const int total_vec = batch * n;

    CArray<double> hA;
    CArray<double> hB;
    hA.resize(total_mat);
    hB.resize(total_vec);
    hA.reset();
    hB.reset();

    std::mt19937 rng(20260320);
    std::uniform_real_distribution<double> offdiag_dist(-0.2, 0.2);
    std::uniform_real_distribution<double> diag_dist(5.0, 20.0);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);

    for (int b = 0; b < batch; ++b)
    {
        const int mat_offset = h_offsets[b];
        const int vec_offset = h_x_offsets[b];

        Eigen::MatrixXd Lmat = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd xref = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                if (i == j) Lmat(i, j) = diag_dist(rng) + 1e-3 * i;
                else Lmat(i, j) = offdiag_dist(rng);
            }
            xref(i) = x_dist(rng);
        }

        const Eigen::MatrixXd A = Lmat * Lmat.transpose();
        const Eigen::VectorXd bvec = A * xref;

        for (int i = 0; i < n; ++i)
        {
            hB[vec_offset + i] = bvec(i);
            for (int j = 0; j < n; ++j)
            {
                hA[mat_offset + i * n + j] = A(i, j);
            }
        }
    }

    DArray<int> d_sizes, d_offsets, d_x_offsets;
    d_sizes.assign(h_sizes);
    d_offsets.assign(h_offsets);
    d_x_offsets.assign(h_x_offsets);

    DArray<double> dA_ours, dX_ours;
    dA_ours.assign(hA);
    dX_ours.assign(hB);
    CholeskyFactorizeHost(
        dA_ours.begin(), dA_ours.begin(),
        d_sizes.begin(), d_offsets.begin(),
        batch, CholeskyMethod::WavefrontTiled);
    cuSafeCall(cudaDeviceSynchronize());
    CholeskySolveHost(
        dA_ours.begin(), dX_ours.begin(),
        d_sizes.begin(), d_offsets.begin(), d_x_offsets.begin(),
        batch, CholeskyMethod::UniformTiled);
    cuSafeCall(cudaDeviceSynchronize());

    CArray<double> hL_ours, hX_ours;
    hL_ours.assign(dA_ours);
    hX_ours.assign(dX_ours);

    DArray<double> dA_cu, dX_cu;
    dA_cu.assign(hA);
    dX_cu.assign(hB);

    std::vector<double*> hA_ptr_rw(batch);
    std::vector<const double*> hA_ptr_ro(batch);
    std::vector<double*> hB_ptr(batch);
    for (int b = 0; b < batch; ++b)
    {
        hA_ptr_rw[b] = dA_cu.begin() + static_cast<size_t>(b) * n * n;
        hA_ptr_ro[b] = dA_cu.begin() + static_cast<size_t>(b) * n * n;
        hB_ptr[b] = dX_cu.begin() + static_cast<size_t>(b) * n;
    }

    double** dA_ptr_rw = nullptr;
    const double** dA_ptr_ro = nullptr;
    double** dB_ptr = nullptr;
    int* d_info = nullptr;
    cuSafeCall(cudaMalloc(&dA_ptr_rw, batch * sizeof(double*)));
    cuSafeCall(cudaMalloc(reinterpret_cast<void**>(&dA_ptr_ro), batch * sizeof(const double*)));
    cuSafeCall(cudaMalloc(&dB_ptr, batch * sizeof(double*)));
    cuSafeCall(cudaMalloc(&d_info, batch * sizeof(int)));
    cuSafeCall(cudaMemcpy(dA_ptr_rw, hA_ptr_rw.data(), batch * sizeof(double*), cudaMemcpyHostToDevice));
    cuSafeCall(cudaMemcpy(dA_ptr_ro, hA_ptr_ro.data(), batch * sizeof(const double*), cudaMemcpyHostToDevice));
    cuSafeCall(cudaMemcpy(dB_ptr, hB_ptr.data(), batch * sizeof(double*), cudaMemcpyHostToDevice));

    CuSolverCholeskyRunner<double> runner;
    ASSERT_TRUE(runner.Initialize());
    runner.Factorize(dA_cu.begin(), dA_ptr_rw, n, batch, d_info, true);
    runner.Solve(dA_cu.begin(), dX_cu.begin(), dA_ptr_ro, dB_ptr, n, batch, true);

    cuSafeCall(cudaDeviceSynchronize());

    CArray<int> h_info;
    h_info.resize(batch);
    cuSafeCall(cudaMemcpy(h_info.begin(), d_info, batch * sizeof(int), cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; ++b)
    {
        EXPECT_EQ(h_info[b], 0);
    }

    CArray<double> hL_cu, hX_cu;
    hL_cu.assign(dA_cu);
    hX_cu.assign(dX_cu);

    double max_rel_l = 0.0;
    double max_rel_x = 0.0;
    for (int b = 0; b < batch; ++b)
    {
        const int mo = h_offsets[b];
        const int xo = h_x_offsets[b];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                const double v0 = hL_ours[mo + i * n + j];
                // cuSOLVER stores matrices in column-major layout.
                const double v1 = hL_cu[mo + j * n + i];
                const double rel = std::abs(v0 - v1) / (std::abs(v1) + 1e-12);
                max_rel_l = std::max(max_rel_l, rel);
            }

            const double x0 = hX_ours[xo + i];
            const double x1 = hX_cu[xo + i];
            const double relx = std::abs(x0 - x1) / (std::abs(x1) + 1e-12);
            max_rel_x = std::max(max_rel_x, relx);
        }
    }

    EXPECT_LT(max_rel_l, tol);
    EXPECT_LT(max_rel_x, tol);

    runner.Release();
    if (dA_ptr_rw) { cuSafeCall(cudaFree(dA_ptr_rw)); }
    if (dA_ptr_ro) { cuSafeCall(cudaFree(reinterpret_cast<void*>(const_cast<double**>(dA_ptr_ro)))); }
    if (dB_ptr) { cuSafeCall(cudaFree(dB_ptr)); }
    if (d_info) { cuSafeCall(cudaFree(d_info)); }
}

} // namespace dyno
