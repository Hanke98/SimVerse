#include "gtest/gtest.h"

#include "Array/Array.h"
#include "SimNode/Utils/cholesky.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace dyno
{
namespace
{
// Edit these constants directly for profiling configuration.
constexpr const char* kInputBinPath = "tests/Cuda/Test_Cholesky/data/spd_512.bin";
constexpr int kProfileNumBlocks = 128;
constexpr int kProfileWarmupIters = 5;
constexpr int kProfileTimedIters = 50;

struct ProfileStats
{
    float avg_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
};

ProfileStats MeasureGpuKernelLoop(
    int warmup_iters,
    int timed_iters,
    const DArray<double>& dA_ref,
    const DArray<double>& dX_ref,
    DArray<double>& dA_work,
    DArray<double>& dX_work,
    const std::function<void()>& fn_factorize_solve)
{
    for (int i = 0; i < warmup_iters; ++i)
    {
        cuSafeCall(cudaMemcpy(dA_work.begin(), dA_ref.begin(), dA_ref.size() * sizeof(double), cudaMemcpyDeviceToDevice));
        cuSafeCall(cudaMemcpy(dX_work.begin(), dX_ref.begin(), dX_ref.size() * sizeof(double), cudaMemcpyDeviceToDevice));
        fn_factorize_solve();
        cuSafeCall(cudaDeviceSynchronize());
    }

    std::vector<float> times;
    times.reserve(timed_iters);

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    cuSafeCall(cudaEventCreate(&ev_start));
    cuSafeCall(cudaEventCreate(&ev_stop));

    for (int i = 0; i < timed_iters; ++i)
    {
        cuSafeCall(cudaMemcpy(dA_work.begin(), dA_ref.begin(), dA_ref.size() * sizeof(double), cudaMemcpyDeviceToDevice));
        cuSafeCall(cudaMemcpy(dX_work.begin(), dX_ref.begin(), dX_ref.size() * sizeof(double), cudaMemcpyDeviceToDevice));

        cuSafeCall(cudaEventRecord(ev_start));
        fn_factorize_solve();
        cuSafeCall(cudaEventRecord(ev_stop));
        cuSafeCall(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        cuSafeCall(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        times.push_back(ms);
    }

    cuSafeCall(cudaEventDestroy(ev_start));
    cuSafeCall(cudaEventDestroy(ev_stop));

    ProfileStats s;
    s.min_ms = *std::min_element(times.begin(), times.end());
    s.max_ms = *std::max_element(times.begin(), times.end());
    float sum = 0.0f;
    for (float v : times) sum += v;
    s.avg_ms = sum / static_cast<float>(times.size());
    return s;
}

std::pair<std::vector<double>, int> LoadSingleSquareMatrixBin(const char* path)
{
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin.is_open())
    {
        throw std::runtime_error(std::string("Failed to open bin file: ") + path);
    }

    const std::streamsize bytes = fin.tellg();
    if (bytes <= 0 || (bytes % static_cast<std::streamsize>(sizeof(double))) != 0)
    {
        throw std::runtime_error("Invalid .bin payload for double matrix");
    }
    fin.seekg(0, std::ios::beg);

    const size_t elems = static_cast<size_t>(bytes / static_cast<std::streamsize>(sizeof(double)));
    const int n = static_cast<int>(std::sqrt(static_cast<double>(elems)));
    if (static_cast<size_t>(n) * static_cast<size_t>(n) != elems)
    {
        throw std::runtime_error("Input .bin is not a single square matrix (n*n)");
    }

    std::vector<double> a(elems);
    fin.read(reinterpret_cast<char*>(a.data()), bytes);
    if (!fin)
    {
        throw std::runtime_error("Failed to read full matrix payload");
    }
    return {std::move(a), n};
}

} // namespace

TEST(CholeskyProfiling, UniformVsCuSolver)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    std::vector<double> base_a;
    int block_size = 0;
    ASSERT_NO_THROW(
        [&]() {
            auto ret = LoadSingleSquareMatrixBin(kInputBinPath);
            base_a = std::move(ret.first);
            block_size = ret.second;
        }()
    );
    ASSERT_GT(block_size, 0);
    const int num_blocks = kProfileNumBlocks;
    const int warmup_iters = kProfileWarmupIters;
    const int timed_iters = kProfileTimedIters;

    const int total_mat = num_blocks * block_size * block_size;
    const int total_vec = num_blocks * block_size;

    CArray<int> h_sizes;
    CArray<int> h_offsets;
    CArray<int> h_x_offsets;
    h_sizes.resize(num_blocks);
    h_offsets.resize(num_blocks);
    h_x_offsets.resize(num_blocks);
    for (int b = 0; b < num_blocks; ++b)
    {
        h_sizes[b] = block_size;
        h_offsets[b] = b * block_size * block_size;
        h_x_offsets[b] = b * block_size;
    }

    CArray<double> hA;
    CArray<double> hB;
    CArray<double> hXRef;
    hA.resize(total_mat);
    hB.resize(total_vec);
    hXRef.resize(total_vec);
    hA.reset();
    hB.reset();
    hXRef.reset();

    std::mt19937 rng(20260320);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
    std::vector<double> x_local(block_size, 0.0);

    for (int b = 0; b < num_blocks; ++b)
    {
        const int mo = h_offsets[b];
        const int xo = h_x_offsets[b];

        for (int idx = 0; idx < block_size * block_size; ++idx)
            hA[mo + idx] = base_a[static_cast<size_t>(idx)];

        for (int i = 0; i < block_size; ++i)
        {
            x_local[i] = x_dist(rng);
            hXRef[xo + i] = x_local[i];
        }

        for (int i = 0; i < block_size; ++i)
        {
            double bi = 0.0;
            const int row = mo + i * block_size;
            for (int j = 0; j < block_size; ++j)
                bi += hA[row + j] * x_local[j];
            hB[xo + i] = bi;
        }
    }

    DArray<int> d_sizes, d_offsets, d_x_offsets;
    d_sizes.assign(h_sizes);
    d_offsets.assign(h_offsets);
    d_x_offsets.assign(h_x_offsets);

    DArray<double> dA_ref, dX_ref;
    dA_ref.assign(hA);
    dX_ref.assign(hB);

    DArray<double> dA_uniform, dX_uniform;
    DArray<double> dA_cu, dX_cu;
    dA_uniform.resize(total_mat);
    dX_uniform.resize(total_vec);
    dA_cu.resize(total_mat);
    dX_cu.resize(total_vec);

    std::vector<double*> hA_ptr_rw(num_blocks);
    std::vector<const double*> hA_ptr_ro(num_blocks);
    std::vector<double*> hB_ptr(num_blocks);
    for (int b = 0; b < num_blocks; ++b)
    {
        hA_ptr_rw[b] = dA_cu.begin() + static_cast<size_t>(b) * block_size * block_size;
        hA_ptr_ro[b] = dA_cu.begin() + static_cast<size_t>(b) * block_size * block_size;
        hB_ptr[b] = dX_cu.begin() + static_cast<size_t>(b) * block_size;
    }

    double** dA_ptr_rw = nullptr;
    const double** dA_ptr_ro = nullptr;
    double** dB_ptr = nullptr;
    int* d_info = nullptr;
    cuSafeCall(cudaMalloc(&dA_ptr_rw, num_blocks * sizeof(double*)));
    cuSafeCall(cudaMalloc(reinterpret_cast<void**>(&dA_ptr_ro), num_blocks * sizeof(const double*)));
    cuSafeCall(cudaMalloc(&dB_ptr, num_blocks * sizeof(double*)));
    cuSafeCall(cudaMalloc(&d_info, num_blocks * sizeof(int)));
    cuSafeCall(cudaMemcpy(dA_ptr_rw, hA_ptr_rw.data(), num_blocks * sizeof(double*), cudaMemcpyHostToDevice));
    cuSafeCall(cudaMemcpy(dA_ptr_ro, hA_ptr_ro.data(), num_blocks * sizeof(const double*), cudaMemcpyHostToDevice));
    cuSafeCall(cudaMemcpy(dB_ptr, hB_ptr.data(), num_blocks * sizeof(double*), cudaMemcpyHostToDevice));

    CuSolverCholeskyRunner<double> runner;
    ASSERT_TRUE(runner.Initialize());

    auto uniform_call = [&]() {
        CholeskyFactorizeHost(
            dA_uniform.begin(), dA_uniform.begin(),
            d_sizes.begin(), d_offsets.begin(),
            num_blocks, CholeskyMethod::UniformTiled);
        CholeskySolveHost(
            dA_uniform.begin(), dX_uniform.begin(),
            d_sizes.begin(), d_offsets.begin(), d_x_offsets.begin(),
            num_blocks, CholeskyMethod::UniformTiled);
    };

    auto cusolver_call = [&]() {
        runner.Factorize(dA_cu.begin(), dA_ptr_rw, block_size, num_blocks, d_info, true);
        runner.Solve(dA_cu.begin(), dX_cu.begin(), dA_ptr_ro, dB_ptr, block_size, num_blocks, true);
    };

    const ProfileStats s_uniform = MeasureGpuKernelLoop(
        warmup_iters, timed_iters, dA_ref, dX_ref, dA_uniform, dX_uniform, uniform_call);
    const ProfileStats s_cusolver = MeasureGpuKernelLoop(
        warmup_iters, timed_iters, dA_ref, dX_ref, dA_cu, dX_cu, cusolver_call);

    CArray<int> h_info;
    h_info.resize(num_blocks);
    cuSafeCall(cudaMemcpy(h_info.begin(), d_info, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; ++b)
    {
        EXPECT_EQ(h_info[b], 0);
    }

    CArray<double> hX_uniform, hX_cu;
    hX_uniform.assign(dX_uniform);
    hX_cu.assign(dX_cu);

    double max_rel_uniform = 0.0;
    double max_rel_cu = 0.0;
    for (int i = 0; i < total_vec; ++i)
    {
        const double ref = hXRef[i];
        const double r0 = std::abs(hX_uniform[i] - ref) / (std::abs(ref) + 1e-12);
        const double r1 = std::abs(hX_cu[i] - ref) / (std::abs(ref) + 1e-12);
        if (r0 > max_rel_uniform) max_rel_uniform = r0;
        if (r1 > max_rel_cu) max_rel_cu = r1;
    }

    EXPECT_LT(max_rel_uniform, 1e-8);
    EXPECT_LT(max_rel_cu, 1e-8);

    std::printf(
        "\n[Profiling] block_size=%d, num_blocks=%d, warmup=%d, iters=%d\n"
        "  UniformTiled : avg=%.3f ms, min=%.3f ms, max=%.3f ms\n"
        "  CuSolverWrap : avg=%.3f ms, min=%.3f ms, max=%.3f ms\n"
        "  Speedup (Uniform/CuSolver) = %.3f x\n",
        block_size, num_blocks, warmup_iters, timed_iters,
        s_uniform.avg_ms, s_uniform.min_ms, s_uniform.max_ms,
        s_cusolver.avg_ms, s_cusolver.min_ms, s_cusolver.max_ms,
        s_uniform.avg_ms / std::max(1e-6f, s_cusolver.avg_ms));

    runner.Release();
    if (dA_ptr_rw) cuSafeCall(cudaFree(dA_ptr_rw));
    if (dA_ptr_ro) cuSafeCall(cudaFree(reinterpret_cast<void*>(const_cast<double**>(dA_ptr_ro))));
    if (dB_ptr) cuSafeCall(cudaFree(dB_ptr));
    if (d_info) cuSafeCall(cudaFree(d_info));
}

} // namespace dyno
