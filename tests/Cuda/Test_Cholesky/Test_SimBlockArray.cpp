#include "gtest/gtest.h"

#include "SimNode/Utils/SimBlockArray.h"

#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace dyno
{
namespace
{

HostBlockArray<int> MakeHostBlockArray(const std::vector<int>& sizes)
{
    HostBlockArray<int> h;
    EXPECT_TRUE(h.BuildFromSizes(sizes));

    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        int* ptr = h.BlockPtr(b);
        const int len = h.BlockLength(b);
        for (int i = 0; i < len; ++i)
        {
            ptr[i] = b * 1000 + i;
        }
    }
    return h;
}

} // namespace

TEST(SimBlockArray, HostBuildFromSizes)
{
    HostBlockArray<int> h;
    ASSERT_TRUE(h.BuildFromSizes(std::vector<int>{3, 5, 2}));

    ASSERT_TRUE(h.IsValid());
    EXPECT_EQ(h.NumBlocks(), 3);
    EXPECT_EQ(h.TotalSize(), 10);

    ASSERT_EQ(h.Sizes().Size(), 3);
    ASSERT_EQ(h.Offsets().Size(), 3);

    EXPECT_EQ(h.Sizes()[0], 3);
    EXPECT_EQ(h.Sizes()[1], 5);
    EXPECT_EQ(h.Sizes()[2], 2);

    EXPECT_EQ(h.Offsets()[0], 0);
    EXPECT_EQ(h.Offsets()[1], 3);
    EXPECT_EQ(h.Offsets()[2], 8);

    EXPECT_EQ(h.BlockOffset(0), 0);
    EXPECT_EQ(h.BlockOffset(1), 3);
    EXPECT_EQ(h.BlockOffset(2), 8);

    EXPECT_EQ(h.BlockLength(0), 3);
    EXPECT_EQ(h.BlockLength(1), 5);
    EXPECT_EQ(h.BlockLength(2), 2);
}

TEST(SimBlockArray, RejectInvalidSizes)
{
    HostBlockArray<int> h;

    EXPECT_FALSE(h.BuildFromSizes(std::vector<int>{3, 0, 2}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());

    EXPECT_FALSE(h.BuildFromSizes(std::vector<int>{3, -1, 2}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());
}

TEST(SimBlockArray, HostDevHostRoundTrip)
{
    const std::vector<int> sizes{4, 1, 3};
    HostBlockArray<int> h_src = MakeHostBlockArray(sizes);

    DevBlockArray<int> d;
    ASSERT_TRUE(d.BuildFromSizes(sizes));
    ASSERT_TRUE(d.Assign(h_src.Data()));

    HostBlockArray<int> h_dst;
    ASSERT_TRUE(d.Download(h_dst));

    ASSERT_TRUE(h_dst.IsValid());
    EXPECT_EQ(h_dst.NumBlocks(), h_src.NumBlocks());
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());

    for (int b = 0; b < h_src.NumBlocks(); ++b)
    {
        EXPECT_EQ(h_dst.Sizes()[b], h_src.Sizes()[b]);
        EXPECT_EQ(h_dst.Offsets()[b], h_src.Offsets()[b]);
    }

    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(SimBlockArray, AssignHostBlockArrayToDev)
{
    const std::vector<int> sizes{2, 6, 4};
    HostBlockArray<int> h_src = MakeHostBlockArray(sizes);

    DevBlockArray<int> d;
    ASSERT_TRUE(d.Assign(h_src));
    ASSERT_TRUE(d.IsValid());

    HostBlockArray<int> h_dst;
    ASSERT_TRUE(d.Download(h_dst));

    EXPECT_EQ(h_dst.NumBlocks(), h_src.NumBlocks());
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());

    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(SimBlockArray, AssignDevToDev)
{
    const std::vector<int> sizes{5, 3};
    HostBlockArray<int> h_src = MakeHostBlockArray(sizes);

    DevBlockArray<int> d0;
    ASSERT_TRUE(d0.Assign(h_src));

    DevBlockArray<int> d1;
    ASSERT_TRUE(d1.Assign(d0));
    ASSERT_TRUE(d1.IsValid());

    HostBlockArray<int> h_dst;
    ASSERT_TRUE(d1.Download(h_dst));

    EXPECT_EQ(h_dst.NumBlocks(), h_src.NumBlocks());
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());
    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(SimBlockArray, RebuildStability)
{
    DevBlockArray<int> d;

    ASSERT_TRUE(d.BuildFromSizes(std::vector<int>{2, 2, 2}));
    EXPECT_EQ(d.NumBlocks(), 3);
    EXPECT_EQ(d.TotalSize(), 6);

    ASSERT_TRUE(d.BuildFromSizes(std::vector<int>{8, 1, 9, 4}));
    EXPECT_EQ(d.NumBlocks(), 4);
    EXPECT_EQ(d.TotalSize(), 22);

    ASSERT_TRUE(d.BuildFromSizes(std::vector<int>{3}));
    EXPECT_EQ(d.NumBlocks(), 1);
    EXPECT_EQ(d.TotalSize(), 3);
    EXPECT_TRUE(d.IsValid());
}

TEST(SimBlockArray, MultiSizeMatricesFromBinaryFiles)
{
    constexpr std::array<int, 6> kDims{128, 256, 512, 1024, 2048, 4096};
    constexpr std::array<const char*, 6> kPaths{
        "tests/Cuda/Test_Cholesky/data/spd_128.bin",
        "tests/Cuda/Test_Cholesky/data/spd_256.bin",
        "tests/Cuda/Test_Cholesky/data/spd_512.bin",
        "tests/Cuda/Test_Cholesky/data/spd_1024.bin",
        "tests/Cuda/Test_Cholesky/data/spd_2048.bin",
        "tests/Cuda/Test_Cholesky/data/spd_4096.bin"};

    std::vector<int> block_sizes;
    block_sizes.reserve(kDims.size());
    for (int n : kDims) block_sizes.push_back(n * n);  // element count per block

    HostBlockArray<double> h;
    ASSERT_TRUE(h.BuildFromSizes(block_sizes));
    ASSERT_TRUE(h.IsValid());
    ASSERT_EQ(h.NumBlocks(), static_cast<int>(kDims.size()));

    int expect_elem_offset = 0;
    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        const int n = kDims[b];
        const int elem_count = n * n;

        EXPECT_EQ(h.BlockLength(b), elem_count);
        EXPECT_EQ(h.BlockOffset(b), expect_elem_offset);
        const int byte_offset = h.BlockOffset(b) * static_cast<int>(sizeof(double));
        EXPECT_EQ(byte_offset, expect_elem_offset * static_cast<int>(sizeof(double)));

        std::ifstream fin(kPaths[b], std::ios::binary);
        ASSERT_TRUE(fin.good()) << "failed to open " << kPaths[b];

        std::vector<double> tmp(elem_count);
        fin.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(elem_count * sizeof(double)));
        ASSERT_EQ(fin.gcount(), static_cast<std::streamsize>(elem_count * sizeof(double)))
            << "file size mismatch: " << kPaths[b];

        double* dst = h.BlockPtr(b);
        for (int i = 0; i < elem_count; ++i) dst[i] = tmp[i];

        expect_elem_offset += elem_count;
    }

    EXPECT_EQ(h.TotalSize(), expect_elem_offset);

    DevBlockArray<double> d;
    ASSERT_TRUE(d.Assign(h));
    ASSERT_TRUE(d.IsValid());

    HostBlockArray<double> h_roundtrip;
    ASSERT_TRUE(d.Download(h_roundtrip));
    ASSERT_TRUE(h_roundtrip.IsValid());
    ASSERT_EQ(h_roundtrip.TotalSize(), h.TotalSize());
    ASSERT_EQ(h_roundtrip.NumBlocks(), h.NumBlocks());

    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        EXPECT_EQ(h_roundtrip.BlockOffset(b), h.BlockOffset(b));
        EXPECT_EQ(h_roundtrip.BlockLength(b), h.BlockLength(b));
    }

    for (int i = 0; i < h.TotalSize(); ++i)
    {
        EXPECT_EQ(h_roundtrip.Data()[i], h.Data()[i]);
    }
}

} // namespace dyno
