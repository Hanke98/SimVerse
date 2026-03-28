#include "gtest/gtest.h"

#include "SimNode/Utils/SimBlockMatrix.h"

#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace dyno
{
namespace
{

HostBlockMatrix<int> MakeHostBlockMat(const std::vector<int>& rows, const std::vector<int>& cols)
{
    HostBlockMatrix<int> h;
    EXPECT_TRUE(h.BuildFromShapes(rows, cols));

    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        int* ptr = h.BlockPtr(b);
        const int r = h.BlockRows(b);
        const int c = h.BlockCols(b);
        for (int i = 0; i < r; ++i)
        {
            for (int j = 0; j < c; ++j)
            {
                ptr[i * c + j] = b * 100000 + i * 1000 + j;
            }
        }
    }
    return h;
}

} // namespace

TEST(BlockMatrix, HostBuildFromShapes)
{
    HostBlockMatrix<int> h;
    ASSERT_TRUE(h.BuildFromShapes(std::vector<int>{2, 3, 4}, std::vector<int>{3, 2, 1}));
    ASSERT_TRUE(h.IsValid());

    EXPECT_EQ(h.NumBlocks(), 3);
    EXPECT_EQ(h.TotalSize(), 2 * 3 + 3 * 2 + 4 * 1);

    EXPECT_EQ(h.BlockRows(0), 2);
    EXPECT_EQ(h.BlockCols(0), 3);
    EXPECT_EQ(h.BlockLength(0), 6);
    EXPECT_EQ(h.BlockOffset(0), 0);

    EXPECT_EQ(h.BlockRows(1), 3);
    EXPECT_EQ(h.BlockCols(1), 2);
    EXPECT_EQ(h.BlockLength(1), 6);
    EXPECT_EQ(h.BlockOffset(1), 6);

    EXPECT_EQ(h.BlockRows(2), 4);
    EXPECT_EQ(h.BlockCols(2), 1);
    EXPECT_EQ(h.BlockLength(2), 4);
    EXPECT_EQ(h.BlockOffset(2), 12);
}

TEST(BlockMatrix, HostBuildFromSquares)
{
    HostBlockMatrix<int> h;
    ASSERT_TRUE(h.BuildFromSquares(std::vector<int>{2, 4, 3}));
    ASSERT_TRUE(h.IsValid());

    EXPECT_EQ(h.NumBlocks(), 3);
    EXPECT_EQ(h.BlockRows(0), 2);
    EXPECT_EQ(h.BlockCols(0), 2);
    EXPECT_EQ(h.BlockRows(1), 4);
    EXPECT_EQ(h.BlockCols(1), 4);
    EXPECT_EQ(h.BlockRows(2), 3);
    EXPECT_EQ(h.BlockCols(2), 3);
    EXPECT_EQ(h.TotalSize(), 4 + 16 + 9);
}

TEST(BlockMatrix, RejectInvalidShapes)
{
    HostBlockMatrix<int> h;

    EXPECT_FALSE(h.BuildFromShapes(std::vector<int>{2, 3}, std::vector<int>{2}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());

    EXPECT_FALSE(h.BuildFromShapes(std::vector<int>{2, 0}, std::vector<int>{2, 3}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());

    EXPECT_FALSE(h.BuildFromShapes(std::vector<int>{2, -1}, std::vector<int>{2, 3}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());
}

TEST(BlockMatrix, HostDevHostRoundTrip)
{
    const std::vector<int> rows{2, 3, 1};
    const std::vector<int> cols{3, 2, 5};
    HostBlockMatrix<int> h_src = MakeHostBlockMat(rows, cols);

    DevBlockMatrix<int> d;
    ASSERT_TRUE(d.BuildFromShapes(rows, cols));
    ASSERT_TRUE(d.Assign(h_src.Data()));
    ASSERT_TRUE(d.IsValid());

    HostBlockMatrix<int> h_dst;
    ASSERT_TRUE(d.Download(h_dst));
    ASSERT_TRUE(h_dst.IsValid());

    EXPECT_EQ(h_dst.NumBlocks(), h_src.NumBlocks());
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());

    for (int b = 0; b < h_src.NumBlocks(); ++b)
    {
        EXPECT_EQ(h_dst.BlockRows(b), h_src.BlockRows(b));
        EXPECT_EQ(h_dst.BlockCols(b), h_src.BlockCols(b));
        EXPECT_EQ(h_dst.BlockOffset(b), h_src.BlockOffset(b));
        EXPECT_EQ(h_dst.BlockLength(b), h_src.BlockLength(b));
    }

    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(BlockMatrix, AssignHostBlockMatrixToDev)
{
    const std::vector<int> rows{4, 2};
    const std::vector<int> cols{3, 5};
    HostBlockMatrix<int> h_src = MakeHostBlockMat(rows, cols);

    DevBlockMatrix<int> d;
    ASSERT_TRUE(d.Assign(h_src));
    ASSERT_TRUE(d.IsValid());

    HostBlockMatrix<int> h_dst;
    ASSERT_TRUE(d.Download(h_dst));
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());
    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(BlockMatrix, AssignDevToDev)
{
    const std::vector<int> rows{2, 2, 3};
    const std::vector<int> cols{2, 4, 1};
    HostBlockMatrix<int> h_src = MakeHostBlockMat(rows, cols);

    DevBlockMatrix<int> d0;
    ASSERT_TRUE(d0.Assign(h_src));

    DevBlockMatrix<int> d1;
    ASSERT_TRUE(d1.Assign(d0));
    ASSERT_TRUE(d1.IsValid());

    HostBlockMatrix<int> h_dst;
    ASSERT_TRUE(d1.Download(h_dst));
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());
    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(BlockMatrix, RebuildStability)
{
    DevBlockMatrix<int> d;

    ASSERT_TRUE(d.BuildFromShapes(std::vector<int>{2, 3}, std::vector<int>{2, 1}));
    EXPECT_EQ(d.NumBlocks(), 2);
    EXPECT_EQ(d.TotalSize(), 2 * 2 + 3 * 1);

    ASSERT_TRUE(d.BuildFromSquares(std::vector<int>{3, 1, 4}));
    EXPECT_EQ(d.NumBlocks(), 3);
    EXPECT_EQ(d.TotalSize(), 9 + 1 + 16);

    ASSERT_TRUE(d.BuildFromShapes(std::vector<int>{5}, std::vector<int>{2}));
    EXPECT_EQ(d.NumBlocks(), 1);
    EXPECT_EQ(d.TotalSize(), 10);
    EXPECT_TRUE(d.IsValid());
}

TEST(BlockMatrix, MultiSizeSPDFromBinaryFiles)
{
    constexpr std::array<int, 6> kDims{128, 256, 512, 1024, 2048, 4096};
    constexpr std::array<const char*, 6> kPaths{
        "tests/Cuda/Test_Cholesky/data/spd_128.bin",
        "tests/Cuda/Test_Cholesky/data/spd_256.bin",
        "tests/Cuda/Test_Cholesky/data/spd_512.bin",
        "tests/Cuda/Test_Cholesky/data/spd_1024.bin",
        "tests/Cuda/Test_Cholesky/data/spd_2048.bin",
        "tests/Cuda/Test_Cholesky/data/spd_4096.bin"};

    std::vector<int> rows(kDims.begin(), kDims.end());
    std::vector<int> cols(kDims.begin(), kDims.end());

    HostBlockMatrix<double> h;
    ASSERT_TRUE(h.BuildFromShapes(rows, cols));
    ASSERT_TRUE(h.IsValid());
    ASSERT_EQ(h.NumBlocks(), static_cast<int>(kDims.size()));

    int expect_elem_offset = 0;
    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        const int n = kDims[b];
        const int elem_count = n * n;

        EXPECT_EQ(h.BlockRows(b), n);
        EXPECT_EQ(h.BlockCols(b), n);
        EXPECT_EQ(h.BlockLength(b), elem_count);
        EXPECT_EQ(h.BlockOffset(b), expect_elem_offset);

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

    DevBlockMatrix<double> d;
    ASSERT_TRUE(d.Assign(h));
    ASSERT_TRUE(d.IsValid());

    HostBlockMatrix<double> h_roundtrip;
    ASSERT_TRUE(d.Download(h_roundtrip));
    ASSERT_TRUE(h_roundtrip.IsValid());

    ASSERT_EQ(h_roundtrip.NumBlocks(), h.NumBlocks());
    ASSERT_EQ(h_roundtrip.TotalSize(), h.TotalSize());

    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        EXPECT_EQ(h_roundtrip.BlockRows(b), h.BlockRows(b));
        EXPECT_EQ(h_roundtrip.BlockCols(b), h.BlockCols(b));
        EXPECT_EQ(h_roundtrip.BlockLength(b), h.BlockLength(b));
        EXPECT_EQ(h_roundtrip.BlockOffset(b), h.BlockOffset(b));
    }
    for (int i = 0; i < h.TotalSize(); ++i)
    {
        EXPECT_EQ(h_roundtrip.Data()[i], h.Data()[i]);
    }
}

} // namespace dyno
