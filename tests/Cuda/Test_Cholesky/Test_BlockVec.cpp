#include "gtest/gtest.h"

#include "SimNode/Utils/SimBlockVector.h"

#include <vector>

namespace dyno
{
namespace
{

HostBlockVector<int> MakeHostBlockVec(const std::vector<int>& sizes)
{
    HostBlockVector<int> h;
    EXPECT_TRUE(h.BuildFromSizes(sizes));

    for (int b = 0; b < h.NumBlocks(); ++b)
    {
        int* ptr = h.BlockPtr(b);
        const int n = h.BlockSize(b);
        for (int i = 0; i < n; ++i)
        {
            ptr[i] = b * 1000 + i;
        }
    }
    return h;
}

} // namespace

TEST(BlockVec, HostBuildFromSizes)
{
    HostBlockVector<int> h;
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
}

TEST(BlockVec, RejectInvalidSizes)
{
    HostBlockVector<int> h;
    EXPECT_FALSE(h.BuildFromSizes(std::vector<int>{3, 0, 2}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());

    EXPECT_FALSE(h.BuildFromSizes(std::vector<int>{3, -1, 2}));
    EXPECT_TRUE(h.Empty());
    EXPECT_TRUE(h.IsValid());
}

TEST(BlockVec, HostDevHostRoundTrip)
{
    const std::vector<int> sizes{4, 1, 3};
    HostBlockVector<int> h_src = MakeHostBlockVec(sizes);

    DevBlockVector<int> d;
    ASSERT_TRUE(d.BuildFromSizes(sizes));
    ASSERT_TRUE(d.Assign(h_src.Data()));
    ASSERT_TRUE(d.IsValid());

    HostBlockVector<int> h_dst;
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

TEST(BlockVec, AssignHostBlockVecToDev)
{
    const std::vector<int> sizes{2, 6, 4};
    HostBlockVector<int> h_src = MakeHostBlockVec(sizes);

    DevBlockVector<int> d;
    ASSERT_TRUE(d.Assign(h_src));
    ASSERT_TRUE(d.IsValid());

    HostBlockVector<int> h_dst;
    ASSERT_TRUE(d.Download(h_dst));
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());
    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(BlockVec, AssignDevToDev)
{
    const std::vector<int> sizes{5, 3};
    HostBlockVector<int> h_src = MakeHostBlockVec(sizes);

    DevBlockVector<int> d0;
    ASSERT_TRUE(d0.Assign(h_src));

    DevBlockVector<int> d1;
    ASSERT_TRUE(d1.Assign(d0));
    ASSERT_TRUE(d1.IsValid());

    HostBlockVector<int> h_dst;
    ASSERT_TRUE(d1.Download(h_dst));
    EXPECT_EQ(h_dst.TotalSize(), h_src.TotalSize());
    for (int i = 0; i < h_src.TotalSize(); ++i)
    {
        EXPECT_EQ(h_dst.Data()[i], h_src.Data()[i]);
    }
}

TEST(BlockVec, RebuildStability)
{
    DevBlockVector<int> d;
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

} // namespace dyno

