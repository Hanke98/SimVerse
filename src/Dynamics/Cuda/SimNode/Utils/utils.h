#pragma once

#include "Array/Array.h"
#include "Array/Array2D.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "Quat.h"

namespace dyno
{
    #define INIT_DYNO_ARRAY(arr, length)    \
        arr.resize(length); \
        arr.reset();
    
    #define INIT_DYNO_ARRAY2D(arr, num_1d, max_length)    \
        arr.resize(num_1d, max_length); \
        arr.reset();

    template<typename T>
    void FlattenArray2D(const DArray2D<T>& src,  DArray<T>& dst, int total_count,
        const DArray<int>& array_lengths, const DArray<int>& array_offsets);
    
    template<typename T>
    void FlattenArray2D(const CArray2D<T>& src,  CArray<T>& dst, int total_count,
        const CArray<int>& array_lengths, const CArray<int>& array_offsets);

#ifdef __CUDACC__
    template<typename T>
    __global__ void FlattenArray2DKernel(
        DArray2D<T> src,
        DArray<T> dst,
        DArray<int> array_lengths,
        DArray<int> array_offsets,
        int total_count)
    {
        int env_id = blockIdx.x;
        if (env_id >= array_lengths.size())
            return;

        int len = array_lengths[env_id];
        int offset = array_offsets[env_id];
        for (int local_id = threadIdx.x; local_id < len; local_id += blockDim.x)
        {
            int out_id = offset + local_id;
            if (out_id >= 0 && out_id < total_count)
                dst[out_id] = src(env_id, local_id);
        }
    }

    template<typename T>
    inline void FlattenArray2D(const DArray2D<T>& src, DArray<T>& dst, int total_count,
        const DArray<int>& array_lengths, const DArray<int>& array_offsets)
    {
        dst.resize(total_count);
        if (total_count <= 0 || array_lengths.size() == 0)
            return;

        FlattenArray2DKernel<<<array_lengths.size(), 128>>>(
            src,
            dst,
            array_lengths,
            array_offsets,
            total_count);
    }
#endif

    template<typename T>
    inline void FlattenArray2D(const CArray2D<T>& src, CArray<T>& dst, int total_count,
        const CArray<int>& array_lengths, const CArray<int>& array_offsets)
    {
        dst.resize(total_count);
        for (uint env_id = 0; env_id < array_lengths.size(); ++env_id)
        {
            int len = array_lengths[env_id];
            int offset = array_offsets[env_id];
            for (int local_id = 0; local_id < len; ++local_id)
            {
                int out_id = offset + local_id;
                if (out_id >= 0 && out_id < total_count)
                    dst[out_id] = src(env_id, local_id);
            }
        }
    }

    template<typename T>
    inline T GetMaxValue(const DArray<T>& arr, int count)
    {
        if(count <= 0)
            return T(0);
        thrust::device_ptr<const T> d_ptr(arr.begin());

        return *thrust::max_element(d_ptr, d_ptr + count);
    }

    template<typename T>
    __global__ void BatchDenseMatrixVectorMul(DArray2D<T> mat, DArray2D<T> vec, DArray2D<T> out, 
        DArray<int> rows, DArray<int> cols, int num_sys, bool is_incremental=false, DArray<int> skip_flag = DArray<int>());

    template<typename T>
    __device__ void DenseAnyMatrixMatrixMul(const DArray2D<T>& matA, const DArray2D<T>& matB, DArray2D<T>& mat_out, 
        int sys_id, Vec2i size_A, Vec2i size_B, Vec2i offset_A, Vec2i offset_B, Vec2i offset_out)
    {
        // A(sub) [size_A.x x size_A.y] * B(sub) [size_B.x x size_B.y]
        // -> Out(sub) [size_A.x x size_B.y]
        if (size_A.y != size_B.x)
            return;

        for (int r = 0; r < size_A.x; r++)
        {
            for (int c = 0; c < size_B.y; c++)
            {
                T sum = 0;

                for (int k = 0; k < size_A.y; k++)
                {
                    const T valA = MatrixAt(matA, sys_id, Vec2i(r, k), offset_A);
                    const T valB = MatrixAt(matB, sys_id, Vec2i(k, c), offset_B);
                    sum += valA * valB;
                }

                MatrixAt(mat_out, sys_id, Vec2i(r, c), offset_out) = sum;
            }
        }
    }

    template<typename T>
    __device__ void DenseMat3x3MatirxAnyMul(const Mat3f& matA, const DArray2D<T>& matB, DArray2D<T>& mat_out,
        int sys_id, Vec2i size_B, Vec2i offset_B, Vec2i offset_out)
    {
        // A [3 x 3] * B(sub) [3 x size_B.y]
        // -> Out(sub) [3 x size_B.y]
        if (size_B.x != 3)
            return;

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < size_B.y; c++)
            {
                T sum = 0;

                for (int k = 0; k < 3; k++)
                {
                    const T valA = matA(r, k);
                    const T valB = MatrixAt(matB, sys_id, Vec2i(k, c), offset_B);
                    sum += valA * valB;
                }

                MatrixAt(mat_out, sys_id, Vec2i(r, c), offset_out) = sum;
            }
        }
    }

    template<typename T>
    inline __host__ __device__ Quat<T> QuatFromAxisAngle(const Vector<T, 3>& axis, T angle)
    {
        T half_angle = angle * 0.5f;
        T s = sin(half_angle);
        return Quat<T>(axis.x * s, axis.y * s, axis.z * s, cos(half_angle));
    }

    inline __host__ __device__ Vec3f RotateVector(const Vec3f& v, const Quat<Real>& q)
    {
        // Rotate vector v by quaternion q
        Vec3f tmp = Vec3f(q.w * v.x + q.y * v.z - q.z * v.y,
                          q.w * v.y + q.z * v.x - q.x * v.z,
                          q.w * v.z + q.x * v.y - q.y * v.x);
        return v + 2.f * Vec3f(q.y * tmp.z - q.z * tmp.y,
                               q.z * tmp.x - q.x * tmp.z,
                               q.x * tmp.y - q.y * tmp.x);
    }

    inline __host__ __device__ void Quat2Vel(const Quat<Real>& quat, Real& speed, Vec3f& angle_vel, Real dt)
    {
        Vector<Real, 3> axis(quat.x, quat.y, quat.z);
        Real sin_half_theta = axis.norm();
        axis.normalize();

        speed = 2.f * atan2(sin_half_theta, quat.w);
        if(speed > M_PI)
            speed -= 2.f * M_PI;

        speed /= dt;
        angle_vel = axis * speed;
    }

    template<typename T>    // 这个函数用来查batch matrix的元素, 相当于vector[sys_id][vec<mat1D>], sys_id是batch_id, mat_id表示第几个小矩阵，row col是小矩阵内的行列，submat_size是小矩阵的尺寸
    inline __device__ T& MatrixAt(DArray2D<T>& mat, int sys_id, int mat_id, int row, int col, Vec2i submat_size)
    {
        int submat_start = submat_size.x * submat_size.y * mat_id;
        // submat是按行展开存
        int idx = submat_start + row * submat_size.y + col;
        return mat(sys_id, idx);
    }

    template<typename T>    // 这个函数用来查batch matrix的元素, 相当于vector[sys_id][mat1D]
    inline __device__ T& MatrixAt(DArray2D<T>& mat, int sys_id, int row, int col, Vec2i mat_size)
    {
        int idx = row * mat_size.y + col;
        return mat(sys_id, idx);
    }

    template<typename T>
    inline __device__ const T& MatrixAt(const DArray2D<T>& mat, int sys_id, int row, int col, Vec2i mat_size)
    {
        int idx = row * mat_size.y + col;
        return mat(sys_id, idx);
    }

    template<typename T>
    inline __device__ T& MatrixAt(DArray2D<T>& mat, int sys_id, Vec2i submat_idx, Vec2i submat_offset)
    {
        // submat_idx is local (row, col) inside the submatrix.
        int local_r = submat_idx.x;
        int local_c = submat_idx.y;

        int global_r = submat_offset.x + local_r;
        int global_c = submat_offset.y + local_c;
        int idx = global_r * static_cast<int>(mat.ny()) + global_c;
        return mat(sys_id, idx);
    }

    template<typename T>
    inline __device__ const T& MatrixAt(const DArray2D<T>& mat, int sys_id, Vec2i submat_idx, Vec2i submat_offset)
    {
        int local_r = submat_idx.x;
        int local_c = submat_idx.y;

        int global_r = submat_offset.x + local_r;
        int global_c = submat_offset.y + local_c;
        int idx = global_r * static_cast<int>(mat.ny()) + global_c;
        return mat(sys_id, idx);
    }

    template<typename T>
    __global__ void PrintVector(DArray2D<T> vec, int sys_id, int length)
    {
        if(threadIdx.x != 0)
            return;

        printf("PrintVector[%d]: ", sys_id);
        for(int i = 0; i < length; i++)
            printf("%f\t", vec(sys_id, i));
        printf("\n\n");
    }

    template<typename T>
    __global__ void PrintVector(DArray<T> vec, int length)
    {
        if(threadIdx.x != 0)
            return;

        printf("PrintVector1D: ");
        for(int i = 0; i < length; i++)
            printf("%f\t", vec[i]);
        printf("\n");
    }

    template<typename T>
    __global__ void SumArray2D(DArray2D<T> arr_src1, DArray2D<T> arr_src2, DArray2D<T> arr_dst, int sys_num, DArray<int> lengths, bool is_sum=true)
    {
        int sys_id = blockIdx.x;
        if(sys_id >= sys_num)
            return;

        int len = lengths[sys_id];
        for(int i = threadIdx.x; i < len; i += blockDim.x)
        {
            T a = arr_src1(sys_id, i);
            T b = arr_src2(sys_id, i);
            arr_dst(sys_id, i) = is_sum ? (a + b) : (a - b);
        }
    }

    template<typename T>
    __global__ void SumArray2D(DArray2D<T> arr_src1, DArray2D<T> arr_src2, DArray2D<T> arr_dst, int sys_num, DArray<int> lengths, DArray<int> skip_flag, bool is_sum=true)
    {
        int sys_id = blockIdx.x;
        if(sys_id >= sys_num)
            return;
        if(skip_flag[sys_id])
            return;

        int len = lengths[sys_id];
        for(int i = threadIdx.x; i < len; i += blockDim.x)
        {
            T a = arr_src1(sys_id, i);
            T b = arr_src2(sys_id, i);
            arr_dst(sys_id, i) = is_sum ? (a + b) : (a - b);
        }
    }

    // Variable-size batched Cholesky solve for compact row-major storage.
    // DArray2D index meaning here:
    // - first index: system/environment id
    // - second index: flattened row-major data of that system
    // A(sys, i * n + j) corresponds to A_ij of that system, where n = n_list[sys].
    // b(sys, i) and x(sys, i) are packed vectors per system.
    // NOTE: This kernel factorizes A in-place (A becomes its lower-triangular Cholesky factor L).
    // leading_dim is kept as an upper-bound guard for compatibility.
    __global__ void BatchCholeskySolveVarSizeKernel(
        DArray2D<Real> A_packed,         // [sys, leading_dim * leading_dim], row-major, overwritten by L
        const DArray2D<Real> b_packed,   // [sys, leading_dim]
        DArray2D<Real> x_packed,         // [sys, leading_dim]
        const DArray<int> n_list,        // [env], actual n for each environment
        int num_envs, DArray<int> skip_flag=DArray<int>());
    
}