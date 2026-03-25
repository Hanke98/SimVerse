#include "utils.h"

namespace dyno
{
    template<typename T>
    __global__ void BatchDenseMatrixVectorMul(DArray2D<T> mat, DArray2D<T> vec, DArray2D<T> out, 
        DArray<int> rows, DArray<int> cols, int num_sys, bool is_incremental, DArray<int> skip_flag)
    {
        int sys_id = blockIdx.x;
        if (sys_id >= num_sys)
            return;
        if(skip_flag.size() > 0)
        {
            if(skip_flag[sys_id])
                return;
        }

        int row = rows[sys_id];
        int col = cols[sys_id];

        int ridx = threadIdx.x;

        if(ridx >= row)
            return;

        T sum = 0;
        for(int cidx = 0; cidx < col; cidx++)
            sum += mat(sys_id, ridx * col + cidx) * vec(sys_id, cidx);

        is_incremental ? out(sys_id, ridx) += sum : out(sys_id, ridx) = sum;
    }

    template __global__ void BatchDenseMatrixVectorMul(DArray2D<Real> mat, DArray2D<Real> vec, DArray2D<Real> out, DArray<int> rows, DArray<int> cols, int num_sys, bool is_incremental, DArray<int> skip_flag);
    template __global__ void BatchDenseMatrixVectorMul(DArray2D<int> mat, DArray2D<int> vec, DArray2D<int> out, DArray<int> rows, DArray<int> cols, int num_sys, bool is_incremental, DArray<int> skip_flag);

    __global__ void BatchCholeskySolveVarSizeKernel(
        DArray2D<Real> A_packed,
        const DArray2D<Real> b_packed,
        DArray2D<Real> x_packed,
        const DArray<int> n_list,
        int num_envs,
        DArray<int> skip_flag)
    {
        const int env_id = blockIdx.x;
        if(env_id >= num_envs)
            return;
        if(skip_flag.size() > 0)
            if(skip_flag[env_id])
                return;

        if(threadIdx.x != 0)
            return;

        const int n = n_list[env_id];

        // Matrices in this solver path are stored compactly per-env using
        // stride n (not padded by leading_dim).
        const int lda = n;

        const Real eps = 1e-12f;

        // Cholesky in-place: A = L * L^T, keep L in lower triangle of A_packed.
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                const int ij = i * lda + j;
                Real sum = A_packed(env_id, ij);
                for(int k = 0; k < j; k++)
                {
                    const int ik = i * lda + k;
                    const int jk = j * lda + k;
                    sum -= A_packed(env_id, ik) * A_packed(env_id, jk);
                }

                if(i == j)
                {
                    if(sum < eps)
                        sum = eps;
                    A_packed(env_id, ij) = sqrtf(sum);
                }
                else
                {
                    const int jj = j * lda + j;
                    const Real d = A_packed(env_id, jj);
                    A_packed(env_id, ij) = (fabsf(d) > eps) ? (sum / d) : 0.f;
                }
            }

            for(int j = i + 1; j < n; j++)
                A_packed(env_id, i * lda + j) = 0.f;
        }

        // Forward solve: L y = b. Reuse x_packed as y workspace.
        for(int i = 0; i < n; i++)
        {
            Real sum = b_packed(env_id, i);
            for(int k = 0; k < i; k++)
                sum -= A_packed(env_id, i * lda + k) * x_packed(env_id, k);

            const Real lii = A_packed(env_id, i * lda + i);
            x_packed(env_id, i) = sum / lii;
        }

        // Backward solve: L^T x = y, in-place on x_packed.
        for(int i = n - 1; i >= 0; i--)
        {
            Real sum = x_packed(env_id, i);
            for(int k = i + 1; k < n; k++)
                sum -= A_packed(env_id, k * lda + i) * x_packed(env_id, k);

            const Real lii = A_packed(env_id, i * lda + i);
            x_packed(env_id, i) = sum / lii;
        }
    }
}