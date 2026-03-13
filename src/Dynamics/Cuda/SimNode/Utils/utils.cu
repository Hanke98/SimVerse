#include "utils.h"

namespace dyno
{
    template<typename T>
    __global__ void BatchDenseMatrixVectorMul(DArray2D<T> mat, DArray2D<T> vec, DArray2D<T> out, DArray<int> rows, DArray<int> cols, int num_sys)
    {
        int sys_id = blockIdx.x;
        if (sys_id >= num_sys)
            return;

        int row = rows[sys_id];
        int col = cols[sys_id];

        int ridx = threadIdx.x;

        if(ridx >= row)
            return;

        T sum = 0;
        for(int cidx = 0; cidx < col; cidx++)
            sum += mat(sys_id, ridx * col + cidx) * vec(sys_id, cidx);

        out(sys_id, ridx) = sum;
    }

    template __global__ void BatchDenseMatrixVectorMul(DArray2D<Real> mat, DArray2D<Real> vec, DArray2D<Real> out, DArray<int> rows, DArray<int> cols, int num_sys);
    template __global__ void BatchDenseMatrixVectorMul(DArray2D<int> mat, DArray2D<int> vec, DArray2D<int> out, DArray<int> rows, DArray<int> cols, int num_sys);
}