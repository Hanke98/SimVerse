#pragma once

#include "SimArray.h"

#include <vector>
#include <iostream>

namespace dyno
{
#ifndef SIM_GPU_FUNC
#ifdef __CUDACC__
#define SIM_GPU_FUNC __device__
#else
#define SIM_GPU_FUNC
#endif
#define SIM_BLOCKMATRIX_LOCAL_GPU_FUNC
#endif

    template<typename T>
    class DevBlockMatrix;

    template<typename T>
    class HostBlockMatrix
    {
    public:
        HostBlockMatrix() = default;
        ~HostBlockMatrix() = default;

        void Clear()
        {
            data_.Clear();
            rows_.Clear();
            cols_.Clear();
            offsets_.Clear();
            num_blocks_ = 0;
            total_size_ = 0;
        }

        bool BuildFromShapes(const std::vector<int>& rows, const std::vector<int>& cols)
        {
            HostArr<int> hrows;
            HostArr<int> hcols;
            hrows.Assign(rows);
            hcols.Assign(cols);
            return BuildFromShapes(hrows, hcols);
        }

        bool BuildFromShapes(const HostArr<int>& rows, const HostArr<int>& cols)
        {
            Clear();
            const int n = rows.Size();
            if (cols.Size() != n) return false;
            if (n == 0) return true;

            HostArr<int> offsets;
            offsets.Resize(n);

            int total = 0;
            for (int b = 0; b < n; ++b)
            {
                const int r = rows[b];
                const int c = cols[b];
                if (r <= 0 || c <= 0)
                {
                    Clear();
                    return false;
                }
                const int len = r * c;
                if (len <= 0)
                {
                    Clear();
                    return false;
                }
                offsets[b] = total;
                total += len;
            }

            num_blocks_ = n;
            total_size_ = total;
            rows_.Assign(rows);
            cols_.Assign(cols);
            offsets_.Assign(offsets);
            data_.Resize(total_size_);
            data_.Reset();
            return true;
        }

        bool BuildFromSquares(const std::vector<int>& dims)
        {
            HostArr<int> d;
            d.Assign(dims);
            return BuildFromSquares(d);
        }

        bool BuildFromSquares(const HostArr<int>& dims)
        {
            HostArr<int> rows;
            HostArr<int> cols;
            rows.Assign(dims);
            cols.Assign(dims);
            return BuildFromShapes(rows, cols);
        }

        void Assign(const DevBlockMatrix<T>& src)
        {
            num_blocks_ = src.NumBlocks();
            total_size_ = src.TotalSize();
            if (num_blocks_ == 0 || total_size_ == 0)
            {
                Clear();
                return;
            }
            rows_.Assign(src.Rows());
            cols_.Assign(src.Cols());
            offsets_.Assign(src.Offsets());
            data_.Assign(src.Data());
        }

        bool Assign(const HostBlockMatrix<T>& src)
        {
            num_blocks_ = src.num_blocks_;
            total_size_ = src.total_size_;
            rows_.Assign(src.rows_);
            cols_.Assign(src.cols_);
            offsets_.Assign(src.offsets_);
            data_.Assign(src.data_);
            return true;
        }

        inline int NumBlocks() const { return num_blocks_; }
        inline int TotalSize() const { return total_size_; }
        inline bool Empty() const { return num_blocks_ == 0 || total_size_ == 0; }

        inline const HostArr<int>& Rows() const { return rows_; }
        inline const HostArr<int>& Cols() const { return cols_; }
        inline const HostArr<int>& Offsets() const { return offsets_; }
        inline HostArr<int>& Rows() { return rows_; }
        inline HostArr<int>& Cols() { return cols_; }
        inline HostArr<int>& Offsets() { return offsets_; }

        inline const HostArr<T>& Data() const { return data_; }
        inline HostArr<T>& Data() { return data_; }

        inline const T* Begin() const { return data_.Begin(); }
        inline T* Begin() { return data_.Begin(); }

        inline int BlockRows(int block_id) const { return rows_[block_id]; }
        inline int BlockCols(int block_id) const { return cols_[block_id]; }
        inline int BlockLength(int block_id) const { return rows_[block_id] * cols_[block_id]; }
        inline int BlockOffset(int block_id) const { return offsets_[block_id]; }

        inline const T* BlockPtr(int block_id) const { return data_.Begin() + offsets_[block_id]; }
        inline T* BlockPtr(int block_id) { return data_.Begin() + offsets_[block_id]; }

        bool IsValid() const
        {
            if (num_blocks_ < 0 || total_size_ < 0) return false;
            if (rows_.Size() != num_blocks_) return false;
            if (cols_.Size() != num_blocks_) return false;
            if (offsets_.Size() != num_blocks_) return false;
            if (data_.Size() != total_size_) return false;
            return true;
        }

    private:
        HostArr<T> data_;
        HostArr<int> rows_;
        HostArr<int> cols_;
        HostArr<int> offsets_;
        int num_blocks_ = 0;
        int total_size_ = 0;
    };

    template<typename T>
    class DevBlockMatrix
    {
    public:
        DevBlockMatrix() = default;
        ~DevBlockMatrix() = default;

        void Clear()
        {
            data_.Clear();
            rows_.Clear();
            cols_.Clear();
            offsets_.Clear();
            num_blocks_ = 0;
            total_size_ = 0;
        }

        bool BuildFromShapes(const std::vector<int>& rows, const std::vector<int>& cols)
        {
            HostBlockMatrix<T> h;
            if (!h.BuildFromShapes(rows, cols)) return false;
            return Upload(h.Rows(), h.Cols(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool BuildFromShapes(const HostArr<int>& rows, const HostArr<int>& cols)
        {
            HostBlockMatrix<T> h;
            if (!h.BuildFromShapes(rows, cols)) return false;
            return Upload(h.Rows(), h.Cols(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool BuildFromSquares(const std::vector<int>& dims)
        {
            HostBlockMatrix<T> h;
            if (!h.BuildFromSquares(dims)) return false;
            return Upload(h.Rows(), h.Cols(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool BuildFromSquares(const HostArr<int>& dims)
        {
            HostBlockMatrix<T> h;
            if (!h.BuildFromSquares(dims)) return false;
            return Upload(h.Rows(), h.Cols(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool Upload(
            const HostArr<int>& rows,
            const HostArr<int>& cols,
            const HostArr<int>& offsets,
            int num_blocks,
            int total_size)
        {
            if (rows.Size() != num_blocks) return false;
            if (cols.Size() != num_blocks) return false;
            if (offsets.Size() != num_blocks) return false;

            Clear();
            num_blocks_ = num_blocks;
            total_size_ = total_size;
            rows_.Assign(rows);
            cols_.Assign(cols);
            offsets_.Assign(offsets);
            data_.Resize(total_size_);
            data_.Reset();
            return true;
        }

        bool Assign(const HostArr<T>& host_data)
        {
            if (host_data.Size() != total_size_) return false;
            data_.Assign(host_data);
            return true;
        }

        bool Assign(const HostBlockMatrix<T>& src)
        {
            return Upload(src.Rows(), src.Cols(), src.Offsets(), src.NumBlocks(), src.TotalSize()) && Assign(src.Data());
        }

        bool Assign(const DevBlockMatrix<T>& src)
        {
            Clear();
            num_blocks_ = src.num_blocks_;
            total_size_ = src.total_size_;
            rows_.Assign(src.rows_);
            cols_.Assign(src.cols_);
            offsets_.Assign(src.offsets_);
            data_.Assign(src.data_);
            return true;
        }

        bool Assign(const std::vector<T>& host_data, const std::vector<int>& rows, const std::vector<int>& cols,
            const std::vector<int>& offsets)
        {
            data_.Assign(host_data);
            rows_.Assign(rows);
            cols_.Assign(cols);
            offsets_.Assign(offsets);
            num_blocks_ = static_cast<int>(rows.size());
            total_size_ = static_cast<int>(host_data.size());
            return true;
        }

        bool Assign(const std::vector<std::vector<T>>& block_data, const std::vector<std::pair<int, int>>& block_dims)
        {
            if(block_data.size() != block_dims.size())
            {
                std::cout << "Block data size " << block_data.size() << " does not match block dims size " << block_dims.size() << std::endl;
                return false;
            }

            std::vector<T> all_data;
            std::vector<int> rows(block_data.size());
            std::vector<int> cols(block_data.size());
            std::vector<int> offsets(block_data.size());
            int total_size = 0;

            for(int i = 0; i < block_data.size(); i++)
            {
                const auto& data = block_data[i];
                const auto& dim = block_dims[i];
                if(dim.first <= 0 || dim.second <= 0)
                {
                    std::cout << "Block " << i << " has invalid dims " << dim.first << "x" << dim.second << std::endl;
                    return false;
                }

                if(data.size() != dim.first * dim.second)
                {
                    std::cout << "Block " << i << " data size " << data.size() << " does not match block dims " << dim.first << "x" << dim.second << std::endl;
                    return false;
                }

                all_data.insert(all_data.end(), data.begin(), data.end());
                rows[i] = dim.first;
                cols[i] = dim.second;
                offsets[i] = total_size;
                total_size += data.size();
            }

            return Assign(all_data, rows, cols, offsets);
        }

        bool Download(HostArr<T>& host_data) const
        {
            host_data.Assign(data_);
            return true;
        }

        bool Download(HostBlockMatrix<T>& host_data) const
        {
            host_data.Assign(*this);
            return true;
        }

        inline int NumBlocks() const { return num_blocks_; }
        inline int TotalSize() const { return total_size_; }
        inline bool Empty() const { return num_blocks_ == 0 || total_size_ == 0; }

        inline const DevArr<int>& Rows() const { return rows_; }
        inline const DevArr<int>& Cols() const { return cols_; }
        inline const DevArr<int>& Offsets() const { return offsets_; }
        inline DevArr<int>& Rows() { return rows_; }
        inline DevArr<int>& Cols() { return cols_; }
        inline DevArr<int>& Offsets() { return offsets_; }

        inline const DevArr<T>& Data() const { return data_; }
        inline DevArr<T>& Data() { return data_; }

        inline const T* Begin() const { return data_.Begin(); }
        inline T* Begin() { return data_.Begin(); }

        inline int BlockRows(int block_id) const { return rows_.Begin()[block_id]; }
        inline int BlockCols(int block_id) const { return cols_.Begin()[block_id]; }
        inline int BlockLength(int block_id) const { return rows_.Begin()[block_id] * cols_.Begin()[block_id]; }
        inline int BlockOffset(int block_id) const { return offsets_.Begin()[block_id]; }

        SIM_GPU_FUNC inline const T* BlockPtr(int block_id) const
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        SIM_GPU_FUNC inline T* BlockPtr(int block_id)
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        SIM_GPU_FUNC inline T& AtBlock(int block_id, int row, int col)
        {
            const int base = offsets_.Begin()[block_id];
            const int c = cols_.Begin()[block_id];
            return data_.Begin()[base + row * c + col];
        }

        SIM_GPU_FUNC inline const T& AtBlock(int block_id, int row, int col) const
        {
            const int base = offsets_.Begin()[block_id];
            const int c = cols_.Begin()[block_id];
            return data_.Begin()[base + row * c + col];
        }

        bool IsValid() const
        {
            if (num_blocks_ < 0 || total_size_ < 0) return false;
            if (rows_.Size() != num_blocks_) return false;
            if (cols_.Size() != num_blocks_) return false;
            if (offsets_.Size() != num_blocks_) return false;
            if (data_.Size() != total_size_) return false;
            return true;
        }

        SIM_GPU_FUNC T* operator[](int block_id)
        {
            return data_.Begin() + offsets_[block_id];
        }

        SIM_GPU_FUNC T& operator()(int bid, int row, int col)
        {
            return AtBlock(bid, row, col);
        }

    private:
        DevArr<T> data_;
        DevArr<int> rows_;
        DevArr<int> cols_;
        DevArr<int> offsets_;
        int num_blocks_ = 0;
        int total_size_ = 0;
    };

    template<typename T>
    using DevMat2D = DevBlockMatrix<T>;

#ifdef SIM_BLOCKMATRIX_LOCAL_GPU_FUNC
#undef SIM_GPU_FUNC
#undef SIM_BLOCKMATRIX_LOCAL_GPU_FUNC
#endif
}

