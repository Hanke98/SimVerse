#pragma once

#include "SimArray.h"

#include <vector>
#include <iostream>
#include <Array/Array.h>

namespace dyno
{
#ifndef SIM_GPU_FUNC
#ifdef __CUDACC__
#define SIM_GPU_FUNC __device__
#else
#define SIM_GPU_FUNC
#endif
#define SIM_BLOCKVECTOR_LOCAL_GPU_FUNC
#endif

#ifndef SIM_DYN_FUNC
#ifdef __CUDACC__
#define SIM_DYN_FUNC __host__ __device__
#else
#define SIM_DYN_FUNC
#endif
#endif

    template<typename T>
    class DevBlockVector;

    template<typename T>
    class HostBlockVector
    {
    public:
        HostBlockVector() = default;
        ~HostBlockVector() = default;

        void Clear()
        {
            data_.Clear();
            sizes_.Clear();
            offsets_.Clear();
            num_blocks_ = 0;
            total_size_ = 0;
        }

        bool BuildFromSizes(const std::vector<int>& sizes)
        {
            HostArr<int> h;
            h.Assign(sizes);
            return BuildFromSizes(h);
        }

        bool BuildFromSizes(const HostArr<int>& sizes)
        {
            Clear();
            const int n = sizes.Size();
            if (n == 0) return true;

            HostArr<int> offsets;
            offsets.Resize(n);

            int total = 0;
            for (int b = 0; b < n; ++b)
            {
                const int size = sizes[b];
                if (size < 0)
                {
                    Clear();
                    return false;
                }
                offsets[b] = total;
                total += size;
            }

            num_blocks_ = n;
            total_size_ = total;
            sizes_.Assign(sizes);
            offsets_.Assign(offsets);
            data_.Resize(total_size_);
            data_.Reset();
            return true;
        }

        void Assign(const DevBlockVector<T>& src)
        {
            num_blocks_ = src.NumBlocks();
            total_size_ = src.TotalSize();
            if (num_blocks_ == 0 || total_size_ == 0)
            {
                Clear();
                return;
            }
            sizes_.Assign(src.Sizes());
            offsets_.Assign(src.Offsets());
            data_.Assign(src.Data());
        }

        bool Assign(const HostBlockVector<T>& src)
        {
            num_blocks_ = src.num_blocks_;
            total_size_ = src.total_size_;
            sizes_.Assign(src.sizes_);
            offsets_.Assign(src.offsets_);
            data_.Assign(src.data_);
            return true;
        }

        inline int NumBlocks() const { return num_blocks_; }
        inline int TotalSize() const { return total_size_; }
        inline bool Empty() const { return num_blocks_ == 0 || total_size_ == 0; }

        inline const HostArr<int>& Sizes() const { return sizes_; }
        inline HostArr<int>& Sizes() { return sizes_; }
        inline const HostArr<int>& Offsets() const { return offsets_; }
        inline HostArr<int>& Offsets() { return offsets_; }

        inline const HostArr<T>& Data() const { return data_; }
        inline HostArr<T>& Data() { return data_; }

        inline const T* Begin() const { return data_.Begin(); }
        inline T* Begin() { return data_.Begin(); }

        inline int BlockOffset(int block_id) const { return offsets_[block_id]; }
        inline int BlockSize(int block_id) const { return sizes_[block_id]; }
        inline const T* BlockPtr(int block_id) const { return data_.Begin() + offsets_[block_id]; }
        inline T* BlockPtr(int block_id) { return data_.Begin() + offsets_[block_id]; }

        inline T& AtBlock(int block_id, int index)
        {
            const int base = offsets_.Begin()[block_id];
            return data_.Begin()[base + index];
        }

        inline const T& AtBlock(int block_id, int index) const
        {
            const int base = offsets_.Begin()[block_id];
            return data_.Begin()[base + index];
        }

        bool IsValid() const
        {
            if (num_blocks_ < 0 || total_size_ < 0) return false;
            if (sizes_.Size() != num_blocks_) return false;
            if (offsets_.Size() != num_blocks_) return false;
            if (data_.Size() != total_size_) return false;
            return true;
        }

    private:
        HostArr<T> data_;
        HostArr<int> sizes_;
        HostArr<int> offsets_;
        int num_blocks_ = 0;
        int total_size_ = 0;
    };

    template<typename T>
    class DevBlockVector
    {
    public:
        DevBlockVector() = default;
        ~DevBlockVector() = default;

        void Clear()
        {
            data_.Clear();
            sizes_.Clear();
            offsets_.Clear();
            num_blocks_ = 0;
            total_size_ = 0;
        }

        void Reset()
        {
            data_.Reset();
        }

        bool BuildFromSizes(const std::vector<int>& sizes)
        {
            HostBlockVector<T> h;
            if (!h.BuildFromSizes(sizes)) return false;
            return Upload(h.Sizes(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool BuildFromSizes(const HostArr<int>& sizes)
        {
            HostBlockVector<T> h;
            if (!h.BuildFromSizes(sizes)) return false;
            return Upload(h.Sizes(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool BuildFromSizes(const DArray<int>& sizes)
        {
            CArray<int> h_sizes;
            h_sizes.assign(sizes);

            std::vector<int> h_sizes_vec(h_sizes.begin(), h_sizes.begin() + h_sizes.size());
            HostBlockVector<T> h;
            if (!h.BuildFromSizes(h_sizes_vec)) return false;
            return Upload(h.Sizes(), h.Offsets(), h.NumBlocks(), h.TotalSize());
        }

        bool Upload(const HostArr<int>& sizes, const HostArr<int>& offsets, int num_blocks, int total_size)
        {
            if (sizes.Size() != num_blocks) return false;
            if (offsets.Size() != num_blocks) return false;

            Clear();
            num_blocks_ = num_blocks;
            total_size_ = total_size;
            sizes_.Assign(sizes);
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

        bool Assign(const HostBlockVector<T>& src)
        {
            return Upload(src.Sizes(), src.Offsets(), src.NumBlocks(), src.TotalSize()) && Assign(src.Data());
        }

        bool Assign(const DevBlockVector<T>& src)
        {
            Clear();
            num_blocks_ = src.num_blocks_;
            total_size_ = src.total_size_;
            sizes_.Assign(src.sizes_);
            offsets_.Assign(src.offsets_);
            data_.Assign(src.data_);
            return true;
        }

        bool Assign(const std::vector<T>& host_data, const std::vector<int>& sizes)
        {
            if(sizes.size() == 0)
            {
                std::cout << "Sizes vector is empty." << std::endl;
                return false;
            }
            if(host_data.size() == 0)
            {
                std::cout << "Data vector is empty." << std::endl;
                return false;
            }

            std::vector<int> offsets(sizes.size());
            int total = 0;
            for (size_t b = 0; b < sizes.size(); ++b)
            {
                offsets[b] = total;
                total += sizes[b];
            }    

            if(total != host_data.size())
            {
                std::cout << "Total size calculated from sizes " << total << " does not match host data size " << host_data.size() << std::endl;
                return false;
            }
            
            data_.Assign(host_data);
            sizes_.Assign(sizes);
            offsets_.Assign(offsets);
            num_blocks_ = static_cast<int>(sizes.size());
            total_size_ = total;

            return true;
        }

        bool Download(HostArr<T>& host_data) const
        {
            host_data.Assign(data_);
            return true;
        }

        bool Download(HostBlockVector<T>& host_data) const
        {
            host_data.Assign(*this);
            return true;
        }

        __device__ __host__ inline int NumBlocks() const { return num_blocks_; }
        inline int TotalSize() const { return total_size_; }
        inline bool Empty() const { return num_blocks_ == 0 || total_size_ == 0; }

        SIM_DYN_FUNC inline const DevArr<int>& Sizes() const { return sizes_; }
        SIM_DYN_FUNC inline DevArr<int>& Sizes() { return sizes_; }
        SIM_DYN_FUNC inline const DevArr<int>& Offsets() const { return offsets_; }
        SIM_DYN_FUNC inline DevArr<int>& Offsets() { return offsets_; }

        SIM_DYN_FUNC inline const DevArr<T>& Data() const { return data_; }
        SIM_DYN_FUNC inline DevArr<T>& Data() { return data_; }

        __device__ __host__ inline const T* Begin() const { return data_.Begin(); }
        __device__ __host__ inline T* Begin() { return data_.Begin(); }

        __device__ __host__ inline int BlockOffset(int block_id) const { return offsets_.Begin()[block_id]; }
        __device__ __host__ inline int BlockSize(int block_id) const { return sizes_.Begin()[block_id]; }

        __device__ __host__ inline const T* BlockPtr(int block_id) const
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        __device__ __host__ inline T* BlockPtr(int block_id)
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        SIM_GPU_FUNC inline T& AtBlock(int block_id, int index)
        {
            const int base = offsets_.Begin()[block_id];
            return data_.Begin()[base + index];
        }

        SIM_GPU_FUNC inline const T& AtBlock(int block_id, int index) const
        {
            const int base = offsets_.Begin()[block_id];
            return data_.Begin()[base + index];
        }

        bool IsValid() const
        {
            if (num_blocks_ < 0 || total_size_ < 0) return false;
            if (sizes_.Size() != num_blocks_) return false;
            if (offsets_.Size() != num_blocks_) return false;
            if (data_.Size() != total_size_) return false;
            return true;
        }

        SIM_GPU_FUNC inline T* operator[](int block_id)
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        SIM_GPU_FUNC inline const T* operator[](int block_id) const
        {
            return data_.Begin() + offsets_.Begin()[block_id];
        }

        SIM_GPU_FUNC inline T& operator()(int block_id, int index)
        {
            return AtBlock(block_id, index);
        }

        SIM_GPU_FUNC inline const T& operator()(int block_id, int index) const
        {
            return AtBlock(block_id, index);
        }

    private:
        DevArr<T> data_;
        DevArr<int> sizes_;
        DevArr<int> offsets_;
        int num_blocks_ = 0;
        int total_size_ = 0;
    };

    template<typename T>
    using DevArr2D = DevBlockVector<T>;

#ifdef SIM_BLOCKVECTOR_LOCAL_GPU_FUNC
#undef SIM_GPU_FUNC
#undef SIM_BLOCKVECTOR_LOCAL_GPU_FUNC
#undef SIM_DYN_FUNC
#endif
}
