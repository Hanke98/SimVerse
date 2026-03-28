#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace dyno 
{
    namespace simarr
    {
        inline void cudaCheck(cudaError_t err, const char* expr)
        {
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(err));
            }
        }
    }

#define SIM_CUDA_CALL(expr) ::dyno::simarr::cudaCheck((expr), #expr)

#ifdef __CUDACC__
#define SIM_DYN_FUNC __host__ __device__
#define SIM_GPU_FUNC __device__
#else
#define SIM_DYN_FUNC
#define SIM_GPU_FUNC
#endif

    template<typename T>
    class DevArr;

    template<typename T>
    class HostArr
    {
    public:
        HostArr() = default;
        explicit HostArr(int num) { Resize(num); }
        ~HostArr() = default;

        void Resize(int n) { mData.resize(n); }
        void Clear() { mData.clear(); }
        void Reset()
        {
            if (!mData.empty())
            {
                std::memset(mData.data(), 0, mData.size() * sizeof(T));
            }
        }

        inline const T* Begin() const { return mData.empty() ? nullptr : mData.data(); }
        inline T* Begin() { return mData.empty() ? nullptr : mData.data(); }

        inline const std::vector<T>* Handle() const { return &mData; }
        inline std::vector<T>* Handle() { return &mData; }

        inline int Size() const { return mData.size(); }
        inline bool IsEmpty() const { return mData.empty(); }

        inline T& operator[](int id) { return mData[id]; }
        inline const T& operator[](int id) const { return mData[id]; }

        void Assign(const HostArr<T>& src)
        {
            if (Size() != src.Size()) Resize(src.Size());
            if (src.Size() > 0) std::memcpy(Begin(), src.Begin(), src.Size() * sizeof(T));
        }

        void Assign(const std::vector<T>& src)
        {
            if (Size() != src.size()) Resize(src.size());
            if (!src.empty()) std::memcpy(Begin(), src.data(), src.size() * sizeof(T));
        }

        void Assign(const DevArr<T>& src);

    private:
        std::vector<T> mData;
    };

    template<typename T>
    class DevArr
    {
    public:
        DevArr() = default;
        explicit DevArr(int num) { Resize(num); }
        ~DevArr() { Clear(); }

        DevArr(const DevArr&) = delete;
        DevArr& operator=(const DevArr&) = delete;

        DevArr(DevArr&& other) noexcept
        {
            mData = other.mData;
            mSize = other.mSize;
            mCapacity = other.mCapacity;
            other.mData = nullptr;
            other.mSize = 0;
            other.mCapacity = 0;
        }

        DevArr& operator=(DevArr&& other) noexcept
        {
            if (this == &other) return *this;
            Clear();
            mData = other.mData;
            mSize = other.mSize;
            mCapacity = other.mCapacity;
            other.mData = nullptr;
            other.mSize = 0;
            other.mCapacity = 0;
            return *this;
        }

        void Resize(int n)
        {
            if (mSize == n) return;

            if (n == 0)
            {
                Clear();
                return;
            }

            constexpr int kPow2Threshold = (1 << 14);
            constexpr int kLargeExtra = 10000;

            // Reallocate only when growing beyond capacity or shrinking too much.
            if (n > mCapacity || n <= mCapacity / 2)
            {
                int bound = n;
                if (n > kPow2Threshold)
                {
                    bound = n + kLargeExtra;
                }
                else
                {
                    const int exp = static_cast<int>(std::ceil(std::log2(static_cast<double>(n))));
                    bound = static_cast<int>(std::pow(2.0, static_cast<double>(exp)));
                }

                Clear();
                mSize = n;
                mCapacity = bound;
                SIM_CUDA_CALL(cudaMalloc(&mData, mCapacity * sizeof(T)));
            }
            else
            {
                mSize = n;
            }
        }

        void Clear()
        {
            if (mData != nullptr)
            {
                SIM_CUDA_CALL(cudaFree(mData));
            }
            mData = nullptr;
            mSize = 0;
            mCapacity = 0;
        }

        void Reset()
        {
            if (mData != nullptr && mSize > 0)
            {
                SIM_CUDA_CALL(cudaMemset(mData, 0, mSize * sizeof(T)));
            }
        }

        SIM_DYN_FUNC inline const T* Begin() const { return mData; }
        SIM_DYN_FUNC inline T* Begin() { return mData; }

        SIM_GPU_FUNC inline T& operator[](int id) { return mData[id]; }
        SIM_GPU_FUNC inline T& operator[](int id) const { return mData[id]; }

        SIM_DYN_FUNC inline int Size() const { return mSize; }
        SIM_DYN_FUNC inline bool IsEmpty() const { return mData == nullptr; }

        void Assign(const DevArr<T>& src)
        {
            if (mSize != src.Size()) Resize(src.Size());
            if (src.Size() > 0)
            {
                SIM_CUDA_CALL(cudaMemcpy(mData, src.Begin(), src.Size() * sizeof(T), cudaMemcpyDeviceToDevice));
            }
        }

        void Assign(const HostArr<T>& src)
        {
            if (mSize != src.Size()) Resize(src.Size());
            if (src.Size() > 0)
            {
                SIM_CUDA_CALL(cudaMemcpy(mData, src.Begin(), src.Size() * sizeof(T), cudaMemcpyHostToDevice));
            }
        }

        void Assign(const std::vector<T>& src)
        {
            if (mSize != src.size()) Resize(src.size());
            if (!src.empty())
            {
                SIM_CUDA_CALL(cudaMemcpy(mData, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
            }
        }

        void Assign(const HostArr<T>& src, int count, int dstOffset = 0, int srcOffset = 0)
        {
            if (count == 0) return;
            SIM_CUDA_CALL(cudaMemcpy(
                mData + dstOffset,
                src.Begin() + srcOffset,
                count * sizeof(T),
                cudaMemcpyHostToDevice));
        }

        void Assign(const DevArr<T>& src, int count, int dstOffset = 0, int srcOffset = 0)
        {
            if (count == 0) return;
            SIM_CUDA_CALL(cudaMemcpy(
                mData + dstOffset,
                src.Begin() + srcOffset,
                count * sizeof(T),
                cudaMemcpyDeviceToDevice));
        }

        void Assign(const std::vector<T>& src, int count, int dstOffset = 0, int srcOffset = 0)
        {
            if (count == 0) return;
            SIM_CUDA_CALL(cudaMemcpy(
                mData + dstOffset,
                src.data() + srcOffset,
                count * sizeof(T),
                cudaMemcpyHostToDevice));
        }

    private:
        T* mData = nullptr;
        int mSize = 0;
        int mCapacity = 0;
    };

    template<typename T>
    inline void HostArr<T>::Assign(const DevArr<T>& src)
    {
        if (Size() != src.Size()) Resize(src.Size());
        if (src.Size() > 0)
        {
            SIM_CUDA_CALL(cudaMemcpy(Begin(), src.Begin(), src.Size() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
}

#undef SIM_CUDA_CALL
#undef SIM_DYN_FUNC
#undef SIM_GPU_FUNC
