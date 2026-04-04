#include <cstdio>

#include <cuda_runtime.h>

struct TriangleLike
{
    int x;
    int y;
    int z;
};

int main()
{
    constexpr size_t width = sizeof(TriangleLike) * 1;
    constexpr size_t height = 13492400;

    TriangleLike* ptr = nullptr;
    size_t pitch = 0;

    const cudaError_t err = cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, width, height);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocPitch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("sizeof(TriangleLike) = %zu bytes\n", sizeof(TriangleLike));
    std::printf("width  = %zu bytes per logical row\n", width);
    std::printf("height = %zu rows\n", height);
    std::printf("pitch  = %zu bytes per allocated row\n", pitch);
    const size_t overflow_threshold = static_cast<size_t>((1ULL << 32) / pitch);
    std::printf("overflow threshold if pitch=%zu: %zu rows\n", pitch, overflow_threshold);

    cudaFree(ptr);
    return 0;
}


/*
    nvcc -std=c++17 test/check_pitch.cu -o test/check_pitch   
    ./test/check_pitch
*/