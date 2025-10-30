#ifndef VAR_H
#define VAR_H

#include <cstddef> // recommended

// Define __host__ and __device__ safely for non-CUDA builds
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

typedef float dfloat;

template <typename T>
__host__ __device__ inline constexpr dfloat toDFloat(const T value)
{
    return static_cast<dfloat>(value);
}
template <typename T>
__host__ __device__ inline constexpr int toInt(const T value)
{
    return static_cast<int>(value);
}

template <typename T>
__host__ __device__ inline constexpr size_t toSize_t(const T value)
{
    return static_cast<size_t>(value);
}

template <typename T>
__host__ __device__ inline constexpr float toFloat(const T value)
{
    return static_cast<float>(value);
}

#define LATTICE_PROPERTIES "solver/latticeProperties.cuh"
#define CONSTANTS "cases/ldc/constants.h"
#include LATTICE_PROPERTIES
#include CONSTANTS

constexpr size_t MAX_THREADS_PER_BLOCK = 1024;
constexpr dim3 findOptimalBlockDim(size_t maxShareMemBytes, size_t bytesPerThread)
{
    size_t bestDimX = 1;
    size_t bestDimY = 1;
    size_t maxThreadsUsed = 0;
    for (size_t dimY = 2; dimY <= 32; dimY *= 2)
    {
        for (size_t dimX = 2; dimX <= 32; dimX *= 2)
        {
            size_t threads = dimX * dimY;
            size_t usedMem = threads * bytesPerThread;

            if (threads > MAX_THREADS_PER_BLOCK)
                continue;

            if (usedMem <= maxShareMemBytes)
            {
                if (threads > maxThreadsUsed)
                {
                    bestDimX = dimX;
                    bestDimY = dimY;
                    maxThreadsUsed = threads;
                }
            }
        }
    }
    return dim3(bestDimX, bestDimY);
}


#include "index.h"
#include "definitions.h"
#include "globalStructs.h"

#endif // VAR_H