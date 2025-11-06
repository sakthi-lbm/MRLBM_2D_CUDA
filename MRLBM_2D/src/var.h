#ifndef VAR_H
#define VAR_H

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

typedef double dfloat;
typedef std::chrono::high_resolution_clock::time_point timestep;

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

// clang-format off
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// select a Case:
//1. ldc

#define BC_PROBLEM ldc
#define REG_ORDER second_order

#define CASE_DIRECTORY cases
#define COLREC_DIRECTORY colrec
#define SOLVER_DIRECTORY solver

#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants.h)
#define CASE_OUTPUTS STR(CASE_DIRECTORY/BC_PROBLEM/outputs.h)
#define RECONSTRUCT STR(SOLVER_DIRECTORY/COLREC_DIRECTORY/REG_ORDER/reconstruction.cuh)
#define CASE_BOUNDARY STR(CASE_DIRECTORY/BC_PROBLEM/boundaries.cuh)


#define LATTICE_PROPERTIES "solver/latticeProperties.cuh"
#include LATTICE_PROPERTIES
#include CASE_CONSTANTS
#include CASE_OUTPUTS
#include RECONSTRUCT

// clang-format on

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

#include "definitions.h"
#include "index.h"
#include "globalStructs.h"
#include "nodeTypeMap.h"
#include "utils/cudaHelpers.cuh"
#include "utils/file_utils.h"

#include CASE_BOUNDARY

#endif // VAR_H