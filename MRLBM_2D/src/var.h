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


#define STRINGIFY(x) #x
#define INCLUDE_FILE(x) STRINGIFY(x)

#define LATTICE_PROPERTIES INCLUDE_FILE(solver/latticeProperties.cuh)
#define CONSTANTS INCLUDE_FILE(cases/ldc/constants.h)
#include LATTICE_PROPERTIES
#include CONSTANTS

#endif // VAR_H