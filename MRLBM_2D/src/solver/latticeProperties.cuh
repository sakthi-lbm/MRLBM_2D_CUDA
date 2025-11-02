#ifndef LATTICE_PROPERTIES_CUH
#define LATTICE_PROPERTIES_CUH

#include "../var.h"

constexpr dfloat PI = 3.14159265358979323846;

constexpr size_t NUMBER_OF_MOMENTS = 6;

constexpr int Q = 9;
constexpr dfloat W0 = 4.0 / 9.0;
constexpr dfloat W1 = 1.0 / 9.0;
constexpr dfloat W2 = 1.0 / 36.0;

__device__ constexpr dfloat w[Q] = {
    W0,
    W1, W1, W1, W1,
    W2, W2, W2, W2};

__device__ constexpr int d_cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ constexpr int d_cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

constexpr int h_cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int h_cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

constexpr dfloat as2 = 3.0;
constexpr dfloat cs2 = 1.0 / as2;

#endif // LATTICE_PROPERTIES_CUH