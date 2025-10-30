#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../../var.h"

constexpr int N = 64;
constexpr int NX = N;
constexpr int NY = N;

constexpr dfloat RE = 100.0;
constexpr dfloat U0 = 0.1;
constexpr dfloat delta_t = 1.0;
constexpr dfloat VISC = U0 * toDFloat(NX - 1) / RE;

constexpr dfloat TAU = VISC * as2 + toDFloat(0.5) * delta_t;
constexpr dfloat OMEGA = toDFloat(1.0) / TAU;

#endif // CONSTANTS_H