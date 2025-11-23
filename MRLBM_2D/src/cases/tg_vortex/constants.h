#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../../var.h"

#define TG

constexpr int N = 128;
constexpr int NX = N;
constexpr int NY = N;

constexpr dfloat LC = toDFloat(2.0) * PI;
constexpr dfloat K = 1.0; // wave number

constexpr dfloat RE = 100;
constexpr dfloat U0 = 0.1;
constexpr dfloat RHO_0 = 1.0;
constexpr dfloat delta_t = 1.0;
constexpr dfloat VISC = U0 * LC / RE;

constexpr dfloat TAU = VISC * as2 + 0.5 * delta_t;
constexpr dfloat OMEGA = 1.0 / TAU;

#define BC_X_PERIODIC 1
#define BC_Y_PERIODIC 1

#endif // CONSTANTS_H