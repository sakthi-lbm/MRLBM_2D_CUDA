#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "var.h"

inline void post_process_function(nodeVar fMom)
{

#ifdef LDC
    sample_ux_vertical_line(fMom);
    sample_uy_horizontal_line(fMom);

#endif

#ifdef TG

#endif
}

__global__ void compute_kinetic_energy(nodeVar dMom, dfloat *Ek_ana, dfloat *Ek_num)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || y >= NY)
        return;

    size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    dfloat xk = LC * toDFloat(x) / toDFloat(NX);
    dfloat yk = LC * toDFloat(y) / toDFloat(NY);
    dfloat ux_ana = U0 * sin(K * xk) * cos(K * yk);
    dfloat uy_ana = -U0 * cos(K * xk) * sin(K * yk);

    dfloat ek_ana = toDFloat(0.5) * (ux_ana * ux_ana + uy_ana * uy_ana);
    atomicAdd(Ek_ana, ek_ana);


    dfloat ux = dMom.ux[idx];
    dfloat uy = dMom.uy[idx];
    dfloat ek = toDFloat(0.5) * (ux * ux + uy * uy);

    atomicAdd(Ek_num, ek);
}

#endif