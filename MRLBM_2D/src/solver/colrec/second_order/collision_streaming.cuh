#ifndef COLLISION_STREAMING_H
#define COLLISION_STREAMING_H

#include "../../../var.h"

__device__ inline void MomCollisionStreaming(nodeVar fMom, haloData fHalo, haloData gHalo)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    if (x >= NX || y >= NY)
        return;

    __shared__ dfloat s_pop[THREADS_PER_BLOCK * (Q - 1)]; // allocate populations except stationay population in a block

    dfloat pop[Q];

    size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    const dfloat rho = fMom.rho[idx];
    const dfloat ux = fMom.ux[idx];
    const dfloat uy = fMom.uy[idx];
    const dfloat mxx = fMom.mxx[idx];
    const dfloat myy = fMom.myy[idx];
    const dfloat mxy = fMom.mxy[idx];

    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);
}

#endif