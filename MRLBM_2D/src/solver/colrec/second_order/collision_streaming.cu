#include <iostream>
#include "collision_streaming.cuh"

__device__ void mom_collision(dfloat ux, dfloat uy, dfloat &mxx, dfloat &myy, dfloat &mxy)
{
    const dfloat omegaVar = OMEGA;
    const dfloat omega_m1 = 1.0 - omegaVar;

    mxx = omega_m1 * mxx + omegaVar * ux * ux;
    myy = omega_m1 * myy + omegaVar * uy * uy;
    mxy = omega_m1 * mxy + omegaVar * ux * uy;
}

__global__ void MomCollisionStreaming(nodeVar fMom, haloData fHalo, haloData gHalo)
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

    // Loading moments from the global memory
    size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    unsigned int nodeType = fMom.nodeType[idx];
    dfloat rho = RHO_0 + fMom.rho[idx];
    dfloat ux = fMom.ux[idx];
    dfloat uy = fMom.uy[idx];
    dfloat mxx = fMom.mxx[idx];
    dfloat myy = fMom.myy[idx];
    dfloat mxy = fMom.mxy[idx];

    // printf("node: %d\n", fMom.nodeType[IDX(0, NY - 1)]);

    dfloat pop[Q];
    // construct populations from the loaded moments
    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    // copy the constructed populations to the shared memory for the streaming
    s_pop[idxPopBlock(tx, ty, 0)] = pop[1];
    s_pop[idxPopBlock(tx, ty, 1)] = pop[2];
    s_pop[idxPopBlock(tx, ty, 2)] = pop[3];
    s_pop[idxPopBlock(tx, ty, 3)] = pop[4];
    s_pop[idxPopBlock(tx, ty, 4)] = pop[5];
    s_pop[idxPopBlock(tx, ty, 5)] = pop[6];
    s_pop[idxPopBlock(tx, ty, 6)] = pop[7];
    s_pop[idxPopBlock(tx, ty, 7)] = pop[8];

    __syncthreads();

    // STREAMING
    const unsigned int xm1 = (threadIdx.x - 1 + blockDim.x) % blockDim.x;
    const unsigned int xp1 = (threadIdx.x + 1 + blockDim.x) % blockDim.x;
    const unsigned int ym1 = (threadIdx.y - 1 + blockDim.y) % blockDim.y;
    const unsigned int yp1 = (threadIdx.y + 1 + blockDim.y) % blockDim.y;

    pop[1] = s_pop[idxPopBlock(xm1, ty, 0)];
    pop[2] = s_pop[idxPopBlock(tx, ym1, 1)];
    pop[3] = s_pop[idxPopBlock(xp1, ty, 2)];
    pop[4] = s_pop[idxPopBlock(tx, yp1, 3)];
    pop[5] = s_pop[idxPopBlock(xm1, ym1, 4)];
    pop[6] = s_pop[idxPopBlock(xp1, ym1, 5)];
    pop[7] = s_pop[idxPopBlock(xp1, yp1, 6)];
    pop[8] = s_pop[idxPopBlock(xm1, yp1, 7)];

    // Loading populations from the halo layers to local thread
    pop_load_from_halo(fHalo, tx, ty, bx, by, pop);

    //========================== Moments evaluation ========================================
    if (nodeType != BULK)
    {
        boundary_condition(nodeType, pop, rho, ux, uy, mxx, myy, mxy);
    }
    else
    {
        rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
        dfloat invRho = 1.0 / rho;

        ux = (pop[1] - pop[3] + pop[5] - pop[6] - pop[7] + pop[8]) * invRho;
        uy = (pop[2] - pop[4] + pop[5] + pop[6] - pop[7] - pop[8]) * invRho;

        mxx = (pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8]) * invRho - cs2;
        myy = (pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]) * invRho - cs2;
        mxy = (pop[5] - pop[6] + pop[7] - pop[8]) * invRho;
    }

    mom_collision(ux, uy, mxx, myy, mxy);

    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    pop_save_to_halo(gHalo, tx, ty, bx, by, pop);

    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;
    fMom.mxx[idx] = mxx;
    fMom.myy[idx] = myy;
    fMom.mxy[idx] = mxy;
}
