#include <iostream>
#include <fstream>

#include "initializeLBM.cuh"

void initialize_domain(nodeVar fMom, haloData gHalo)
{
    // gpu_initialize_nodeType<<<grid, block>>>(fMom);
    // checkKernelExecution();

    gpu_initialize_Moments_nodeType_GhostInterface<<<grid, block>>>(fMom, gHalo);
    checkKernelExecution();
}

// __global__ void gpu_initialize_nodeType(nodeVar fMom)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     // bounds check
//     if (x >= NX || y >= NY)
//         return;

//     size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

//     fMom.nodeType[idx] = boundary_definitions(x, y);
// }

__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar fMom, haloData gHalo)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    // bounds check
    if (x >= NX || y >= NY)
        return;

    dfloat rho = RHO_0;
    dfloat ux = toDFloat(0.0);
    dfloat uy = toDFloat(0.0);
    dfloat mxx, myy, mxy;

    size_t idx = IDX_BLOCK(threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    //========================== Initialize nodeTypes and Moments=============================================

    fMom.nodeType[idx] = boundary_definitions(x, y);
    fMom.rho[idx] = rho - RHO_0;
    fMom.ux[idx] = ux;
    fMom.uy[idx] = uy;

    dfloat pop[Q];
    for (size_t i = 0; i < Q; i++)
    {
        dfloat umag = ux * ux + uy * uy;
        dfloat udotc = ux * d_cx[i] + uy * d_cy[i];

        // Equlibrium populations
        pop[i] = w[i] * rho * (toDFloat(1.0) + as2 * udotc + toDFloat(0.5) * as2 * as2 * udotc * udotc + toDFloat(0.5) * as2 * umag);
    }
    const dfloat inv_rho = toDFloat(1.0) / rho;
    fMom.mxx[idx] = (pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8]) * inv_rho - cs2;
    fMom.myy[idx] = (pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]) * inv_rho - cs2;
    fMom.mxy[idx] = (pop[5] - pop[6] + pop[7] - pop[8]) * inv_rho;

    //========================== Halo Interface =============================================
    rho = RHO_0 + fMom.rho[idx];
    ux = fMom.ux[idx];
    uy = fMom.uy[idx];
    mxx = fMom.mxx[idx];
    myy = fMom.myy[idx];
    mxy = fMom.mxy[idx];

    pop_reconstruction(rho, ux, uy, mxx, myy, mxy, pop);

    const int tx = threadIdx.x; // local thread x id
    const int ty = threadIdx.y; // local thread y id
    const int bx = blockIdx.x;  // local block x id
    const int by = blockIdx.y;  // local block y id

    pop_save_to_halo(gHalo, tx, ty, bx, by, pop);
}
