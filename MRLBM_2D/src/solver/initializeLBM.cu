#include <iostream>
#include <fstream>

#include "initializeLBM.cuh"

__global__ void gpu_initialize_moments(nodeVar fMom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockIdx.y + threadIdx.y;

    dfloat rho = RHO_0;
    dfloat ux = toDFloat(0.0);
    dfloat uy = toDFloat(0.0);

    fMom.rho[IDX(x, y)] = rho - RHO_0;
    fMom.ux[IDX(x, y)] = ux;
    fMom.uy[IDX(x, y)] = uy;

    dfloat feq[Q];
    for (size_t i = 0; i < Q; i++)
    {
        dfloat Hxx = d_cx[i] * d_cx[i] - cs2;
        dfloat Hyy = d_cy[i] * d_cy[i] - cs2;
        dfloat Hxy = d_cx[i] * d_cy[i];

        feq[i] = w[i] * rho * (toDFloat(1.0) + as2 * (ux * d_cx[i] + uy * d_cy[i]) + toDFloat(0.5) * as2 * as2 * (Hxx * ux * ux + Hyy * uy * uy + toDFloat(2.0) * Hxy * ux * uy));
    }
}
