#ifndef HALO_INTERFACE_H
#define HALO_INTERFACE_H

#include "../var.h"

__device__ inline void pop_load_from_halo(haloData fHalo, unsigned int tx, unsigned int ty, unsigned int bx, unsigned int by, dfloat *pop)
{
    const unsigned int txm1 = (tx - 1 + blockDim.x) % blockDim.x;
    const unsigned int txp1 = (tx + 1 + blockDim.x) % blockDim.x;
    const unsigned int tym1 = (ty - 1 + blockDim.y) % blockDim.y;
    const unsigned int typ1 = (ty + 1 + blockDim.y) % blockDim.y;

    const unsigned int bxm1 = (bx - 1 + gridDim.x) % gridDim.x;
    const unsigned int bxp1 = (bx + 1 + gridDim.x) % gridDim.x;
    const unsigned int bym1 = (by - 1 + gridDim.y) % gridDim.y;
    const unsigned int byp1 = (by + 1 + gridDim.y) % gridDim.y;

    if (tx == 0)
    {
        // WEST face in the block
        pop[1] = fHalo.X_EAST[idxPopX(ty, 0, bxm1, by)];
        pop[5] = fHalo.X_EAST[idxPopX(tym1, 1, bxm1, ((ty == 0) ? bym1 : by))];
        pop[8] = fHalo.X_EAST[idxPopX(typ1, 2, bxm1, ((ty == (blockDim.y - 1) ? byp1 : by)))];
    }

    if (tx == (blockDim.x - 1))
    {
        // EAST face in the block
        pop[3] = fHalo.X_WEST[idxPopX(ty, 0, bxp1, by)];
        pop[6] = fHalo.X_WEST[idxPopX(tym1, 1, bxp1, ((ty == 0) ? bym1 : by))];
        pop[7] = fHalo.X_WEST[idxPopX(typ1, 2, bxp1, ((ty == (blockDim.y - 1)) ? byp1 : by))];
        // printf("BX=%u BY=%u | LOAD EAST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[3], pop[6], pop[7]);
    }

    if (ty == 0)
    {
        // SOUTH face in the block
        pop[2] = fHalo.Y_NORTH[idxPopY(tx, 0, bx, bym1)];
        pop[5] = fHalo.Y_NORTH[idxPopY(txm1, 1, ((tx == 0) ? bxm1 : bx), bym1)];
        pop[6] = fHalo.Y_NORTH[idxPopY(txp1, 2, ((tx == (blockDim.x - 1)) ? bxp1 : bx), bym1)];
    }

    if (ty == (blockDim.y - 1))
    {
        // NORTH face in the block
        pop[4] = fHalo.Y_SOUTH[idxPopY(tx, 0, bx, byp1)];
        pop[7] = fHalo.Y_SOUTH[idxPopY(txp1, 1, ((tx == (blockDim.x - 1)) ? bxp1 : bx), byp1)];
        pop[8] = fHalo.Y_SOUTH[idxPopY(txm1, 2, ((tx == 0) ? bxm1 : bx), byp1)];
    }
}

__device__ inline void pop_save_to_halo(haloData gHalo, unsigned int tx, unsigned int ty, unsigned int bx, unsigned int by, dfloat *pop)
{
    if (tx == 0)
    {
        // WEST FACE
        gHalo.X_WEST[idxPopX(ty, 0, bx, by)] = pop[3];
        gHalo.X_WEST[idxPopX(ty, 1, bx, by)] = pop[6];
        gHalo.X_WEST[idxPopX(ty, 2, bx, by)] = pop[7];
        // printf("BX=%u BY=%u | WEST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[3], pop[6], pop[7]);
    }

    if (tx == (blockDim.x - 1))
    {
        // EAST FACE
        gHalo.X_EAST[idxPopX(ty, 0, bx, by)] = pop[1];
        gHalo.X_EAST[idxPopX(ty, 1, bx, by)] = pop[5];
        gHalo.X_EAST[idxPopX(ty, 2, bx, by)] = pop[8];
        // printf("BX=%u BY=%u | EAST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[1], pop[5], pop[8]);
    }

    if (ty == 0)
    {
        // SOUTH FACE
        gHalo.Y_SOUTH[idxPopY(tx, 0, bx, by)] = pop[4];
        gHalo.Y_SOUTH[idxPopY(tx, 1, bx, by)] = pop[7];
        gHalo.Y_SOUTH[idxPopY(tx, 2, bx, by)] = pop[8];
        // printf("BX=%u BY=%u | SOUTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[4], pop[7], pop[8]);
    }

    if (ty == (blockDim.y - 1))
    {
        // NORTH FACE
        gHalo.Y_NORTH[idxPopY(tx, 0, bx, by)] = pop[2];
        gHalo.Y_NORTH[idxPopY(tx, 1, bx, by)] = pop[5];
        gHalo.Y_NORTH[idxPopY(tx, 2, bx, by)] = pop[6];
        // printf("BX=%u BY=%u | NORTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[2], pop[5], pop[6]);
    }
}

#endif