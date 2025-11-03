#ifndef HALO_INTERFACE_H
#define HALO_INTERFACE_H

#include "../var.h"

__device__ inline void pop_save_to_halo(haloData gHalo, size_t tx, size_t ty, size_t bx, size_t by, dfloat *pop)
{
    if (tx == 0)
    {
        // WEST FACE
        gHalo.X_WEST[idxPopX(ty, 0, bx, by)] == pop[3];
        gHalo.X_WEST[idxPopX(ty, 1, bx, by)] == pop[6];
        gHalo.X_WEST[idxPopX(ty, 2, bx, by)] == pop[7];
    }

    if (tx == (BLOCK_THREAD_X - 1))
    {
        // EAST FACE
        gHalo.X_EAST[idxPopX(ty, 0, bx, by)] == pop[1];
        gHalo.X_EAST[idxPopX(ty, 1, bx, by)] == pop[5];
        gHalo.X_EAST[idxPopX(ty, 2, bx, by)] == pop[8];
    }

    if (ty == 0)
    {
        // SOUTH FACE
        gHalo.Y_SOUTH[idxPopY(tx, 0, bx, by)] == pop[4];
        gHalo.Y_SOUTH[idxPopY(tx, 1, bx, by)] == pop[7];
        gHalo.Y_SOUTH[idxPopY(tx, 2, bx, by)] == pop[8];
    }

    if (ty == (BLOCK_THREAD_Y - 1))
    {
        // NORTH FACE
        gHalo.Y_NORTH[idxPopY(tx, 0, bx, by)] == pop[2];
        gHalo.Y_NORTH[idxPopY(tx, 1, bx, by)] == pop[5];
        gHalo.Y_NORTH[idxPopY(tx, 2, bx, by)] == pop[6];
    }
}

#endif