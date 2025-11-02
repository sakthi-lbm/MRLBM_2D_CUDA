#ifndef INDEX_H
#define INDEX_H

#include "var.h"

// __host__ __device__ inline int IDX(int x, int y)
// {
//     return x + (y * NX);
// }

__host__ __device__ size_t __forceinline__ IDX_BLOCK(
    const int tx,
    const int ty,
    const int bx,
    const int by)
{
    return tx + BLOCK_THREAD_X * (ty + BLOCK_THREAD_Y * (bx + GRID_BLOCK_X * by));
}

#endif // INDEX_H