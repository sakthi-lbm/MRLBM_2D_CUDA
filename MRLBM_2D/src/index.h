#ifndef INDEX_H
#define INDEX_H

#include "var.h"

__host__ __device__ inline int IDX(int x, int y)
{
    return x + (y * NX);
}

// __host__ __device__ __forceinline__
//     size_t
//     IDX_BLOCK(
//         const unsigned int tx,
//         const unsigned int ty,
//         const unsigned int bx,
//         const unsigned int by)
// {
//     return tx + BLOCK_THREAD_X * (ty + BLOCK_THREAD_Y * (bx + GRID_BLOCK_X * by));
// }

__host__ __device__ __forceinline__ size_t IDX_BLOCK(const unsigned int tx, const unsigned int ty, const unsigned int bx, const unsigned int by)
{
    const size_t threads_per_block = BLOCK_THREAD_X * BLOCK_THREAD_Y;
    const size_t block_index = bx + GRID_BLOCK_X * by;
    const size_t thread_index = tx + BLOCK_THREAD_X * ty;
    return thread_index + threads_per_block * block_index;
}

__host__ __device__ __forceinline__
    size_t
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int pop)
{
    const size_t popOffset = pop * THREADS_PER_BLOCK; // starting index of this population
    const size_t yOffset = ty * BLOCK_THREAD_X;       // offset for this row within population

    return tx + yOffset + popOffset; // final linear index
}

__device__ __forceinline__ size_t idxPopX(int ty, int pop, int bx, int by)
{
    const size_t block_id = bx + GRID_BLOCK_X * by; // stride to jump between blocks in the grid
    const size_t pop_id = pop + QF * block_id;       // stride to jump between populations in all blocks
    return ty + BLOCK_THREAD_Y * pop_id;            // stride to jump within a population (along local threads)
}

__device__ __forceinline__ int idxPopY(int tx, int pop, int bx, int by)
{
    const size_t block_id = bx + GRID_BLOCK_X * by; // stride to jump between blocks in the grid
    const size_t pop_id = pop + QF * block_id;      // stride to jump between populations in all blocks
    return tx + BLOCK_THREAD_X * pop_id;   // stride to jump within a population (along local threads)
}

// for 3D
// __host__ __device__ __forceinline__
// size_t IDX_BLOCK_3D(
//     const unsigned int tx, const unsigned int ty, const unsigned int tz,
//     const unsigned int bx, const unsigned int by, const unsigned int bz)
// {
//     const size_t threads_per_block =
//         BLOCK_THREAD_X * BLOCK_THREAD_Y * BLOCK_THREAD_Z;

//     const size_t blocks_per_layer =
//         GRID_BLOCK_X * GRID_BLOCK_Y;

//     const size_t block_index =
//         bx + GRID_BLOCK_X * (by + GRID_BLOCK_Y * bz);

//     const size_t thread_index =
//         tx + BLOCK_THREAD_X * (ty + BLOCK_THREAD_Y * tz);

//     return thread_index + threads_per_block * block_index;
// }

#endif // INDEX_H