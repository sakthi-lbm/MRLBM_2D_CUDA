#ifndef DEFINITIONS_H
#define DEFINITIIONS_H

#include <iostream>
#include <fstream>

#include "var.h"

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);
constexpr size_t BYTES_PER_KB = (1 << 10);

constexpr size_t MAX_SHARED_MEM_BYTES = 48 * BYTES_PER_KB; // 48 kb shared memory
constexpr size_t SHARED_MEM_PER_THREAD = (Q - 1) * sizeof(dfloat);

constexpr dim3 OPTIMAL_BLOCK = findOptimalBlockDim(MAX_SHARED_MEM_BYTES, SHARED_MEM_PER_THREAD);

constexpr size_t BLOCK_THREAD_X = OPTIMAL_BLOCK.x;
constexpr size_t BLOCK_THREAD_Y = OPTIMAL_BLOCK.y;

constexpr size_t GRID_BLOCK_X = (NX + BLOCK_THREAD_X - 1) / BLOCK_THREAD_X;
constexpr size_t GRID_BLOCK_Y = (NY + BLOCK_THREAD_Y - 1) / BLOCK_THREAD_Y;

constexpr size_t THREADS_PER_BLOCK = BLOCK_THREAD_X * BLOCK_THREAD_Y;
constexpr size_t NUMBER_OF_BLOCKS = GRID_BLOCK_X * GRID_BLOCK_Y;
constexpr size_t TOTAL_NUMBER_OF_THREADS = NUMBER_OF_BLOCKS * THREADS_PER_BLOCK;

constexpr dim3 block(BLOCK_THREAD_X, BLOCK_THREAD_Y);
constexpr dim3 grid(GRID_BLOCK_X, GRID_BLOCK_Y);

constexpr size_t NUM_LBM_NODES = TOTAL_NUMBER_OF_THREADS;
constexpr size_t MEM_SIZE_LBM_NODES = NUM_LBM_NODES * sizeof(dfloat);

#endif // DEFINITIONS_H