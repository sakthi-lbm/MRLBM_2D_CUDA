#ifndef INDEX_H
#define INDEX_H

#include "var.h"

__host__ __device__ inline int IDX(int x, int y)
{
    return x + (y * NX);
}

#endif // INDEX_H