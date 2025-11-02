#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../var.h"

__host__ __device__ inline unsigned int boundary_definitions(const int x, const int y)
{
    if (x == 0 && y == 0)
    {
        return SOUTH_WEST;
    }
    else if (x == 0 && y == (NY - 1))
    {
        return NORTH_WEST;
    }
    else if (x == (NX - 1) && y == 0)
    {
        return SOUTH_EAST;
    }
    else if (x == (NX - 1) && y == (NY - 1))
    {
        return NORTH_EAST;
    }
    else if (x == 0)
    {
        return WEST;
    }
    else if (x == (NX - 1))
    {
        return EAST;
    }
    else if (y == 0)
    {
        return SOUTH;
    }
    else if (y == (NY - 1))
    {
        return NORTH;
    }
    else
    {
        return BULK;
    }
}


// // Mark as host & device
// __host__ __device__ inline unsigned int boundary_definitions(int x, int y)
// {
//     unsigned int mask = 0;
//     mask |= (x == 0)      << 0; // bit 0
//     mask |= (x == NX-1)   << 1; // bit 1
//     mask |= (y == 0)      << 2; // bit 2
//     mask |= (y == NY-1)   << 3; // bit 3

//     switch (mask) {
//         case 0b0000: return BULK;           // interior
//         case 0b0001: return WEST;           // x==0
//         case 0b0010: return EAST;           // x==NX-1
//         case 0b0100: return SOUTH;          // y==0
//         case 0b1000: return NORTH;          // y==NY-1
//         case 0b0101: return SOUTH_WEST;     // x==0 && y==0
//         case 0b1001: return NORTH_WEST;     // x==0 && y==NY-1
//         case 0b0110: return SOUTH_EAST;     // x==NX-1 && y==0
//         case 0b1010: return NORTH_EAST;     // x==NX-1 && y==NY-1
//         default:   return MISSING_DEFINITION;
//     }
// }

#endif // BOUDARIES_H