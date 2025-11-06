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

__device__ inline void boundary_condition(unsigned int nodeType, dfloat *pop, dfloat &rhoVar, dfloat &ux, dfloat &uy, dfloat &mxx, dfloat &myy, dfloat &mxy)
{
    
    switch (nodeType)
    {
    case NORTH:
    {
        const dfloat rhoI = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[1] + pop[3] + pop[5] + pop[6]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[2] + pop[5] + pop[6]) * inv_rhoI - cs2;
        const dfloat mxyI = (pop[5] - pop[6]) * inv_rhoI;

        ux = U0;
        uy = 0.0;

        rhoVar = 3.0 * rhoI * (4.0 - 3.0 * (OMEGA - 1.0) * myyI) / (9.0 + OMEGA);

        mxx = 6.0 * rhoI * mxxI / (5.0 * (rhoVar));
        myy = (9.0 * rhoI * myyI + (rhoVar)) / (6.0 * (rhoVar));
        mxy = (6.0 * rhoI * mxyI - U0 * (rhoVar)) / (3.0 * (rhoVar));

        break;
    }
    case SOUTH:
    {
        const dfloat rhoI = pop[0] + pop[1] + pop[3] + pop[4] + pop[7] + pop[8];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[1] + pop[3] + pop[7] + pop[8]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[4] + pop[7] + pop[8]) * inv_rhoI - cs2;
        const dfloat mxyI = (pop[7] - pop[8]) * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        rhoVar = 3.0 * rhoI * (4.0 - 3.0 * (OMEGA - 1.0) * myyI) / (9.0 + OMEGA);

        mxx = 6.0 * rhoI * mxxI / (5.0 * (rhoVar));
        myy = (rhoVar + 9.0 * rhoI * myyI) / (6.0 * (rhoVar));
        mxy = 2.0 * rhoI * mxyI / (rhoVar);

        break;
    }
    case WEST:
    {
        const dfloat rhoI = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[3] + pop[6] + pop[7]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[2] + pop[4] + pop[6] + pop[7]) * inv_rhoI - cs2;
        const dfloat mxyI = (pop[7] - pop[6]) * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        rhoVar = 3.0 * rhoI * (4.0 - 3.0 * (OMEGA - 1.0) * mxxI) / (9.0 + OMEGA);

        mxx = (9.0 * rhoI * mxxI + (rhoVar)) / (6.0 * (rhoVar));
        myy = (6.0 * rhoI * myyI) / (5.0 * (rhoVar));
        mxy = 2.0 * rhoI * mxyI / (rhoVar);

        break;
    }
    case EAST:
    {
        const dfloat rhoI = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[1] + pop[5] + pop[8]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[2] + pop[4] + pop[5] + pop[8]) * inv_rhoI - cs2;
        const dfloat mxyI = (pop[5] - pop[8]) * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        rhoVar = 3.0 * rhoI * (4.0 - 3.0 * (OMEGA - 1.0) * mxxI) / (9.0 + OMEGA);

        mxx = (9.0 * rhoI * mxxI + (rhoVar)) / (6.0 * (rhoVar));
        myy = (6.0 * rhoI * myyI) / (5.0 * (rhoVar));
        mxy = 2.0 * rhoI * mxyI / (rhoVar);

        break;
    }
    case SOUTH_WEST:
    {
        
        const dfloat rhoI = pop[0] + pop[3] + pop[4] + pop[7];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[3] + pop[7]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[4] + pop[7]) * inv_rhoI - cs2;
        const dfloat mxyI = pop[7] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        rhoVar = -12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * (OMEGA - 1.0) * mxxI - 7.0 * (OMEGA - 1.0) * mxyI + 3.0 * OMEGA * myyI) / (16.0 + 9.0 * OMEGA);

        mxx = 2.0 * (9.0 * rhoI * mxxI - 6.0 * mxyI * rhoI + (rhoVar)) / (9.0 * (rhoVar));
        myy = -2.0 * (6.0 * rhoI * mxyI - 9.0 * myyI * rhoI - (rhoVar)) / (9.0 * (rhoVar));
        mxy = -(18.0 * rhoI * mxxI - 132.0 * mxyI * rhoI + 18.0 * rhoI * myyI + 7.0 * (rhoVar)) / (27.0 * (rhoVar));

        break;
    }
    case SOUTH_EAST:
    {
        const dfloat rhoI = pop[0] + pop[1] + pop[4] + pop[8];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[1] + pop[8]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[4] + pop[8]) * inv_rhoI - cs2;
        const dfloat mxyI = -pop[8] * inv_rhoI;

        ux = 0.0;
        uy = 0.0;

        rhoVar = -12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * (OMEGA - 1.0) * mxxI + 7.0 * (OMEGA - 1.0) * mxyI + 3.0 * OMEGA * myyI) / (16.0 + 9.0 * OMEGA);

        mxx = 2.0 * (9.0 * rhoI * mxxI + 6.0 * mxyI * rhoI + (rhoVar)) / (9.0 * (rhoVar));
        myy = 2.0 * (6.0 * rhoI * mxyI + 9.0 * myyI * rhoI + (rhoVar)) / (9.0 * (rhoVar));
        mxy = -(-18.0 * rhoI * mxxI - 132.0 * mxyI * rhoI - 18.0 * rhoI * myyI - 7.0 * (rhoVar)) / (27.0 * (rhoVar));

        break;
    }
    case NORTH_WEST:
    {
        const dfloat rhoI = pop[0] + pop[2] + pop[3] + pop[6];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[3] + pop[6]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[2] + pop[6]) * inv_rhoI - cs2;
        const dfloat mxyI = -pop[6] * inv_rhoI;

        ux = U0;
        uy = 0.0;

        rhoVar = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * (OMEGA - 1.0) * mxxI + 7.0 * (OMEGA - 1.0) * mxyI + 3.0 * OMEGA * myyI) / (-2.0 * (8.0 + 7.0 * U0) + OMEGA * (-9.0 - U0 + 15.0 * U0 * U0));

        mxx = 2.0 * (9.0 * rhoI * mxxI + 6.0 * mxyI * rhoI + (rhoVar) + 2.0 * U0 * (rhoVar)) / (9.0 * (rhoVar));
        myy = -2.0 * (-6.0 * rhoI * mxyI - 9.0 * myyI * rhoI - (rhoVar) + U0 * (rhoVar)) / (9.0 * (rhoVar));
        mxy = -(-18.0 * rhoI * mxxI - 132.0 * mxyI * rhoI - 18.0 * rhoI * myyI - 7.0 * (rhoVar) + 7.0 * U0 * (rhoVar)) / (27.0 * (rhoVar));

        break;
    }
    case NORTH_EAST:
    {
        const dfloat rhoI = pop[0] + pop[1] + pop[2] + pop[5];
        const dfloat inv_rhoI = 1.0 / rhoI;
        const dfloat mxxI = (pop[1] + pop[5]) * inv_rhoI - cs2;
        const dfloat myyI = (pop[2] + pop[5]) * inv_rhoI - cs2;
        const dfloat mxyI = pop[5] * inv_rhoI;

        ux = U0;
        uy = 0.0;

        rhoVar = 12.0 * rhoI * (-3.0 - 3.0 * myyI + 3.0 * (OMEGA - 1.0) * mxxI - 7.0 * (OMEGA - 1.0) * mxyI + 3.0 * OMEGA * myyI) / (2.0 * (-8.0 + 7.0 * U0) + OMEGA * (-9.0 + U0 + 15.0 * U0 * U0));

        mxx = -2.0 * (-9.0 * rhoI * mxxI + 6.0 * mxyI * rhoI - (rhoVar) + 2.0 * U0 * (rhoVar)) / (9.0 * (rhoVar));
        myy = 2.0 * (-6.0 * rhoI * mxyI + 9.0 * myyI * rhoI + (rhoVar) + U0 * (rhoVar)) / (9.0 * (rhoVar));
        mxy = -(18.0 * rhoI * mxxI - 132.0 * mxyI * rhoI + 18.0 * rhoI * myyI + 7.0 * (rhoVar) + 7.0 * U0 * (rhoVar)) / (27.0 * (rhoVar));

        break;
    }

    default:
        break;
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