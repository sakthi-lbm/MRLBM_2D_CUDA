#ifndef GLOBAL_STRUCTS_H
#define GLOBAL_STRUCTS_H

#include "var.h"

struct nodeVar
{
    unsigned int *nodeType;
    dfloat *rho;
    dfloat *ux;
    dfloat *uy;
    dfloat *mxx;
    dfloat *myy;
    dfloat *mxy;
};

struct haloData
{
    dfloat *X_WEST;
    dfloat *X_EAST;
    dfloat *Y_SOUTH;
    dfloat *Y_NORTH;
};

#endif // GLOBAL_STRUCTS_H