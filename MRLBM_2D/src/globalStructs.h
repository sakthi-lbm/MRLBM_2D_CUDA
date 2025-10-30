#ifndef GLOBAL_STRUCTS_H
#define GLOBAL_STRUCTS_H

#include "var.h"

struct nodeVar
{
    dfloat *rho;
    dfloat *ux;
    dfloat *uy;
    dfloat *mxx;
    dfloat *myy;
    dfloat *mxy;
};

#endif // GLOBAL_STRUCTS_H