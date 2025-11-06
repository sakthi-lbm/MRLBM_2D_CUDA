#ifndef COLLISION_STREAMING_H
#define COLLISION_STREAMING_H

#include "../../../var.h"
#include "../../../halo_interface/halo_interface.cuh"

__device__ void mom_collision(dfloat ux, dfloat uy, dfloat &mxx, dfloat &myy, dfloat &mxy);
__global__ void MomCollisionStreaming(nodeVar fMom, haloData fHalo, haloData gHalo);

#endif