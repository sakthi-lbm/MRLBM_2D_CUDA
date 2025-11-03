#ifndef INITIALIZELBM_H
#define INITIALIZELBM_H

#include "../var.h"
#include "../halo_interface/halo_interface.cuh"

void initialize_domain(nodeVar fMom, haloData gHalo);
// __global__ void gpu_initialize_nodeType(nodeVar fMom);
__global__ void gpu_initialize_Moments_nodeType_GhostInterface(nodeVar fMom, haloData gHalo);

#endif // INITIALIZELBM_H