#ifndef INITIALIZELBM_H
#define INITIALIZELBM_H

#include "../var.h"

void initialize_domain(nodeVar fMom);
__global__ void gpu_initialize_nodeType(nodeVar fMom);
__global__ void gpu_initialize_moments(nodeVar fMom);

#endif // INITIALIZELBM_H