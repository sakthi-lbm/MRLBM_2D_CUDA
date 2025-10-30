#ifndef DEFINITIONS_H
#define DEFINITIIONS_H

#include "var.h"

constexpr int NUM_NODES = NX * NY;
constexpr size_t MEM_SIZE_MOM = toSize_t(NUM_NODES) * sizeof(dfloat);

#endif // DEFINITIONS_H