#ifndef MAIN_CUH
#define MAIN_CUH

#include "globalStructs.h"
#include "definitions.h"

inline void allocateHostMemory(nodeVar h_fMom)
{
    cudaMallocHost((void **)&h_fMom.rho, MEM_SIZE_MOM);
    cudaMallocHost(&h_fMom.ux, MEM_SIZE_MOM);
    cudaMallocHost(&h_fMom.uy, MEM_SIZE_MOM);
    cudaMallocHost(&h_fMom.mxx, MEM_SIZE_MOM);
    cudaMallocHost(&h_fMom.myy, MEM_SIZE_MOM);
    cudaMallocHost(&h_fMom.mxy, MEM_SIZE_MOM);
}

inline void freeHostMemory(nodeVar h_fMom)
{
    cudaFreeHost(h_fMom.rho);
    cudaFreeHost(h_fMom.ux);
    cudaFreeHost(h_fMom.uy);
    cudaFreeHost(h_fMom.mxx);
    cudaFreeHost(h_fMom.myy);
    cudaFreeHost(h_fMom.mxy);
}

// inline void allocateDeviceMemory(nodeVar &d_fMom)
// {
// }

#endif // MAIN_CUH