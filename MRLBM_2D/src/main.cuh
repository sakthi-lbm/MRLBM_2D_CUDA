#ifndef MAIN_CUH
#define MAIN_CUH

#include "var.h"

// ---------------Host memory allocation----------------------
inline void allocateHostMemory(nodeVar &h_fMom)
{
    cudaMallocHost(&h_fMom.nodeType, NUM_LBM_NODES * sizeof(unsigned int));
    cudaMallocHost(&(h_fMom.rho), MEM_SIZE_LBM_NODES);
    cudaMallocHost(&(h_fMom.ux), MEM_SIZE_LBM_NODES);
    cudaMallocHost(&(h_fMom.uy), MEM_SIZE_LBM_NODES);
    cudaMallocHost(&(h_fMom.mxx), MEM_SIZE_LBM_NODES);
    cudaMallocHost(&(h_fMom.myy), MEM_SIZE_LBM_NODES);
    cudaMallocHost(&(h_fMom.mxy), MEM_SIZE_LBM_NODES);
}

//---------------- Device Memory allocation------------------------
inline void allocateDeviceMemory(nodeVar &d_fMom)
{
    cudaMalloc(&d_fMom.nodeType, NUM_LBM_NODES * sizeof(unsigned int));
    cudaMalloc(&(d_fMom.rho), MEM_SIZE_LBM_NODES);
    cudaMalloc(&(d_fMom.ux), MEM_SIZE_LBM_NODES);
    cudaMalloc(&(d_fMom.uy), MEM_SIZE_LBM_NODES);
    cudaMalloc(&(d_fMom.mxx), MEM_SIZE_LBM_NODES);
    cudaMalloc(&(d_fMom.myy), MEM_SIZE_LBM_NODES);
    cudaMalloc(&(d_fMom.mxy), MEM_SIZE_LBM_NODES);
}

//-------------- Freeing host memory---------------------------
inline void freeHostMemory(nodeVar &h_fMom)
{
    cudaFreeHost(h_fMom.nodeType);
    cudaFreeHost(h_fMom.rho);
    cudaFreeHost(h_fMom.ux);
    cudaFreeHost(h_fMom.uy);
    cudaFreeHost(h_fMom.mxx);
    cudaFreeHost(h_fMom.myy);
    cudaFreeHost(h_fMom.mxy);
}

//--------------- Freeing device Memory------------------------
inline void freeDeviceMemory(nodeVar &d_fMom)
{
    cudaFree(d_fMom.nodeType);
    cudaFree(d_fMom.rho);
    cudaFree(d_fMom.ux);
    cudaFree(d_fMom.uy);
    cudaFree(d_fMom.mxx);
    cudaFree(d_fMom.myy);
    cudaFree(d_fMom.mxy);
}

// ------------------------Copy host --> Device--------------------------------
inline void copyMomentsHostToDevice(nodeVar &df_Mom, const nodeVar h_fMom)
{
    cudaMemcpy(df_Mom.rho, h_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(df_Mom.ux, h_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(df_Mom.uy, h_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(df_Mom.mxx, h_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(df_Mom.myy, h_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(df_Mom.mxy, h_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice);
}

// ------------------------Copy Device --> Host--------------------------------
inline void copyMomentsDeviceToHost(nodeVar &h_fMom, const nodeVar &d_fMom)
{
    cudaMemcpy(h_fMom.rho, d_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fMom.ux, d_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fMom.uy, d_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fMom.mxx, d_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fMom.myy, d_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fMom.mxy, d_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost);
}

#endif // MAIN_CUH