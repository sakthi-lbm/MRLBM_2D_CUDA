#ifndef MAIN_CUH
#define MAIN_CUH

#include "var.h"
#include "solver/colrec/second_order/collision_streaming.cuh"
#include "save_data.cuh"

// ---------------Host memory allocation----------------------
inline void allocateHostMemory(nodeVar &h_fMom)
{
    checkCudaErrors(cudaMallocHost(&h_fMom.nodeType, NUM_LBM_NODES * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocHost(&(h_fMom.rho), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.ux), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.uy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxx), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.myy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMallocHost(&(h_fMom.mxy), MEM_SIZE_LBM_NODES));
}

//---------------- Device Memory allocation------------------------
inline void allocateDeviceMemory(nodeVar &d_fMom)
{
    checkCudaErrors(cudaMalloc(&d_fMom.nodeType, NUM_LBM_NODES * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&(d_fMom.rho), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.ux), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.uy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxx), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.myy), MEM_SIZE_LBM_NODES));
    checkCudaErrors(cudaMalloc(&(d_fMom.mxy), MEM_SIZE_LBM_NODES));
}

inline void allocateHaloInterfaceMemory(haloData &fHalo_interface, haloData &gHalo_interface)
{
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_WEST, NUM_HALO_FACE_X * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.X_EAST, NUM_HALO_FACE_X * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_SOUTH, NUM_HALO_FACE_Y * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&fHalo_interface.Y_NORTH, NUM_HALO_FACE_Y * QF * sizeof(dfloat)));

    checkCudaErrors(cudaMalloc(&gHalo_interface.X_WEST, NUM_HALO_FACE_X * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.X_EAST, NUM_HALO_FACE_X * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_SOUTH, NUM_HALO_FACE_Y * QF * sizeof(dfloat)));
    checkCudaErrors(cudaMalloc(&gHalo_interface.Y_NORTH, NUM_HALO_FACE_Y * QF * sizeof(dfloat)));
}

//---------------------------- Swap halo interface pointers: fHalo <--> gHalo
__host__ __device__ inline void swapPointers(dfloat *&pt1, dfloat *&pt2)
{
    dfloat *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

inline void swapHaloInterfaces(haloData &fHalo, haloData &gHalo)
{
    swapPointers(fHalo.X_WEST, gHalo.X_WEST);
    swapPointers(fHalo.X_EAST, gHalo.X_EAST);
    swapPointers(fHalo.Y_SOUTH, gHalo.Y_SOUTH);
    swapPointers(fHalo.Y_NORTH, gHalo.Y_NORTH);
}

void copyHaloInterfaces(haloData &dst, const haloData &src)
{
    checkCudaErrors(cudaMemcpy(dst.X_WEST, src.X_WEST, sizeof(dfloat) * NUM_HALO_FACE_X * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.X_EAST, src.X_EAST, sizeof(dfloat) * NUM_HALO_FACE_X * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_SOUTH, src.Y_SOUTH, sizeof(dfloat) * NUM_HALO_FACE_Y * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dst.Y_NORTH, src.Y_NORTH, sizeof(dfloat) * NUM_HALO_FACE_Y * QF, cudaMemcpyDeviceToDevice));
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

inline void freeHaloInterfaceMemory(haloData &fHalo_interface, haloData &gHalo_interface)
{
    cudaFree(fHalo_interface.X_WEST);
    cudaFree(fHalo_interface.X_EAST);
    cudaFree(fHalo_interface.Y_SOUTH);
    cudaFree(fHalo_interface.Y_NORTH);
    cudaFree(gHalo_interface.X_WEST);
    cudaFree(gHalo_interface.X_EAST);
    cudaFree(gHalo_interface.Y_SOUTH);
    cudaFree(gHalo_interface.Y_NORTH);
}

// ------------------------Copy host --> Device--------------------------------
inline void copyMomentsHostToDevice(nodeVar &df_Mom, const nodeVar &h_fMom)
{
    checkCudaErrors(cudaMemcpy(df_Mom.rho, h_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.ux, h_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.uy, h_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.mxx, h_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.myy, h_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(df_Mom.mxy, h_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyHostToDevice));
}

// ------------------------Copy Device --> Host--------------------------------
inline void copyMomentsDeviceToHost(nodeVar &h_fMom, const nodeVar &d_fMom)
{
    checkCudaErrors(cudaMemcpy(h_fMom.rho, d_fMom.rho, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.ux, d_fMom.ux, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.uy, d_fMom.uy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxx, d_fMom.mxx, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.myy, d_fMom.myy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_fMom.mxy, d_fMom.mxy, MEM_SIZE_LBM_NODES, cudaMemcpyDeviceToHost));
}

#endif // MAIN_CUH