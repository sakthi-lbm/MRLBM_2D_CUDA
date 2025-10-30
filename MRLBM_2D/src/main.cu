#include <iostream>
#include <fstream>

#include "main.cuh"
#include "solver/initializeLBM.cuh"

int main()
{

    nodeVar h_fMom;
    nodeVar d_fMom;

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);

    gpu_initialize_moments<<<grid, block>>>(d_fMom);




    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);

    return 0;
}