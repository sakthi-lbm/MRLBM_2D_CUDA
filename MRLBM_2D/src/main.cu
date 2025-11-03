#include <iostream>
#include <fstream>

#include "main.cuh"
#include "solver/initializeLBM.cuh"

int main()
{
    create_output_directory();
    writeSimInfo();

    timestep sim_start_time = std::chrono::high_resolution_clock::now();
    timestep start_time = std::chrono::high_resolution_clock::now();
    timestep end_time;
    dfloat mlups;

    nodeVar h_fMom;
    nodeVar d_fMom;
    haloData fHalo_interface;
    haloData gHalo_interface;

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);
    allocateHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    initialize_domain(d_fMom, gHalo_interface);

    copyMomentsDeviceToHost(h_fMom, d_fMom);
    checkCudaErrors(cudaMemcpy(h_fMom.nodeType, d_fMom.nodeType, NUM_LBM_NODES * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Time loop
    for (int iter = 0; iter <= MAX_ITER; iter++)
    {
    }

    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);

    return 0;
}