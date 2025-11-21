#include <iostream>
#include <fstream>

#include "main.cuh"
#include "solver/initializeLBM.cuh"

int main()
{
    gpu_properties();
    create_output_directory();
    writeSimInfo();

    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    timestep sim_start_time = std::chrono::high_resolution_clock::now();
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
    copyHaloInterfaces(fHalo_interface, gHalo_interface);

    write_grid();

    timestep start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        MomCollisionStreaming<<<grid, block>>>(d_fMom, fHalo_interface, gHalo_interface);
        checkKernelExecution();

        checkCudaErrors(cudaDeviceSynchronize());
        swapHaloInterfaces(fHalo_interface, gHalo_interface);

        if (iter % MACR_SAVE == 0)
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            write_solution(h_fMom, iter);

            printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);
        }
    }
    copyMomentsDeviceToHost(h_fMom, d_fMom);
    if (POST_PROCESS)
    {
        post_process(h_fMom);
    }

    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);
    freeHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    return 0;
}