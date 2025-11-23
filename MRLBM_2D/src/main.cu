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

#ifdef TG
        dfloat *d_Ek_ana;
        dfloat *d_Ek_num;
        checkCudaErrors(cudaMalloc(&d_Ek_ana, sizeof(dfloat)));
        checkCudaErrors(cudaMalloc(&d_Ek_num, sizeof(dfloat)));

        checkCudaErrors(cudaMemset(d_Ek_ana, 0, sizeof(dfloat)));
        checkCudaErrors(cudaMemset(d_Ek_num, 0, sizeof(dfloat)));
#endif

        if (iter % MACR_SAVE == 0)
        {
            copyMomentsDeviceToHost(h_fMom, d_fMom);
            write_solution(h_fMom, iter);

            printf("\n---------------------- (%d/%d) %.2f%% ----------------------\n", iter, MAX_ITER, toFloat(iter) / toFloat(MAX_ITER) * 100.0f);

            if (POST_PROCESS)
            {
                compute_kinetic_energy<<<grid, block>>>(d_fMom, d_Ek_ana, d_Ek_num);
                dfloat Ek_0;
                dfloat Ek_num;
                checkCudaErrors(cudaMemcpy(&Ek_0, d_Ek_ana, sizeof(dfloat), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&Ek_num, d_Ek_num, sizeof(dfloat), cudaMemcpyDeviceToHost));

                dfloat Ek_analytical = Ek_0 * exp(-toDFloat(2.0) * VISC * K * K * iter);

                printf("Kinetic energy:\n Numerical = %.10e, Analytical = %.10e\n",
                       Ek_num, Ek_analytical);
            }
        }
    }
    copyMomentsDeviceToHost(h_fMom, d_fMom);
    if (POST_PROCESS)
    {
        post_process_function(h_fMom);
    }

    calculate_mlups(sim_start_time, end_time, MAX_ITER, mlups);
    std::cout << "GLOBAL MLUPS: " << mlups << std::endl;

    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);
    freeHaloInterfaceMemory(fHalo_interface, gHalo_interface);

    return 0;
}