#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include "../var.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

inline void create_output_directory()
{
#if defined(_WIN32)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "\\\\"; // adds "\\"
    strPath += ID_SIM;
    std::string cmd = "md ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !_WIN32

#if defined(__APPLE__) || defined(__MACH__) || defined(__linux__)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "/";
    strPath += ID_SIM;
    strPath += "/";
    strPath += "plots";
    std::string cmd = "mkdir -p ";
    cmd += strPath;
    const int i = system(cmd.c_str());
    static_cast<void>(i);
    return;

#endif
    printf("I don't know how to setup folders for your operational system :(\n");
    return;
}

inline std::string getSimInfoString()
{
    std::string precision;
    if (typeid(dfloat) == typeid(double))
        precision = "double";
    else if (typeid(dfloat) == typeid(float))
        precision = "float";
    else
        precision = "unknown";

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);

    const int labelWidth = 25;

    out << "========================= SIMULATION INFORMATION =========================\n";
    out << std::left;
    out << std::setw(labelWidth) << "Simulation ID" << " : " << ID_SIM << "\n";
    out << std::setw(labelWidth) << "Velocity set" << " : D2Q9\n";
    out << std::setw(labelWidth) << "Re" << " : " << RE << "\n";
    out << std::setw(labelWidth) << "Precision" << " : " << precision << "\n";
    out << std::setw(labelWidth) << "NX" << " : " << NX << "\n";
    out << std::setw(labelWidth) << "NY" << " : " << NY << "\n";
    out << std::setw(labelWidth) << "Total Grid points" << " : " << NY * NY << "\n";
    out << std::setw(labelWidth) << "uo" << " : " << U0 << "\n";
    out << std::setw(labelWidth) << "Viscosity" << " : " << VISC << "\n";
    out << std::setw(labelWidth) << "Tau" << " : " << TAU << "\n";
    out << std::setw(labelWidth) << "Omega" << " : " << OMEGA << "\n";
    // out << std::setw(labelWidth) << "Macr_save" << " : " << MACR_SAVE << "\n";
    out << std::setw(labelWidth) << "Nsteps" << " : " << MAX_ITER << "\n";
    // out << std::setw(labelWidth) << "MLUPS" << " : " << MLUPS << "\n";
    out << "\n";
    out << "----------------------------- CUDA Parameters -----------------------------\n";
    out << std::setw(labelWidth) << "Num of Blocks in X" << " : " << GRID_BLOCK_X << "\n";
    out << std::setw(labelWidth) << "Num of Blocks in X" << " : " << GRID_BLOCK_Y << "\n";
    out << std::setw(labelWidth) << "Total threads per block" << " : " << THREADS_PER_BLOCK << "\n";
    out << std::setw(labelWidth) << "Available Shared Memory (kb)" << " : " << MAX_SHARED_MEM_BYTES/BYTES_PER_KB << "\n";
    out << std::setw(labelWidth) << "Shared Memory used (kb)" << " : " << USED_SHARED_MEMORY/BYTES_PER_KB << "\n";
    out << std::setw(labelWidth) << "Global Memory used (kb)" << " : " << USED_GLOBAL_MEMORY/BYTES_PER_KB << "\n";
    out << "==========================================================================\n";

    return out.str();
}

inline void writeSimInfo()
{
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "/";
    strPath += ID_SIM;
    strPath += "/";
    strPath += ID_SIM;
    strPath += "_Sim_info.txt";

    std::string info = getSimInfoString();

    // Print to console
    std::cout << info;

    // Write to file
    std::ofstream outFile(strPath, std::ios::out);
    if (!outFile)
    {
        std::cerr << "Error: Could not open file " << strPath << " for writing simulation info.\n";
        return;
    }

    outFile << info;
    outFile.close();
}

inline void calculate_mlups(timestep &tstart, timestep &tend, int steps, dfloat &mlups)
{
    tend = std::chrono::high_resolution_clock::now();
    double step_time = std::chrono::duration<double>(tend - tstart).count();
    if (step_time > 0.0)
        mlups = (NUM_LBM_NODES * steps / 1e6) / step_time;
    else
        mlups = 0.0;

    tstart = std::chrono::high_resolution_clock::now();
}

#endif