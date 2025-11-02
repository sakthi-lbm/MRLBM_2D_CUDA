#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Error checking macro
inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s\n",
                file, line, func, static_cast<int>(err), cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(err) __checkCudaErrors(err, #err, __FILE__, __LINE__)

// Kernel error checking (optional utility)
inline void checkKernelExecution()
{
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}





#endif //CUDA_HELPERS_H