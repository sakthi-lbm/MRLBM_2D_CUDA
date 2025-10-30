#include <iostream>
#include <fstream>

#include "main.cuh"

int main()
{

    nodeVar h_fMom;
    nodeVar d_fMom;

    allocateHostMemory(h_fMom);
    allocateDeviceMemory(d_fMom);

    freeHostMemory(h_fMom);
    freeDeviceMemory(d_fMom);

    return 0;
}