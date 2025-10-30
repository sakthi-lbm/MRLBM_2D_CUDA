#include <iostream>
#include <fstream>

#include "main.cuh"

int main()
{

    nodeVar h_fMom;

    std::cout << OMEGA << TAU << as2 << Q << NX << NY << std::endl;

    if (h_fMom.rho == nullptr)
    {
        std::cerr << "Erro: memória não alocada!" << std::endl;
    }
    else
    {
        std::cout << "Memória alocada para h_fMom." << std::endl;
    }

    allocateHostMemory(h_fMom);

    if (h_fMom.rho == nullptr)
    {
        std::cerr << "Erro: memória não alocada!" << std::endl;
    }
    else
    {
        std::cout << "Memória alocada para h_fMom." << std::endl;
    }

    std::cout << sizeof(h_fMom.rho) << std::endl;

    freeHostMemory(h_fMom);

    return 0;
}