#ifndef POST_CASE_H
#define POST_CASE_H

#include "../../var.h"

inline void sample_ux_vertical_line(nodeVar fMom)
{
    dfloat x_center = (NX - 1) / 2;

    int x = toInt(floor(x_center));
    int xp1 = x + 1;

    std::string strInf3 = PATH_FILES;
    strInf3 += "/";
    strInf3 += ID_SIM;
    strInf3 += "/";
    strInf3 += "ux_vertical.dat";
    std::ofstream uxfile(strInf3);

    for (size_t y = 0; y < NY; y++)
    {
        dfloat ux = fMom.ux[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
        dfloat uxp1 = fMom.ux[IDX_BLOCK(xp1 % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, xp1 / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
        dfloat ux_center = toDFloat(0.5) * (ux + uxp1);

        if (uxfile.is_open())
        {
            uxfile << toDFloat(y) / toDFloat(NY - 1) << " " << ux_center / U0 << std::endl;
        }
    }
}

inline void sample_uy_horizontal_line(nodeVar fMom)
{
    dfloat y_center = (NY - 1) / 2;

    int y = toInt(floor(y_center));
    int yp1 = y + 1;

    std::string strInf3 = PATH_FILES;
    strInf3 += "/";
    strInf3 += ID_SIM;
    strInf3 += "/";
    strInf3 += "uy_horizontal.dat";
    std::ofstream uyfile(strInf3);

    for (size_t x = 0; x < NX; x++)
    {
        dfloat uy = fMom.uy[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
        dfloat uyp1 = fMom.uy[IDX_BLOCK(x % BLOCK_THREAD_X, yp1 % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, yp1 / BLOCK_THREAD_Y)];
        dfloat uy_center = toDFloat(0.5) * (uy + uyp1);

        if (uyfile.is_open())
        {
            uyfile << toDFloat(x) / toDFloat(NX - 1) << " " << uy_center / U0 << std::endl;
        }
    }
}

#endif