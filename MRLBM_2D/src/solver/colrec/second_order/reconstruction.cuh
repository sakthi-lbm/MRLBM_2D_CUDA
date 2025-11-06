#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "../../../var.h"



__device__ inline void pop_reconstruction(const dfloat rhoVar, const dfloat uxVar, const dfloat uyVar, const dfloat mxxVar, const dfloat myyVar, const dfloat mxyVar, dfloat *pop)
{
    // for (size_t i = 0; i < Q; i++)
    // {
    //     const dfloat Hxx = d_cx[i] * d_cx[i] - cs2;
    //     const dfloat Hyy = d_cy[i] * d_cy[i] - cs2;
    //     const dfloat Hxy = d_cx[i] * d_cy[i];

    //     pop[i] = w[i] * rho * (toDFloat(1.0) + as2 * (ux * d_cx[i] + uy * d_cy[i]) + toDFloat(0.5) * as2 * as2 * (Hxx * mxx + Hyy * myy + toDFloat(2.0) * Hxy * mxy));
    // }

    const dfloat rho = rhoVar * F_M_0_SCALE;
    const dfloat ux = uxVar * F_M_I_SCALE;
    const dfloat uy = uyVar * F_M_I_SCALE;
    const dfloat mxx = mxxVar * F_M_II_SCALE;
    const dfloat myy = myyVar * F_M_II_SCALE;
    const dfloat mxy = mxyVar * F_M_IJ_SCALE;

    dfloat pics2 = toDFloat(1.0) - cs2 * (mxx + myy);

    dfloat multiplyTerm = W0 * (rho);
    pop[0] = multiplyTerm * (pics2);

    multiplyTerm = W1 * (rho);
    pop[1] = multiplyTerm * (pics2 + ux + mxx);
    pop[2] = multiplyTerm * (pics2 + uy + myy);
    pop[3] = multiplyTerm * (pics2 - ux + mxx);
    pop[4] = multiplyTerm * (pics2 - uy + myy);

    multiplyTerm = W2 * (rho);
    pop[5] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy);
    pop[6] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy);
    pop[7] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy);
    pop[8] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy);
}

#endif