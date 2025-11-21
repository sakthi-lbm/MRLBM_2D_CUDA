#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "var.h"

inline void post_process(nodeVar fMom)
{

#ifdef LDC
    sample_ux_vertical_line(fMom);
    sample_uy_horizontal_line(fMom);

#endif
}

#endif