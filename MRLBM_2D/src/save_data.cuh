#ifndef SAVE_DATA_H
#define SAVE_DATA_H

#include "var.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream> // std::cout, std::fixed
#include <iomanip>  // std::setprecision
#include "globalStructs.h"

void write_grid();

void write_solution(nodeVar h_fMom, size_t iter);

#endif