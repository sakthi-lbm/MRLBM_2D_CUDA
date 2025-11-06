#!/bin/bash

LT="D2Q9"
CASE=${1:-ldc}    # use 'ldc' as default if not given

# Detect GPU compute capability if not manually set
if [ -z "$CompCap" ]; then
    CompCap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CompCap" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

echo "Building for Compute Capability sm_${CompCap}, Case: ${CASE}, Lattice: ${LT}"

# Cleanup old binaries and outputs safely
rm -f ../*sim_${LT}_sm${CompCap} 2>/dev/null
rm -rf ../LDC/${CASE}/ 2>/dev/null

# Compile all .cu files inside src/
nvcc -std=c++17 \
    -gencode arch=compute_${CompCap},code=sm_${CompCap} \
    -rdc=true -O3 --restrict \
    -Iinclude \
    $(find src -name "*.cu") \
    -lcudadevrt -lcurand \
    -o ../${CASE}sim_${LT}_sm${CompCap}

# Run the simulation
cd ../
./${CASE}sim_${LT}_sm${CompCap}
