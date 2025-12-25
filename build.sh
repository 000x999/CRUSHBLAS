#!/bin/bash

# build.sh
echo "========= BUILDING CPPDL ========="

# Arg1: configure preset name
# Arg2: build preset name
# Arg3: AVX flag (ON/OFF or 1/0, optional)
# Arg4: BLAS flag
# Arg5: BUILD DEBUG

CONFIG_PRESET="${1:-linux-release}"
BUILD_PRESET="${2:-linux-release-build}"
AVX_FLAG="${3:-ON}"
BLAS_FLAG="${4:-ON}"
BUILD_DEBUG="${5:-ON}"

echo "[build] Using CONFIG_PRESET=$CONFIG_PRESET"
echo "[build] Using BUILD_PRESET=$BUILD_PRESET"
echo "[build] Using USE_AVX256=$AVX_FLAG"
echo "[build] Using USE_BLAS=$BLAS_FLAG"
echo "[build] Using DEBUG=$BUILD_DEBUG"

echo "[build] Configuring with CMake..."
cmake --preset "$CONFIG_PRESET" -DUSE_AVX256="$AVX_FLAG" -DUSE_BLAS="$BLAS_FLAG" -DDEBUG="$BUILD_DEBUG"

if [ $? -ne 0 ]; then
    echo "========= BUILD FAILED ========="
    exit 1
fi

echo "[build] Building..."
cmake --build --preset "$BUILD_PRESET"

if [ $? -ne 0 ]; then
    echo "========= BUILD FAILED ========="
    exit 1
fi

echo "========= BUILD COMPLETE ========="
