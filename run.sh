#!/bin/bash

# run.sh
if [ ! -f "build/cppDL" ]; then
    echo "[run_cppDL] Error: build/cppDL not found."
    echo "Build it first with: cmake --build --preset mingw-debug-build"
    exit 1
fi

echo "[run_cppDL] Running cppDL..."
./build/cppDL
