#!/bin/bash
set -e

if [ ! -d "emsdk" ]; then
    echo "Step 1: Installing Emscripten SDK..."
    git clone https://github.com/emscripten-core/emsdk.git
    cd emsdk
    ./emsdk install latest
    ./emsdk activate latest
    cd ..
else
    echo "Step 1: Emscripten SDK already installed."
fi

echo "Step 2: Sourcing Emscripten Environment..."
source ./emsdk/emsdk_env.sh

echo "Step 3: Building WebAssembly Module..."
mkdir -p build_wasm
cd build_wasm
emcmake cmake ..
emmake make -j$(nproc) zerocross_wasm

echo "Step 4: Copying WASM artifacts to frontend..."
mkdir -p ../server/frontend
cp zerocross.js ../server/frontend/
cp zerocross.wasm ../server/frontend/

echo "SUCCESS: WASM Build Complete and ready for the browser!"