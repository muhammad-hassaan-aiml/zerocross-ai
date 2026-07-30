#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "Step 1: Installing Python Dependencies"
pip install -r requirements.txt

echo "Step 2: Building C++ Engine"
mkdir -p build
cd build
cmake ..
make -j$(nproc) zerocross_engine test_game_state test_mcts

echo "Step 3: Running C++ Unit Tests"
./test_game_state
./test_mcts

echo "Step 4: Running Python Integration Smoke Test"
cd ..
python -c "import sys; sys.path.extend(['.', 'build']); import zerocross_engine as z; s=z.GameState(); print('SUCCESS: Engine imported! Encode length:', len(s.encode()))"

echo "Build and Smoke Tests Complete. Ready for Training."