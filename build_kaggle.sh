#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Step 1: Installing Python Dependencies"
python -m pip install --no-deps pybind11 pandas matplotlib

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
python -c "import sys; sys.path.extend(['.', 'build']); import zerocross_engine as z; s=z.GameState(); print('SUCCESS: GameState imported! Encode length:', len(s.encode())); t=z.MCTSTree(s); leaf=t.request_leaf(); t.submit_result([0.0]*81, 0.0) if leaf is not None else None; print('SUCCESS: MCTSTree imported and evaluated!')"

echo "Build and Smoke Tests Complete. Ready for Training."