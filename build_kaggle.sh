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
python - <<'EOF'
import sys
import traceback

sys.path.extend(['.', 'build'])

try:
    import zerocross_engine as z
    print("SUCCESS: zerocross_engine imported!")
except Exception:
    print("FATAL: Import failed.")
    traceback.print_exc()
    sys.exit(1)

# Basic GameState checks
s = z.GameState()
encode_len = len(s.encode())
assert encode_len == 486, f"Encode length mismatch: {encode_len}"
print(f"SUCCESS: GameState encode length is {encode_len}")

mask = s.legal_mask()
initial_legal_moves = sum(mask)
assert initial_legal_moves == 81, f"Expected 81 legal moves, got {initial_legal_moves}"
print(f"SUCCESS: Initial legal moves count on empty board is {initial_legal_moves}")

# Occupancy-guard check
action = 0
s.play(action)
mask_after = s.legal_mask()
legal_moves_after = sum(mask_after)
assert legal_moves_after < initial_legal_moves, "Legal moves should decrease after a play."
print("SUCCESS: Valid move played successfully.")

try:
    s.play(action)
    print("SUCCESS: Occupancy guard safely ignored playing on the same cell.")
except Exception:
    print("FATAL: Second play on same cell raised an unexpected exception.")
    traceback.print_exc()
    sys.exit(1)

# MCTS smoke test
t = z.MCTSTree(s)
leaf = t.request_leaf()
assert leaf is not None, "Initial request_leaf should not be None."

try:
    t.submit_result([0.0]*81, 0.0)
    print("SUCCESS: MCTSTree request_leaf and submit_result passed.")
except Exception:
    print("FATAL: submit_result raised an exception.")
    traceback.print_exc()
    sys.exit(1)
EOF

echo "Build and Smoke Tests Complete. Ready for Training."