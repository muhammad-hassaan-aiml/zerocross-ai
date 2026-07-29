#include "mcts.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

// Simulates a neural network returning unnormalized logits 
// (all zeros equals uniform probability after softmax)
void get_dummy_evaluation(std::vector<float>& policy, float& value) {
    policy.assign(GameState::NUM_CELLS, 0.0f);
    value = 0.0f; 
}

void test_resumable_mcts_deterministic_execution() {
    GameState state;
    // Disable noise for strictly deterministic validation
    MCTSTree tree(state, false); 

    int total_simulations = 100;
    int completed_simulations = 0;

    // Drive the resumable interface sequentially to simulate a blocking search loop
    while (!tree.is_done(total_simulations)) {
        auto leaf = tree.request_leaf();
        
        // If leaf has a value, it is not terminal and requires neural network evaluation
        if (leaf.has_value()) {
            std::vector<float> policy;
            float value;
            get_dummy_evaluation(policy, value);
            
            // Submit the dummy NN output to backpropagate and expand the tree
            tree.submit_result(policy, value);
        }
        completed_simulations++;
        
        // Failsafe: Ensures tree traversal doesn't enter an infinite PUCT loop
        assert(completed_simulations <= total_simulations + 5);
    }

    // Validate tree state after reaching the simulation target
    assert(tree.is_done(total_simulations));
    
    // Extract root policy using standard temperature
    auto root_policy = tree.root_policy(1.0f);
    float prob_sum = 0.0f;
    for (float p : root_policy) {
        prob_sum += p;
    }
    
    // Assert the extracted policy forms a valid probability distribution summing to ~1.0
    assert(std::abs(prob_sum - 1.0f) < 1e-4f);
}

void test_mcts_tree_reuse() {
    GameState state;
    MCTSTree tree(state, false);

    // Run an initial batch of 50 simulations
    while (!tree.is_done(50)) {
        auto leaf = tree.request_leaf();
        if (leaf.has_value()) {
            std::vector<float> policy(GameState::NUM_CELLS, 0.0f);
            tree.submit_result(policy, 0.1f); // Arbitrary slight positive value
        }
    }

    // Find the most visited action to simulate a chosen move
    auto policy = tree.root_policy(0.01f); // Near-zero temperature acts as argmax
    int best_action = -1;
    float max_p = -1.0f;
    for (int i = 0; i < GameState::NUM_CELLS; ++i) {
        if (policy[i] > max_p) {
            max_p = policy[i];
            best_action = i;
        }
    }

    assert(best_action != -1);

    // Advance the tree. This critically tests the std::weak_ptr implementation.
    // Unselected child branches will have their reference counts drop to zero and deallocate.
    tree.advance(best_action);

    // After advancing, the new root inherits only a fraction of the original 50 visits.
    // The tree should correctly report that it is no longer "done" for a 50-simulation target.
    assert(!tree.is_done(50));
}

void test_tactical_one_move_win() {
    // Construct a board where Player X has already won macro-grids 1 and 2,
    // and can win macro-grid 0 (and thus the whole game) on action 2.
    std::vector<int> board(81, 0);

    // Macro-grid 0: X has cells 0 & 1. Cell 2 is open for the win!
    board[0] = 1;  board[1] = 1;  board[2] = 0;
    board[3] = -1; board[4] = -1; // O has 2 stones here

    // Macro-grid 1: Won by X
    board[9] = 1; board[10] = 1; board[11] = 1; // X has 3 stones here

    // Macro-grid 2: Won by X
    board[18] = 1; board[19] = 1; board[20] = 1; // X has 3 stones here

    // Add opponent stones elsewhere to exactly balance turn count (8 X stones, 8 O stones -> X to move)
    board[27] = -1; board[28] = -1; // O has 2 stones
    board[36] = -1; board[37] = -1; // O has 2 stones
    board[45] = -1; board[46] = -1; // O has 2 stones -> Total O = 8!

    // Active grid is macro-grid 0
    GameState state = GameState::from_array(board, 0);
    
    assert(state.get_current_player() == GameState::PLAYER_X);
    assert(state.legal_mask()[2] == true); // Move 2 must be legal

    MCTSTree tree(state, false); // No noise

    // Run 20 MCTS simulations with uniform dummy neural network evaluation
    while (!tree.is_done(20)) {
        auto leaf = tree.request_leaf();
        if (leaf.has_value()) {
            std::vector<float> policy;
            float value;
            get_dummy_evaluation(policy, value);
            tree.submit_result(policy, value);
        }
    }

    // Extract policy with low temperature (argmax behavior)
    auto policy = tree.root_policy(0.1f);

    // Action 2 MUST be identified as the winning move and receive the vast majority of visits
    assert(policy[2] > 0.5f);
}

int main() {
    std::cout << "Running MCTS Core unit tests..." << std::endl;
    
    test_resumable_mcts_deterministic_execution();
    test_mcts_tree_reuse();
    test_tactical_one_move_win();
    
    std::cout << "All MCTS unit tests passed successfully." << std::endl;
    return 0;
}