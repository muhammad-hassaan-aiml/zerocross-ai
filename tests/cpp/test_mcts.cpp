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

int main() {
    std::cout << "Running MCTS Core unit tests..." << std::endl;
    
    test_resumable_mcts_deterministic_execution();
    test_mcts_tree_reuse();
    
    std::cout << "All MCTS unit tests passed successfully." << std::endl;
    return 0;
}