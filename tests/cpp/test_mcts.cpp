#include "mcts.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

void get_dummy_evaluation(std::vector<float>& policy, float& value) {
    policy.assign(GameState::NUM_CELLS, 0.0f);
    value = 0.0f;
}

void test_resumable_mcts_deterministic_execution() {
    GameState state;
    MCTSTree tree(state, false); 

    int total_simulations = 100;
    int completed_simulations = 0;

    while (!tree.is_done(total_simulations)) {
        auto leaves = tree.request_leaves(1);
        if (!leaves.empty()) {
            std::vector<float> policy;
            float value;
            get_dummy_evaluation(policy, value);
            
            tree.submit_results({policy}, {value});
        }
        completed_simulations++;
        assert(completed_simulations <= total_simulations + 5);
    }
    
    assert(tree.is_done(total_simulations));
    auto root_policy = tree.root_policy(1.0f);
    float prob_sum = 0.0f;
    for (float p : root_policy) { prob_sum += p; }
    assert(std::abs(prob_sum - 1.0f) < 1e-4f);
}

void test_mcts_tree_reuse() {
    GameState state;
    MCTSTree tree(state, false);

    while (!tree.is_done(50)) {
        auto leaves = tree.request_leaves(1);
        if (!leaves.empty()) {
            std::vector<float> policy(GameState::NUM_CELLS, 0.0f);
            tree.submit_results({policy}, {0.1f}); 
        }
    }

    auto policy = tree.root_policy(0.01f); 
    int best_action = -1;
    float max_p = -1.0f;
    for (int i = 0; i < GameState::NUM_CELLS; ++i) {
        if (policy[i] > max_p) {
            max_p = policy[i];
            best_action = i;
        }
    }
    assert(best_action != -1);
    
    tree.advance(best_action);
    assert(!tree.is_done(50));
}

void test_tactical_one_move_win() {
    std::vector<int> board(81, 0);
    board[0] = 1;  board[1] = 1;  board[2] = 0;
    board[3] = -1; board[4] = -1; 
    board[9] = 1; board[10] = 1; board[11] = 1; 
    board[18] = 1; board[19] = 1; board[20] = 1; 
    board[27] = -1; board[28] = -1; 
    board[36] = -1; board[37] = -1; 
    board[45] = -1; board[46] = -1; 
    GameState state = GameState::from_array(board, 0);
    
    assert(state.get_current_player() == GameState::PLAYER_X);
    assert(state.legal_mask()[2] == true); 

    MCTSTree tree(state, false); 

    while (!tree.is_done(20)) {
        auto leaves = tree.request_leaves(2);
        if (!leaves.empty()) {
            std::vector<std::vector<float>> policies;
            std::vector<float> values;
            for(size_t i=0; i<leaves.size(); ++i) {
                std::vector<float> policy;
                float value;
                get_dummy_evaluation(policy, value);
                policies.push_back(policy);
                values.push_back(value);
            }
            tree.submit_results(policies, values);
        }
    }

    auto policy = tree.root_policy(0.1f);
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