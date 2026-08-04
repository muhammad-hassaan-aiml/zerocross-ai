#pragma once

#include "game_state.h"
#include <memory>
#include <vector>
#include <optional>
#include <array>
#include <random>

struct MCTSNode {
    std::weak_ptr<MCTSNode> parent;
    std::vector<std::shared_ptr<MCTSNode>> children;
    
    int visit_count = 0;
    float total_value = 0.0f;
    float prior_p = 0.0f;
    int action_taken = -1;
    
    int virtual_loss = 0;
    
    GameState state; 

    bool is_expanded() const {
        return !children.empty();
    }
};

class MCTSTree {
public:
    MCTSTree(const GameState& root_state, bool add_noise = true);

    std::vector<std::array<float, 486>> request_leaves(int batch_size);
    void submit_results(const std::vector<std::vector<float>>& policies, const std::vector<float>& values);
    
    bool is_done(int n_simulations) const;
    std::vector<float> root_policy(float temperature) const;
    void advance(int action);

private:
    std::shared_ptr<MCTSNode> root;
    std::vector<std::shared_ptr<MCTSNode>> pending_leaves;
    bool add_noise_flag;

    static constexpr float DIRICHLET_ALPHA = 0.3f; 
    static constexpr float DIRICHLET_EPSILON = 0.25f; 
    
    std::mt19937 rng;

    std::vector<float> generate_dirichlet(int num_valid_actions);
    float calculate_puct(std::shared_ptr<MCTSNode> parent, std::shared_ptr<MCTSNode> child) const;
};

class ParallelMCTS {
public:
    ParallelMCTS(int num_games, bool add_noise);

    void set_state(int game_idx, const GameState& state);
    void advance(int game_idx, int action);
    bool is_done(int game_idx, int n_simulations) const;
    std::vector<float> root_policy(int game_idx, float temperature) const;

    std::vector<std::array<float, 486>> request_batch(int n_simulations, int batch_per_tree);
    void submit_batch(const std::vector<std::vector<float>>& policies, const std::vector<float>& values);
    
    // Expose the mapping so Python bindings can route evaluations
    const std::vector<int>& get_leaf_game_mapping() const { return leaf_game_mapping; }

private:
    int num_games;
    bool add_noise_flag;
    std::vector<std::shared_ptr<MCTSTree>> trees;
    std::vector<int> leaf_game_mapping;
};