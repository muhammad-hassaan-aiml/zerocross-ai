#pragma once

#include "game_state.h"
#include <memory>
#include <vector>
#include <optional>
#include <array>
#include <random>

struct MCTSNode {
    std::weak_ptr<MCTSNode> parent; // Cycle completely broken
    std::vector<std::shared_ptr<MCTSNode>> children;
    
    int visit_count = 0;
    float total_value = 0.0f;
    float prior_p = 0.0f;
    int action_taken = -1; // -1 for root node
    
    GameState state; 

    // Helper method to cleanly check expansion status
    bool is_expanded() const {
        return !children.empty();
    }
};

class MCTSTree {
public:
    // Constructor
    MCTSTree(const GameState& root_state, bool add_noise = true);

    // Resumable Request/Submit Interface
    std::optional<std::array<float, 486>> request_leaf();
    void submit_result(const std::vector<float>& policy, float value);
    
    // Evaluation & Advancement
    bool is_done(int n_simulations) const;
    std::vector<float> root_policy(float temperature) const;
    void advance(int action);

private:
    std::shared_ptr<MCTSNode> root;
    std::shared_ptr<MCTSNode> pending_leaf;
    bool add_noise_flag;

    // Standard AlphaZero hyperparameters
    static constexpr float DIRICHLET_ALPHA = 0.3f; 
    static constexpr float DIRICHLET_EPSILON = 0.25f; 
    
    // Internal PRNG for tree-specific randomness (noise)
    std::mt19937 rng;

    // Internal Helpers
    std::vector<float> generate_dirichlet(int num_valid_actions);
    float calculate_puct(std::shared_ptr<MCTSNode> parent, std::shared_ptr<MCTSNode> child) const;
};