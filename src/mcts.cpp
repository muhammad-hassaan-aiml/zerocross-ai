#include "mcts.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

MCTSTree::MCTSTree(const GameState& root_state, bool add_noise) 
    : add_noise_flag(add_noise), rng(42) {
    
    root = std::make_shared<MCTSNode>();
    root->state = root_state;
    pending_leaf = nullptr;
}

std::vector<float> MCTSTree::generate_dirichlet(int num_valid_actions) {
    std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);
    std::vector<float> noise(num_valid_actions);
    float sum = 0.0f;
    
    for (int i = 0; i < num_valid_actions; ++i) {
        noise[i] = gamma(rng);
        sum += noise[i];
    }
    
    // Prevent division by zero in the rare case gamma outputs all zeros
    if (sum < 1e-6f) sum = 1e-6f;
    
    for (int i = 0; i < num_valid_actions; ++i) {
        noise[i] /= sum;
    }
    
    return noise;
}

float MCTSTree::calculate_puct(std::shared_ptr<MCTSNode> parent, std::shared_ptr<MCTSNode> child) const {
    // Q is the average value of the child node. If unvisited, it defaults to 0.
    float q_value = (child->visit_count > 0) ? (child->total_value / child->visit_count) : 0.0f;
    
    // DeepMind's Dynamic Exploration Constant (replaces static C_PUCT)
    float c_puct = 1.25f + std::log((parent->visit_count + 19652.0f + 1.0f) / 19652.0f);
    
    // U is the exploration term based on the parent's visit count and the child's prior probability
    float u_value = c_puct * child->prior_p * std::sqrt(static_cast<float>(parent->visit_count)) / (1.0f + child->visit_count);
    
    return q_value + u_value;
}

std::optional<std::array<float, 486>> MCTSTree::request_leaf() {
    // Strict invariant check: Only one outstanding request at a time
    if (pending_leaf != nullptr) {
        return pending_leaf->state.encode();
    }

    std::shared_ptr<MCTSNode> current = root;

    // Traverse the tree using PUCT until an unexpanded leaf is found
    while (current->is_expanded()) {
        float best_puct = -1e9f;
        std::shared_ptr<MCTSNode> best_child = nullptr;

        for (const auto& child : current->children) {
            float puct = calculate_puct(current, child);
            if (puct > best_puct) {
                best_puct = puct;
                best_child = child;
            }
        }
        current = best_child;
    }

    // Handle terminal states instantly without neural network evaluation
    if (current->state.is_terminal()) {
        // In a zero-sum game, if a state is terminal, the player whose turn it is has already lost or drawn.
        // We backpropagate the exact game outcome.
        float terminal_value = 0.0f; 
        
        // Use a dummy game state to check the macro board win condition cleanly
        GameState dummy; 
        
        // Turn parity inversion backpropagation
        std::shared_ptr<MCTSNode> backprop_node = current;
        int current_player = current->state.get_current_player();
        
        while (backprop_node != nullptr) {
            backprop_node->visit_count++;
            
            // If the backprop node's turn matches the player who faces the terminal value, add it.
            // Otherwise, subtract it (zero-sum perspective).
            if (backprop_node->state.get_current_player() == current_player) {
                backprop_node->total_value += terminal_value; // Draws remain 0
            } else {
                backprop_node->total_value -= terminal_value; 
            }
            backprop_node = backprop_node->parent.lock();
        }
        
        // Return nullopt to signal the caller to request again, as no NN eval is needed
        return std::nullopt; 
    }

    // Valid, non-terminal leaf found. Mark as pending and yield state for evaluation.
    pending_leaf = current;
    return pending_leaf->state.encode();
}

void MCTSTree::submit_result(const std::vector<float>& policy, float value) {
    if (pending_leaf == nullptr) {
        throw std::runtime_error("submit_result called but no leaf is pending.");
    }

    auto mask = pending_leaf->state.legal_mask();
    std::vector<int> legal_actions;
    float policy_sum = 0.0f;

    for (int i = 0; i < GameState::NUM_CELLS; ++i) {
        if (mask[i]) {
            legal_actions.push_back(i);
            policy_sum += std::exp(policy[i]); // Softmax extraction for valid moves
        }
    }

    bool is_root = (pending_leaf == root);
    std::vector<float> noise;
    
    if (is_root && add_noise_flag) {
        noise = generate_dirichlet(legal_actions.size());
    }

    // Expand the pending leaf
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        int action = legal_actions[i];
        float prior = std::exp(policy[action]) / policy_sum;
        
        if (is_root && add_noise_flag) {
            prior = (1.0f - DIRICHLET_EPSILON) * prior + DIRICHLET_EPSILON * noise[i];
        }

        auto child = std::make_shared<MCTSNode>();
        child->parent = pending_leaf; // Assigns to weak_ptr safely
        child->prior_p = prior;
        child->action_taken = action;
        
        // Clone state and apply the move
        child->state = pending_leaf->state;
        child->state.play(action);

        pending_leaf->children.push_back(child);
    }

    // Backpropagate the value up the tree
    std::shared_ptr<MCTSNode> backprop_node = pending_leaf;
    int evaluated_player = pending_leaf->state.get_current_player();

    while (backprop_node != nullptr) {
        backprop_node->visit_count++;
        
        // Zero-sum perspective flip
        if (backprop_node->state.get_current_player() == evaluated_player) {
            backprop_node->total_value += value;
        } else {
            backprop_node->total_value -= value;
        }
        
        backprop_node = backprop_node->parent.lock(); // Elevate weak_ptr to shared_ptr for traversal
    }

    // Clear the pending status to allow the next request
    pending_leaf = nullptr;
}

bool MCTSTree::is_done(int n_simulations) const {
    return root->visit_count >= n_simulations;
}

std::vector<float> MCTSTree::root_policy(float temperature) const {
    std::vector<float> policy(GameState::NUM_CELLS, 0.0f);
    
    if (root->children.empty()) {
        return policy;
    }

    if (temperature < 1e-3f) {
        // Temperature approaches 0: deterministic play (argmax)
        int best_action = -1;
        int max_visits = -1;
        
        for (const auto& child : root->children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_action = child->action_taken;
            }
        }
        if (best_action != -1) {
            policy[best_action] = 1.0f;
        }
    } else {
        // Temperature-scaled exploration
        float sum = 0.0f;
        std::vector<float> adjusted_visits(root->children.size());
        
        for (size_t i = 0; i < root->children.size(); ++i) {
            adjusted_visits[i] = std::pow(static_cast<float>(root->children[i]->visit_count), 1.0f / temperature);
            sum += adjusted_visits[i];
        }
        
        for (size_t i = 0; i < root->children.size(); ++i) {
            policy[root->children[i]->action_taken] = adjusted_visits[i] / sum;
        }
    }

    return policy;
}

void MCTSTree::advance(int action) {
    std::shared_ptr<MCTSNode> next_root = nullptr;
    
    for (const auto& child : root->children) {
        if (child->action_taken == action) {
            next_root = child;
            break;
        }
    }

    if (next_root != nullptr) {
        // Shift root down the selected path. 
        // The old root's reference count drops, and unselected subtrees are automatically destroyed.
        root = next_root;
        root->parent.reset(); // The new root has no parent
    } else {
        // The requested action is outside the explored tree (e.g., unexpected opponent move).
        // Create a new root from scratch to recover safely.
        GameState next_state = root->state;
        next_state.play(action);
        
        root = std::make_shared<MCTSNode>();
        root->state = next_state;
    }
    
    pending_leaf = nullptr;
}