#include "mcts.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <mutex>

MCTSTree::MCTSTree(const GameState& root_state, bool add_noise) 
    : add_noise_flag(add_noise), rng(std::random_device{}()) {
    
    root = std::make_shared<MCTSNode>();
    root->state = root_state;
    pending_leaves.clear();
}

std::vector<float> MCTSTree::generate_dirichlet(int num_valid_actions) {
    std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);
    std::vector<float> noise(num_valid_actions);
    float sum = 0.0f;
    
    for (int i = 0; i < num_valid_actions; ++i) {
        noise[i] = gamma(rng);
        sum += noise[i];
    }
    
    if (sum < 1e-6f) sum = 1e-6f;
    
    for (int i = 0; i < num_valid_actions; ++i) {
        noise[i] /= sum;
    }
    
    return noise;
}

float MCTSTree::calculate_puct(std::shared_ptr<MCTSNode> parent, std::shared_ptr<MCTSNode> child) const {
    int child_visits = child->visit_count + child->virtual_loss;
    int parent_visits = parent->visit_count + parent->virtual_loss;

    float q_value = (child_visits > 0) ? -(child->total_value + child->virtual_loss) / child_visits : 0.0f;
    
    float c_puct = 1.25f + std::log((parent_visits + 19652.0f + 1.0f) / 19652.0f);
    float u_value = c_puct * child->prior_p * std::sqrt(static_cast<float>(parent_visits)) / (1.0f + child_visits);
    
    return q_value + u_value;
}

std::vector<std::array<float, 486>> MCTSTree::request_leaves(int batch_size) {
    std::vector<std::array<float, 486>> encoded_states;
    if (!pending_leaves.empty()) return encoded_states; 

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<MCTSNode> current = root;
        std::vector<std::shared_ptr<MCTSNode>> search_path;
        
        while (current->is_expanded()) {
            search_path.push_back(current);
            float best_puct = -1e9f;
            std::shared_ptr<MCTSNode> best_child = current->children[0]; 

            for (const auto& child : current->children) {
                float puct = calculate_puct(current, child);
                if (puct > best_puct && !std::isnan(puct)) {
                    best_puct = puct;
                    best_child = child;
                }
            }
            current = best_child;
        }
        search_path.push_back(current);

        if (current->state.is_terminal()) {
            int winner = current->state.get_winner();
            int current_player = current->state.get_current_player();
            
            float terminal_value = 0.0f;
            if (winner != 0) { 
                terminal_value = (winner == current_player) ? 1.0f : -1.0f;
            }
            
            for (auto& node : search_path) {
                node->visit_count++;
                if (node->state.get_current_player() == current_player) {
                    node->total_value += terminal_value; 
                } else {
                    node->total_value -= terminal_value; 
                }
            }
            continue; 
        }

        for (auto& node : search_path) {
            node->virtual_loss++;
        }
        
        pending_leaves.push_back(current);
        encoded_states.push_back(current->state.encode());
    }

    return encoded_states;
}

void MCTSTree::submit_results(const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
    if (policies.size() != pending_leaves.size() || values.size() != pending_leaves.size()) {
        throw std::runtime_error("submit_results called with mismatched batch sizes.");
    }

    for (size_t b = 0; b < pending_leaves.size(); ++b) {
        auto leaf = pending_leaves[b];
        const auto& policy = policies[b];
        float value = values[b];

        auto mask = leaf->state.legal_mask();
        std::vector<int> legal_actions;
        
        float max_logit = -1e9f;
        for (int i = 0; i < GameState::NUM_CELLS; ++i) {
            if (mask[i] && policy[i] > max_logit) {
                max_logit = policy[i];
            }
        }

        float policy_sum = 0.0f;
        for (int i = 0; i < GameState::NUM_CELLS; ++i) {
            if (mask[i]) {
                legal_actions.push_back(i);
                policy_sum += std::exp(policy[i] - max_logit);
            }
        }

        bool is_root = (leaf == root);
        std::vector<float> noise;
        
        if (is_root && add_noise_flag) {
            noise = generate_dirichlet(legal_actions.size());
        }

        if (!leaf->is_expanded()) {
            for (size_t i = 0; i < legal_actions.size(); ++i) {
                int action = legal_actions[i];
                float prior = std::exp(policy[action] - max_logit) / policy_sum;
                
                if (is_root && add_noise_flag) {
                    prior = (1.0f - DIRICHLET_EPSILON) * prior + DIRICHLET_EPSILON * noise[i];
                }

                auto child = std::make_shared<MCTSNode>();
                child->parent = leaf; 
                child->prior_p = prior;
                child->action_taken = action;
                
                child->state = leaf->state;
                child->state.play(action);

                leaf->children.push_back(child);
            }
        }

        std::shared_ptr<MCTSNode> backprop_node = leaf;
        int evaluated_player = leaf->state.get_current_player();

        while (backprop_node != nullptr) {
            backprop_node->visit_count++;
            backprop_node->virtual_loss--;
            
            if (backprop_node->state.get_current_player() == evaluated_player) {
                backprop_node->total_value += value;
            } else {
                backprop_node->total_value -= value;
            }
            
            backprop_node = backprop_node->parent.lock(); 
        }
    }
    
    pending_leaves.clear();
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
        root = next_root;
        root->parent.reset(); 
    } else {
        GameState next_state = root->state;
        next_state.play(action);
        
        root = std::make_shared<MCTSNode>();
        root->state = next_state;
    }
    
    pending_leaves.clear();
}

ParallelMCTS::ParallelMCTS(int num_games, bool add_noise) 
    : num_games(num_games), add_noise_flag(add_noise) {
    GameState initial_state;
    for (int i = 0; i < num_games; ++i) {
        trees.push_back(std::make_shared<MCTSTree>(initial_state, add_noise));
    }
}

void ParallelMCTS::set_state(int game_idx, const GameState& state) {
    trees[game_idx] = std::make_shared<MCTSTree>(state, add_noise_flag);
}

void ParallelMCTS::advance(int game_idx, int action) {
    trees[game_idx]->advance(action);
}

bool ParallelMCTS::is_done(int game_idx, int n_simulations) const {
    return trees[game_idx]->is_done(n_simulations);
}

std::vector<float> ParallelMCTS::root_policy(int game_idx, float temperature) const {
    return trees[game_idx]->root_policy(temperature);
}

std::vector<std::array<float, 486>> ParallelMCTS::request_batch(int n_simulations, int batch_per_tree) {
    std::vector<std::array<float, 486>> global_leaves;
    leaf_game_mapping.clear();
    
    std::mutex mtx;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_games; ++i) {
        if (!trees[i]->is_done(n_simulations)) {
            threads.emplace_back([this, i, batch_per_tree, &global_leaves, &mtx]() {
                auto leaves = trees[i]->request_leaves(batch_per_tree);
                
                std::lock_guard<std::mutex> lock(mtx);
                for (const auto& leaf : leaves) {
                    global_leaves.push_back(leaf);
                    leaf_game_mapping.push_back(i);
                }
            });
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return global_leaves;
}

void ParallelMCTS::submit_batch(const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
    std::vector<std::vector<std::vector<float>>> tree_policies(num_games);
    std::vector<std::vector<float>> tree_values(num_games);
    
    for (size_t i = 0; i < leaf_game_mapping.size(); ++i) {
        int game_idx = leaf_game_mapping[i];
        tree_policies[game_idx].push_back(policies[i]);
        tree_values[game_idx].push_back(values[i]);
    }
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_games; ++i) {
        if (!tree_policies[i].empty()) {
            threads.emplace_back([this, i, &tree_policies, &tree_values]() {
                trees[i]->submit_results(tree_policies[i], tree_values[i]);
            });
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
}