#include <emscripten/bind.h>
#include <vector>
#include <array>
#include <cstdlib>
#include <cstring>
#include "game_state.h"
#include "mcts.h"

using namespace emscripten;

// --- Memory Management Helpers ---
uintptr_t allocate_memory(size_t bytes) {
    return reinterpret_cast<uintptr_t>(std::malloc(bytes));
}

void free_memory(uintptr_t ptr) {
    std::free(reinterpret_cast<void*>(ptr));
}

// --- GameState Helpers ---
GameState create_state(uintptr_t board_ptr, int active_grid) {
    const int* b_data = reinterpret_cast<const int*>(board_ptr);
    std::vector<int> board(b_data, b_data + 81);
    return GameState::from_array(board, active_grid);
}

std::array<float, 486> g_encode_buffer;
uintptr_t encode_state(const GameState& state) {
    g_encode_buffer = state.encode();
    return reinterpret_cast<uintptr_t>(g_encode_buffer.data());
}

// --- MCTS Helpers ---
std::vector<float> g_leaves_buffer;
uintptr_t request_leaves(MCTSTree& tree, int batch_size) {
    auto leaves = tree.request_leaves(batch_size);
    size_t num_leaves = leaves.size();
    
    g_leaves_buffer.resize(num_leaves * 486);
    for (size_t i = 0; i < num_leaves; ++i) {
        std::memcpy(g_leaves_buffer.data() + (i * 486), leaves[i].data(), 486 * sizeof(float));
    }
    
    return reinterpret_cast<uintptr_t>(g_leaves_buffer.data());
}

int get_leaves_count() {
    return g_leaves_buffer.size() / 486;
}

void submit_results(MCTSTree& tree, uintptr_t policies_ptr, uintptr_t values_ptr, int num_leaves) {
    const float* p_data = reinterpret_cast<const float*>(policies_ptr);
    const float* v_data = reinterpret_cast<const float*>(values_ptr);

    std::vector<std::vector<float>> policies(num_leaves, std::vector<float>(81));
    std::vector<float> values(num_leaves);
    
    for (int i = 0; i < num_leaves; ++i) {
        std::memcpy(policies[i].data(), p_data + (i * 81), 81 * sizeof(float));
        values[i] = v_data[i];
    }
    
    tree.submit_results(policies, values);
}

std::vector<float> g_policy_buffer;
uintptr_t get_root_policy(MCTSTree& tree, float temp) {
    g_policy_buffer = tree.root_policy(temp);
    return reinterpret_cast<uintptr_t>(g_policy_buffer.data());
}

EMSCRIPTEN_BINDINGS(zerocross_module) {
    class_<GameState>("GameState")
        .constructor<>()
        .function("is_terminal", &GameState::is_terminal);

    class_<MCTSTree>("MCTSTree")
        .constructor<const GameState&, bool>()
        .function("is_done", &MCTSTree::is_done);

    function("create_state", &create_state);
    function("encode_state", &encode_state);
    
    function("request_leaves", &request_leaves);
    function("get_leaves_count", &get_leaves_count);
    function("submit_results", &submit_results);
    function("get_root_policy", &get_root_policy);
    
    function("allocate_memory", &allocate_memory);
    function("free_memory", &free_memory);
}