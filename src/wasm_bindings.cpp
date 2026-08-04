#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <array>
#include <cstring>
#include "game_state.h"
#include "mcts.h"

using namespace emscripten;

GameState create_state(val js_board, int active_grid) {
    std::vector<int> board = vecFromJSArray<int>(js_board);
    return GameState::from_array(board, active_grid);
}

std::array<float, 486> g_encode_buffer;
val encode_state(const GameState& state) {
    g_encode_buffer = state.encode();
    return val(typed_memory_view(486, g_encode_buffer.data()));
}

std::vector<float> g_leaves_buffer;
val request_leaves(MCTSTree& tree, int batch_size) {
    auto leaves = tree.request_leaves(batch_size);
    size_t num_leaves = leaves.size();
    
    g_leaves_buffer.resize(num_leaves * 486);
    for (size_t i = 0; i < num_leaves; ++i) {
        std::memcpy(g_leaves_buffer.data() + (i * 486), leaves[i].data(), 486 * sizeof(float));
    }
    
    return val(typed_memory_view(g_leaves_buffer.size(), g_leaves_buffer.data()));
}

int get_leaves_count() {
    return g_leaves_buffer.size() / 486;
}

void submit_results(MCTSTree& tree, val js_policies, val js_values, int num_leaves) {
    std::vector<float> p_data = vecFromJSArray<float>(js_policies);
    std::vector<float> v_data = vecFromJSArray<float>(js_values);

    std::vector<std::vector<float>> policies(num_leaves, std::vector<float>(81));
    std::vector<float> values(num_leaves);
    
    for (int i = 0; i < num_leaves; ++i) {
        std::memcpy(policies[i].data(), p_data.data() + (i * 81), 81 * sizeof(float));
        values[i] = v_data[i];
    }
    
    tree.submit_results(policies, values);
}

std::vector<float> g_policy_buffer;
val get_root_policy(MCTSTree& tree, float temp) {
    g_policy_buffer = tree.root_policy(temp);
    return val(typed_memory_view(g_policy_buffer.size(), g_policy_buffer.data()));
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
}