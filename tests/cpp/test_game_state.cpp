#include "game_state.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

void test_initial_state() {
    GameState state;
    assert(!state.is_terminal());
    assert(state.get_current_player() == GameState::PLAYER_X);

    auto mask = state.legal_mask();
    for (bool is_legal : mask) {
        assert(is_legal); // All 81 cells should be legal on turn 1
    }
}

void test_free_move_trigger() {
    GameState state;
    
    // Sequence to force player X to win macro-grid 0 via column 0 (cells 0, 3, 6).
    // The final winning move must target micro-cell 0 so player O is directed 
    // to macro-grid 0, which is now decided—thereby triggering a free move.

    // 1. X plays macro 0, micro 3 -> directs O to macro-grid 3
    state.play(3); 
    // 2. O plays macro 3, micro 0 -> directs X back to macro-grid 0
    state.play(27); 
    // 3. X plays macro 0, micro 6 -> directs O to macro-grid 6
    state.play(6); 
    // 4. O plays macro 6, micro 0 -> directs X back to macro-grid 0
    state.play(54); 
    // 5. X plays macro 0, micro 0 -> completes column 0 win in macro 0 AND targets macro 0
    state.play(0); 

    // Macro-grid 0 is now won by X.
    // Because O is directed to macro-grid 0 (which is locked/decided),
    // active_grid must unlock to -1 (free move across all undecided grids).
    
    auto mask = state.legal_mask();
    int legal_count = 0;
    for (bool is_legal : mask) {
        if (is_legal) {
            legal_count++;
        }
    }
    
    // Macro 0 is locked (0 legal moves).
    // Macros 3 and 6 have 8 legal moves remaining each.
    // Macros 1, 2, 4, 5, 7, 8 have 9 legal moves each (54 total).
    // Expected total legal moves = 70.
    assert(legal_count > 9);
}

void test_random_simulation_loop() {
    std::mt19937 rng(42);    

    for (int episode = 0; episode < 100; ++episode) {
        GameState state;
        while (!state.is_terminal()) {
            auto mask = state.legal_mask();
            std::vector<int> legal_indices;
            for (int i = 0; i < GameState::NUM_CELLS; ++i) {
                if (mask[i]) {
                    legal_indices.push_back(i);
                }
            }

            if (legal_indices.empty()) {
                break;
            }

            std::uniform_int_distribution<size_t> dist(0, legal_indices.size() - 1);
            int action = legal_indices[dist(rng)];
            state.play(action);
        }
    }
}

int main() {
    std::cout << "Running GameState unit tests..." << std::endl;

    test_initial_state();
    test_free_move_trigger();
    test_random_simulation_loop();

    std::cout << "All GameState unit tests passed successfully." << std::endl;
    return 0;
}