#include "game_state.h"
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>

int main() {
    std::cout << "Starting 1,000,000 random games stress test..." << std::endl;

    std::mt19937 rng(42); 
    int total_games = 1000000;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total_games; ++i) {
        GameState state;
        
        while (!state.is_terminal()) {
            // Correctly use std::array to match your optimized C++ engine
            std::array<bool, 81> mask = state.legal_mask();
            std::vector<int> legal_moves;
            
            for (int m = 0; m < 81; ++m) {
                if (mask[m]) {
                    legal_moves.push_back(m);
                }
            }
            
            if (legal_moves.empty()) {
                std::cerr << "FATAL: Non-terminal state has 0 legal moves!" << std::endl;
                return 1;
            }
            
            std::uniform_int_distribution<int> dist(0, legal_moves.size() - 1);
            int chosen_move = legal_moves[dist(rng)];
            
            state.play(chosen_move);
        }
        
        int winner = state.get_winner();
        if (winner < -1 || winner > 1) {
            std::cerr << "FATAL: Invalid winner detected: " << winner << std::endl;
            return 1;
        }

        if ((i + 1) % 100000 == 0) {
            std::cout << "Completed " << (i + 1) << " games..." << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "SUCCESS: 1,000,000 games completed without crashes or memory leaks." << std::endl;
    std::cout << "Time elapsed: " << diff.count() << " seconds." << std::endl;
    std::cout << "Games per second: " << (total_games / diff.count()) << std::endl;

    return 0;
}