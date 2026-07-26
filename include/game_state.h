#pragma once

#include <array>
#include <vector>

class GameState {
public:
    // Core board dimensions for Ultimate Tic-Tac-Toe
    static constexpr int NUM_CELLS = 81;
    static constexpr int NUM_MACRO = 9;

    // State representations
    static constexpr int PLAYER_X = 1;
    static constexpr int PLAYER_O = -1;
    static constexpr int EMPTY = 0;
    static constexpr int DRAW = 2; // Distinct value for a locked/drawn micro-board

    // Default constructor initializes an empty board, X to move, any grid active
    GameState();

    // Returns the flat 486-element (6x9x9 channels) tensor representation
    std::array<float, 486> encode() const;

    // Returns an 81-element boolean mask of valid moves for the current turn
    std::array<bool, NUM_CELLS> legal_mask() const;

    // Applies a move using canonical indexing: action = (macro_index * 9) + micro_index
    void play(int action);

    // Evaluates if the global macro-board has reached a terminal state (win or fully drawn)
    bool is_terminal() const;

    // Reconstructs an exact state from a flat Python/NumPy array representation
    static GameState from_array(const std::vector<int>& board, int active_grid);

    // Accessor for the current player, required for neural network value targeting
    int get_current_player() const { return current_player; }

private:
    std::array<int, NUM_CELLS> micro_board; // 81 individual cells
    std::array<int, NUM_MACRO> macro_board; // 9 macro grids status
    
    int current_player; 
    
    // Tracks the currently active macro-grid (0-8). 
    // A value of -1 designates a free move across the entire board.
    int active_grid; 

    // Internal helper functions to process game rules
    int check_win(const std::array<int, 9>& grid) const;
    std::array<int, 9> get_micro_grid(int macro_index) const;
};