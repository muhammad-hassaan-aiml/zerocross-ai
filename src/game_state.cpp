#include "game_state.h"
#include <algorithm>
#include <stdexcept>

GameState::GameState() 
    : current_player(PLAYER_X), active_grid(-1) { 
    micro_board.fill(EMPTY); 
    macro_board.fill(EMPTY);
}

std::array<int, 9> GameState::get_micro_grid(int macro_index) const {
    std::array<int, 9> grid;
    int start_idx = macro_index * 9;
    for (int i = 0; i < 9; ++i) {
        grid[i] = micro_board[start_idx + i];
    }
    return grid;
}

int GameState::check_win(const std::array<int, 9>& grid) const {
    // 8 winning line combinations on a 3x3 grid
    static constexpr int WIN_PATTERNS[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Columns
        {0, 4, 8}, {2, 4, 6}             // Diagonals
    };

    for (const auto& pattern : WIN_PATTERNS) {
        if (grid[pattern[0]] != EMPTY &&
            grid[pattern[0]] != DRAW &&
            grid[pattern[0]] == grid[pattern[1]] &&
            grid[pattern[1]] == grid[pattern[2]]) {
            return grid[pattern[0]];
        }
    }

    // Evaluate for micro-grid draw condition
    bool full = true;
    for (int cell : grid) {
        if (cell == EMPTY) {
            full = false;
            break;
        }
    }

    return full ? DRAW : EMPTY;
}

void GameState::play(int action) {
    if (action < 0 || action >= NUM_CELLS) {
        return;
    }

    // Occupancy guard: prevent overwriting a non-empty cell
    if (micro_board[action] != EMPTY) {
        return;
    }

    int macro_idx = action / 9;
    int micro_idx = action % 9;

    // Apply move to the designated micro-cell
    micro_board[action] = current_player;

    // Update macro-board state if the sub-grid was undecided
    if (macro_board[macro_idx] == EMPTY) {
        std::array<int, 9> current_micro = get_micro_grid(macro_idx);
        macro_board[macro_idx] = check_win(current_micro);
    }

    // Determine target macro-grid for the next player
    if (macro_board[micro_idx] != EMPTY) {
        // Free-move rule: Target sub-grid is already decided (won or drawn)
        active_grid = -1;
    } else {
        active_grid = micro_idx;
    }

    // Swap active player turn
    current_player = (current_player == PLAYER_X) ? PLAYER_O : PLAYER_X;
}

bool GameState::is_terminal() const {
    // Game ends if the global macro-board has a winner
    if (check_win(macro_board) != EMPTY) {
        return true;
    }

    // Game ends if no valid moves remain anywhere on the board
    auto mask = legal_mask();
    for (bool is_legal : mask) {
        if (is_legal) {
            return false;
        }
    }

    return true;
}

int GameState::get_winner() const {
    int status = check_win(macro_board);
    
    // If PLAYER_X or PLAYER_O won the macro board, return their value
    if (status == PLAYER_X || status == PLAYER_O) {
        return status;
    }
    
    // If EMPTY or DRAW, return 0 (no one gets a win reward)
    return 0;
}

std::array<bool, GameState::NUM_CELLS> GameState::legal_mask() const {
    std::array<bool, NUM_CELLS> mask;
    mask.fill(false);

    // If global macro-board is already decided, mask remains all false
    if (check_win(macro_board) != EMPTY) {
        return mask;
    }

    for (int i = 0; i < NUM_CELLS; ++i) {
        int macro_idx = i / 9;

        // Disallow moves in already-decided macro-grids
        if (macro_board[macro_idx] != EMPTY) {
            continue;
        }

        // Disallow moves on non-empty micro-cells
        if (micro_board[i] != EMPTY) {
            continue;
        }

        // Validate active macro-grid constraints
        if (active_grid == -1 || active_grid == macro_idx) {
            mask[i] = true;
        }
    }

    return mask;
}

std::array<float, 486> GameState::encode() const {
    std::array<float, 486> tensor;
    tensor.fill(0.0f);

    int opponent = (current_player == PLAYER_X) ? PLAYER_O : PLAYER_X;

    for (int i = 0; i < NUM_CELLS; ++i) {
        int macro_idx = i / 9;

        // Channel 0: Current player stones
        if (micro_board[i] == current_player) {
            tensor[0 * NUM_CELLS + i] = 1.0f;
        }
        // Channel 1: Opponent stones
        else if (micro_board[i] == opponent) {
            tensor[1 * NUM_CELLS + i] = 1.0f;
        }

        // Channel 2: Active-grid mask
        if (active_grid == -1) {
            if (macro_board[macro_idx] == EMPTY) {
                tensor[2 * NUM_CELLS + i] = 1.0f;
            }
        } else if (macro_idx == active_grid) {
            tensor[2 * NUM_CELLS + i] = 1.0f;
        }

        // Channel 3: Macro-grids won by current player
        if (macro_board[macro_idx] == current_player) {
            tensor[3 * NUM_CELLS + i] = 1.0f;
        }
        // Channel 4: Macro-grids won by opponent
        else if (macro_board[macro_idx] == opponent) {
            tensor[4 * NUM_CELLS + i] = 1.0f;
        }
        // Channel 5: Drawn macro-grids
        else if (macro_board[macro_idx] == DRAW) {
            tensor[5 * NUM_CELLS + i] = 1.0f;
        }
    }

    return tensor;
}

GameState GameState::from_array(const std::vector<int>& board, int active_grid) {
    GameState state;

    if (board.size() != NUM_CELLS) {
        return state;
    }

    // Populate internal micro-board state
    for (int i = 0; i < NUM_CELLS; ++i) {
        state.micro_board[i] = board[i];
    }

    state.active_grid = active_grid;

    // Reconstruct macro-board statuses and count pieces to infer active player turn
    int x_count = 0;
    int o_count = 0;

    for (int m = 0; m < NUM_MACRO; ++m) {
        std::array<int, 9> micro = state.get_micro_grid(m);
        state.macro_board[m] = state.check_win(micro);

        for (int cell : micro) {
            if (cell == PLAYER_X) {
                x_count++;
            } else if (cell == PLAYER_O) {
                o_count++;
            }
        }
    }

    // Parity validation: O can never have more pieces than X, 
    // and X can never be more than 1 piece ahead of O.
    if (o_count > x_count || x_count > o_count + 1) {
        throw std::invalid_argument("Invalid board state: turn parity mismatch.");
    }

    // Turn parity inference
    state.current_player = (x_count > o_count) ? PLAYER_O : PLAYER_X;

    return state;
}