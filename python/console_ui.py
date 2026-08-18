import requests
import sys
import random

# ANSI Color Codes for terminal UI
C_X = '\033[91m'  # Red
C_O = '\033[94m'  # Blue
C_W = '\033[93m'  # Yellow/Gold
C_D = '\033[90m'  # Dark Gray (Draw)
C_R = '\033[0m'   # Reset

def check_macro_win(macro_grid):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for p in win_patterns:
        if macro_grid[p[0]] != 0 and macro_grid[p[0]] != 2 and \
           macro_grid[p[0]] == macro_grid[p[1]] and \
           macro_grid[p[1]] == macro_grid[p[2]]:
            return macro_grid[p[0]]
    if all(cell != 0 for cell in macro_grid):
        return 2 # DRAW
    return 0 # EMPTY

def print_board(board, macro_status, active_grid):
    symbols = {1: f'{C_X}X{C_R}', -1: f'{C_O}O{C_R}', 0: '.', 2: f'{C_D}#{C_R}'}
    print(f"\n{C_W}=================================================={C_R}")
    print(f" {C_W}Ultimate Tic-Tac-Toe Arena (1-9 Indexing){C_R}")
    print(f"{C_W}=================================================={C_R}")
    
    # Print reference guide for macro grids on the side
    print(f" Macro Layout:        Live Board State:")
    for i in range(9):
        # Pad exactly 18 spaces to align with the 18-character macro guide string
        # Added +1 to display 1-9 instead of 0-8
        row = f"  {i//3*3+1} | {i//3*3+2} | {i//3*3+3}  -->  " if i in [1, 4, 7] else "                  "
        
        # Build the actual board row
        board_row = ""
        for j in range(9):
            macro = (i // 3) * 3 + (j // 3)
            micro = (i % 3) * 3 + (j % 3)
            idx = macro * 9 + micro
            
            # Highlight active grid cells slightly or show symbols
            if macro_status[macro] == 1:
                board_row += f"{C_X}X{C_R} "
            elif macro_status[macro] == -1:
                board_row += f"{C_O}O{C_R} "
            elif macro_status[macro] == 2:
                board_row += f"{C_D}#{C_R} "
            else:
                board_row += symbols[board[idx]] + " "
                
            if j % 3 == 2 and j != 8:
                board_row += "| "
                
        print(row + board_row)
        if i % 3 == 2 and i != 8:
            print("                  -------------------------")
    print(f"{C_W}=================================================={C_R}\n")

def print_cell_helper():
    print(f"{C_D}[Cell Index Reference inside any 3x3 Grid]:")
    print(" 1 (Top-Left)     | 2 (Top-Mid)     | 3 (Top-Right)")
    print(" 4 (Center-Left)  | 5 (Center)      | 6 (Center-Right)")
    print(f" 7 (Bottom-Left)  | 8 (Bottom-Mid)  | 9 (Bottom-Right){C_R}\n")

def main():
    print("\nConnecting to AI Engine...")
    try:
        # Simple ping to verify the server is running
        res = requests.get("http://127.0.0.1:8000/")
        if res.status_code == 200:
            print("Successfully connected to the backend!")
        else:
            print("Failed to reach server API.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Could not connect. Make sure FastAPI (uvicorn server.main:app) is running!")
        sys.exit(1)

    print(f"\n{C_W}Select Engine Difficulty:{C_R}")
    print("  1. Easy (50 simulations)")
    print("  2. Medium (200 simulations)")
    print("  3. Hard (800 simulations)")
    
    diff_choice = 0
    while diff_choice not in [1, 2, 3]:
        try:
            diff_choice = int(input("\nSelect difficulty (1-3): "))
        except ValueError:
            pass
            
    sims_map = {1: 50, 2: 200, 3: 800}
    simulations = sims_map[diff_choice]

    side = ""
    while side not in ['X', 'O', 'R']:
        side = input("\nPlay as X (First), O (Second), or Random? [X/O/R]: ").strip().upper()
    
    if side == 'R':
        side = random.choice(['X', 'O'])
        print(f"\n{C_W}Coin toss result: You are playing as {side}!{C_R}")

    human_p = -1 if side == 'O' else 1
    ai_p = 1 if human_p == -1 else -1

    board = [0] * 81
    active_grid = -1
    current_turn = 1 
    
    print_cell_helper()

    while True:
        macro_status = [0] * 9
        for m in range(9):
            macro_grid = [board[m * 9 + i] for i in range(9)]
            macro_status[m] = check_macro_win(macro_grid)

        global_win = check_macro_win(macro_status)
        if global_win != 0:
            print_board(board, macro_status, active_grid)
            if global_win == human_p:
                print(f"{C_W}🏆 GAME OVER! YOU WIN! 🏆{C_R}")
            elif global_win == ai_p:
                print(f"{C_X}💀 GAME OVER! AI WINS! 💀{C_R}")
            else:
                print(f"{C_D}🤝 GAME OVER! IT'S A DRAW! 🤝{C_R}")
            break

        print_board(board, macro_status, active_grid)
        
        grid_msg = f"Macro Grid {active_grid + 1}" if active_grid != -1 else "ANY (Free Move - Choose any open Macro Grid 1-9)"
        print(f"Target Constraint: {C_W}{grid_msg}{C_R}")
        
        if current_turn == human_p:
            try:
                raw_input = input(f"Your Move [Format: MacroCell (e.g. 55 or 5 5)]: ").strip()
                if not raw_input:
                    continue
                
                cleaned_input = raw_input.replace(" ", "")
                
                if len(cleaned_input) != 2 or not cleaned_input.isdigit():
                    print(f"{C_X}Error: Please provide exactly two digits representing Macro and Micro cell (e.g., '71' or '7 1'){C_R}")
                    continue
                    
                macro = int(cleaned_input[0]) - 1
                micro = int(cleaned_input[1]) - 1
                
                if not (0 <= macro <= 8 and 0 <= micro <= 8):
                    print(f"{C_X}Error: Numbers must be between 1 and 9.{C_R}")
                    continue
                    
                move = (macro * 9) + micro 
                
                if board[move] != 0:
                    print(f"{C_X}Error: Cell [{macro + 1}, {micro + 1}] is already taken!{C_R}")
                    continue
                    
                if macro_status[macro] != 0:
                    print(f"{C_X}Error: Macro-grid {macro + 1} is already decided/closed!{C_R}")
                    continue
                    
                if active_grid != -1 and macro != active_grid:
                    print(f"{C_X}ILLEGAL: You are forced to play inside Macro-grid {active_grid + 1}!{C_R}")
                    continue
                    
                print(f"{C_D}--> Confirmed: Placed in Macro {macro + 1}, Micro Cell {micro + 1}{C_R}")
                
                board[move] = human_p 
                
                target_grid_cells = [board[micro * 9 + i] for i in range(9)]
                if check_macro_win(target_grid_cells) != 0:
                    active_grid = -1 
                else:
                    active_grid = micro 
                    
                current_turn = ai_p 
                
            except ValueError:
                print(f"{C_X}Invalid input.{C_R}")
                continue

        else:
            print(f"\nAI is thinking ({simulations} simulations)...")
            try:
                res = requests.post("http://127.0.0.1:8000/move", json={
                    "board": board,
                    "active_grid": active_grid,
                    "simulations": simulations
                })
                
                if res.status_code == 200:
                    ai_move = res.json()["move"]
                    if ai_move == -1:
                        break 
                        
                    ai_macro = ai_move // 9
                    ai_micro = ai_move % 9
                    
                    print(f"AI plays: Macro {ai_macro + 1}, Micro Cell {ai_micro + 1}")
                    
                    board[ai_move] = ai_p  
                    
                    target_grid_cells = [board[ai_micro * 9 + i] for i in range(9)]
                    if check_macro_win(target_grid_cells) != 0:
                        active_grid = -1 
                    else:
                        active_grid = ai_micro 
                        
                    current_turn = human_p 
                else:
                    print("Server Error:", res.text)
                    sys.exit(1)
            except requests.exceptions.ConnectionError:
                print("Could not connect to FastAPI server!")
                sys.exit(1)

if __name__ == "__main__":
    main()