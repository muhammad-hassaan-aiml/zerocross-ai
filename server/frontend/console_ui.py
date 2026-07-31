import requests
import sys

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: '.', 2: '#'}
    print("\n" + "="*23)
    for i in range(9):
        row = ""
        for j in range(9):
            macro = (i // 3) * 3 + (j // 3)
            micro = (i % 3) * 3 + (j % 3)
            idx = macro * 9 + micro
            row += symbols[board[idx]] + " "
            if j % 3 == 2 and j != 8:
                row += "| "
        print(row)
        if i % 3 == 2 and i != 8:
            print("-" * 21)
    print("="*23 + "\n")

def main():
    print("Welcome to the ZeroCross AI Arena!")
    board = [0] * 81
    active_grid = -1
    
    while True:
        print_board(board)
        print(f"Active Grid: {active_grid if active_grid != -1 else 'ANY (Free Move)'}")
        
        # Human Turn
        # Human Turn
        try:
            move = int(input("Enter your move (0-80): "))
            macro_target = move // 9  # Calculates which of the 9 macro-grids you clicked in
            
            if move < 0 or move > 80 or board[move] != 0:
                print("Invalid move: Cell is taken or out of bounds. Try again.")
                continue
                
            # RULE CHECK: Are you playing in the forced macro-grid?
            if active_grid != -1 and macro_target != active_grid:
                print(f"ILLEGAL MOVE: You MUST play in macro-grid {active_grid}!")
                continue
                
            board[move] = 1  # Human is X
            active_grid = move % 9
        except ValueError:
            continue

        print("\nAI is thinking...")
        # AI Turn
        try:
            res = requests.post("http://127.0.0.1:8000/move", json={
                "board": board,
                "active_grid": active_grid,
                "simulations": 200
            })
            
            if res.status_code == 200:
                ai_move = res.json()["move"]
                print(f"AI plays: {ai_move}")
                board[ai_move] = -1  # AI is O
                active_grid = ai_move % 9
            else:
                print("Server Error:", res.text)
        except requests.exceptions.ConnectionError:
            print("Could not connect. Make sure FastAPI (uvicorn server/main:app) is running!")
            sys.exit(1)

if __name__ == "__main__":
    main()