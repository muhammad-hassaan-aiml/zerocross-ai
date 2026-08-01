import requests
import sys

URL = "http://127.0.0.1:8000/move"

def fetch_move(board, active_grid):
    payload = {"board": board, "active_grid": active_grid, "simulations": 10}
    try:
        res = requests.post(URL, json=payload)
        if res.status_code == 200:
            return res.status_code, res.json()
        return res.status_code, res.text
    except Exception as e:
        return 500, str(e)

try:
    board_1 = [0] * 81
    status_1, data_1 = fetch_move(board_1, 4)
    move_1 = data_1.get("move", -1) if isinstance(data_1, dict) else -1
    assert status_1 == 200 and 36 <= move_1 <= 44, "Failed: Macro-grid confinement"

    board_2 = [0] * 81
    board_2[40] = 1
    board_2[0] = -1
    status_2, data_2 = fetch_move(board_2, 4)
    move_2 = data_2.get("move", -1) if isinstance(data_2, dict) else -1
    assert status_2 == 200 and move_2 != 40 and 36 <= move_2 <= 44, "Failed: Played on occupied cell"

    board_3 = [0] * 81
    board_3[0] = board_3[1] = board_3[2] = 1
    board_3[9] = board_3[10] = board_3[18] = -1
    status_3, data_3 = fetch_move(board_3, -1)
    move_3 = data_3.get("move", -1) if isinstance(data_3, dict) else -1
    assert status_3 == 200 and not (0 <= move_3 <= 8), "Failed: Free move played in won grid"

    board_4 = [0] * 81
    board_4[0] = board_4[1] = board_4[2] = 1 
    status_4, data_4 = fetch_move(board_4, 4)
    assert status_4 == 400 and "parity" in str(data_4).lower(), "Failed: Turn parity validation"

    res_5 = requests.post(URL, json={"board": [0] * 80, "active_grid": 4, "simulations": 10})
    assert res_5.status_code in [400, 422], "Failed: Array length validation"

    print("SUCCESS! ZeroCross C++ Backend and API are mathematically bulletproof.")
    sys.exit(0)

except AssertionError as e:
    print(f"WARNING: Engine violations detected. {e}")
    sys.exit(1)