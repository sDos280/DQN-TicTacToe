import math

State = list[int, int, int, int, int, int, int, int, int]
Action = list[int, int, int, int, int, int, int, int, int]

episodes = 1_000
discount_factor = 1

def print_board(board):
    print("-------")
    for x in range(3):
        string = '|'
        for y in range(3):
            string += 'X' * abs(int(board[x * 3 + y] == 1))
            string += 'O' * abs(int(board[x * 3 + y] == -1))
            string += ' ' * abs(int(board[x * 3 + y] == 0))
            string += '|'
        print(string)
    print("-------")

def calculate_epsilon(turn: int) -> float:
    return min(1 / (turn * turn * math.log(turn + 1)), 1.0)
