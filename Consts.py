import math
from itertools import chain, repeat, islice

State = list[int, int, int, int, int, int, int, int, int]
Action = int  # 0 to 8

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
    return min(1 / math.pow(turn * math.log(turn + 1), 2), 1.0)


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)
