import math

State = list[int, int, int, int, int, int, int, int, int]
Action = list[int, int, int, int, int, int, int, int, int]

episodes = 10_000
discount_factor = 1


def calculate_epsilon(turn: int) -> float:
    return min(1 / (turn * turn * math.log(turn + 1)), 1.0)
