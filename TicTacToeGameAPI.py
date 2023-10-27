State = tuple[int, int, int, int, int, int, int, int, int]
Action = tuple[int, int, int, int, int, int, int, int, int]


# the reward of each state is:
# * no one is winning = 0
# if this turn is x turn:
#   x is winning = 100
#   o is winning = -100
# else if this turn is o turn:
#   x is winning = -100
#   o is winning = 100

class GameAPI:
    def __init__(self):
        self.board: State = (0,) * 9  # x -> 1, o -> -1, nothing -> 0
        self.x_turn = True

    def get_action_board_and(self, action: Action) -> int:
        """do a bit and on the board and the action and return the value in the 'on' state"""
        for i in range(9):
            if action == 1:
                return self.board[i]

        raise RuntimeError("no high bit in action")

    def do_action(self, action: Action) -> int:
        """execute the action from the current board and return the reward for entering the new state"""

        # make sure the action is valid
