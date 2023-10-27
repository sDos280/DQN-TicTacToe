State = list[int, int, int, int, int, int, int, int, int]
Action = list[int, int, int, int, int, int, int, int, int]


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
        self.board: State = [0, ] * 9  # x -> 1, o -> -1, nothing -> 0
        self.x_turn = True

    def get_action_board_and(self, action: Action) -> int:
        """do a bit and on the board and the action and return the value in the 'on' state"""
        for i in range(9):
            if action == 1:
                return self.board[i]

        raise RuntimeError("No high bit in action (get_action_board_and)")

    def get_board_reward(self) -> int:
        """return the reward of current board based on the turn"""
        for i in range(3):
            # rows
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                if self.x_turn:
                    if self.board[i] == 1:
                        return 100
                    elif self.board[i] == -1:
                        return -100
                else:
                    if self.board[i] == 1:
                        return -100
                    elif self.board[i] == -1:
                        return 100

            # columns
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                if self.x_turn:
                    if self.board[i] == 1:
                        return 100
                    elif self.board[i] == -1:
                        return -100
                else:
                    if self.board[i] == 1:
                        return -100
                    elif self.board[i] == -1:
                        return 100

        return 0

    def do_action(self, action: Action) -> int:
        """execute the action from the current board and return the reward for entering the new state"""

        # make sure the action is valid
        and_cell = self.get_action_board_and(action)

        if and_cell != 0:
            raise RuntimeError("Cell is already occupied (do_action)")

        action_index_cell = action.index(1)

        self.board[action_index_cell] = 1 if self.x_turn else -1

        reward = self.get_board_reward()

        self.x_turn = not self.x_turn  # change the turn to the next turn

        return reward
