# the reward of each state is:
# * no one is winning = 0
# if this turn is x turn:
#   x is winning = 1
#   o is winning = -1
# else if this turn is o turn:
#   x is winning = -1
#   o is winning = 1
import Consts


class GameAPI:
    def __init__(self):
        self.board: Consts.State = [0] * 9  # x -> 1, o -> -1, nothing -> 0
        self.x_turn = True

    def get_action_board_and(self, action: Consts.Action) -> int:
        """do a bit and on the board and the action and return the value in the 'on' state"""
        for i in range(9):
            if action[i] == 1:
                return self.board[i]

        raise RuntimeError("No high bit in action (get_action_board_and)")

    def get_board_reward(self, turn) -> float:
        """return the reward of current board based on the turn"""
        for i in range(3):
            # rows
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                if turn:
                    if self.board[i] == 1:
                        return 1.0
                    elif self.board[i] == -1:
                        return -1.0
                else:
                    if self.board[i] == 1:
                        return -1.0
                    elif self.board[i] == -1:
                        return 1.0

            # columns
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                if turn:
                    if self.board[i] == 1:
                        return 1.0
                    elif self.board[i] == -1:
                        return -1.0
                else:
                    if self.board[i] == 1:
                        return -1.0
                    elif self.board[i] == -1:
                        return 1.0

            # diagonals
            if self.board[0] == self.board[4] == self.board[8] != 0:
                if turn:
                    if self.board[i] == 1:
                        return 1.0
                    elif self.board[i] == -1:
                        return -1.0
                else:
                    if self.board[i] == 1:
                        return -1.0
                    elif self.board[i] == -1:
                        return 1.0
            elif self.board[2] == self.board[4] == self.board[6] != 0:
                if turn:
                    if self.board[i] == 1:
                        return 1.0
                    elif self.board[i] == -1:
                        return -1.0
                else:
                    if self.board[i] == 1:
                        return -1.0
                    elif self.board[i] == -1:
                        return 1.0

        return 0.0

    def do_action(self, action: Consts.Action) -> int:
        """execute the action from the current board and return the reward for entering the new state"""

        # make sure the action is valid
        and_cell = self.get_action_board_and(action)

        if and_cell != 0:
            raise RuntimeError("Cell is already occupied (do_action)")

        action_index_cell = action.index(1)

        self.board[action_index_cell] = 1 if self.x_turn else -1

        reward = self.get_board_reward(self.x_turn)

        self.x_turn = not self.x_turn  # change the turn to the next turn

        return reward

    def get_all_valid_actions(self) -> list[Consts.Action]:
        """return a list of all the valid actions from the current board"""
        valid_actions = []

        for i in range(9):
            if self.board[i] == 0:
                action = [0] * 9
                action[i] = 1

                valid_actions.append(action)

        return valid_actions

    def is_terminal(self) -> bool:
        """return if a board state is a terminal state, aka a win of a full board"""

        # check for a win
        for i in range(3):
            # rows
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                return True

            # columns
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                return True

        # diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return True
        elif self.board[2] == self.board[4] == self.board[6] != 0:
            return True

        # check for full board
        for i in range(9):
            if self.board[i] == 0:
                return False

        return True

    def clear(self):
        self.board = [0] * 9
        self.x_turn = True
