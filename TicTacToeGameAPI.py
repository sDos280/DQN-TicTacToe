# the reward of each state is:
# * no one is winning = 0
# if someone is winning = 1

import enum

import Consts


class BoardSituationKind(enum.Enum):
    X_WIN = enum.auto()
    O_WIN = enum.auto()
    DRAW = enum.auto()  # full board but no win NEITHER
    NEITHER = enum.auto()  # no win / full board


class GameAPI:
    def __init__(self):
        self.board: Consts.State = [0] * 9  # x -> 1, o -> -1, nothing -> 0
        self.x_turn = True

    def get_cell_from_action(self, action: Consts.Action) -> int:
        return self.board[action]

    def get_board_situation(self):
        # wins check
        for i in range(3):
            # rows
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                if self.board[i * 3] == 1:
                    return BoardSituationKind.X_WIN
                else:
                    return BoardSituationKind.O_WIN

            # columns
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                if self.board[i] == 1:
                    return BoardSituationKind.X_WIN
                else:
                    return BoardSituationKind.O_WIN

        # diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            if self.board[4] == 1:
                return BoardSituationKind.X_WIN
            else:
                return BoardSituationKind.O_WIN
        elif self.board[2] == self.board[4] == self.board[6] != 0:
            if self.board[4] == 1:
                return BoardSituationKind.X_WIN
            else:
                return BoardSituationKind.O_WIN

        for i in range(9):
            if self.board[i] != 0:
                continue
            else:
                break
        else:  # no break
            return BoardSituationKind.DRAW

        return BoardSituationKind.NEITHER

    def get_board_reward(self):
        # win -> +1, draw/else -> 0
        board_situation: BoardSituationKind = self.get_board_situation()

        if board_situation in [BoardSituationKind.X_WIN, BoardSituationKind.O_WIN]:
            return 2.0

        return 1.0

    def step(self, action: Consts.Action) -> tuple[Consts.State, float, bool]:
        # make sure the action is valid
        and_cell = self.get_cell_from_action(action)

        if and_cell != 0:
            raise RuntimeError("Cell is already occupied (do_action)")

        self.board[action] = 1 if self.x_turn else -1

        reward = self.get_board_reward()

        self.x_turn = not self.x_turn  # change the turn to the next turn

        terminated = self.is_terminal()

        return self.board.copy(), reward, terminated

    def get_all_valid_actions(self) -> list[Consts.Action]:
        return [action for action, cell in enumerate(self.board) if cell == 0]

    def is_terminal(self) -> bool:
        board_situation: BoardSituationKind = self.get_board_situation()

        return not board_situation == BoardSituationKind.NEITHER

    def clear(self):
        self.board = [0] * 9
        self.x_turn = True
