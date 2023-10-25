import random

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

def reward_function(state: State, x_turn: bool):
    checks = [
        # rows
        [
            (0, 0),
            (0, 1),
            (0, 2),
        ],
        [
            (1, 0),
            (1, 1),
            (1, 2),
        ],
        [
            (2, 0),
            (2, 1),
            (2, 2),
        ],

        # columns
        [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
        [
            (0, 1),
            (1, 1),
            (2, 1),
        ],
        [
            (0, 2),
            (1, 2),
            (2, 2),
        ],

        # diagonals
        [
            (0, 0),
            (1, 1),
            (2, 2),
        ],
        [
            (2, 0),
            (1, 1),
            (0, 2),
        ]
    ]

    for check in checks:
        current_check_sum = get_cell_from_state(*check[0], state=state) + \
                            get_cell_from_state(*check[1], state=state) + \
                            get_cell_from_state(*check[2], state=state)

        if x_turn:
            if current_check_sum == 3:  # x is winning
                return 100
            elif current_check_sum == -3:  # o is winning
                return -100
            else:  # loop back for the next check
                continue
        else:
            if current_check_sum == 3:  # x is winning
                return -100
            elif current_check_sum == -3:  # o is winning
                return 100
            else:  # loop back for the next check
                continue

    return 0


# the top left sell is cell 0, 0
# to get the cell in the r-th row and in the c-th column we use the following state[row * 3 + column]
def get_cell_from_state(row: int, column: int, state: State):
    return state[row * 3 + column]


def get_cell_from_action_mask(action: Action, state: State):
    return get_cell_from_state(*get_row_column_from_action(action), state)


def get_row_column_from_action(action: State):
    for row in range(3):
        for column in range(3):
            if action[row * 3 + column] != 0:
                return row, column

    raise RuntimeError("no mask in the action")


# a state is a terminal state if one of the player won or when the board is full
def is_state_terminal(state: State) -> bool:
    # check for a win
    checks = [
        # rows
        [
            (0, 0),
            (0, 1),
            (0, 2),
        ],
        [
            (1, 0),
            (1, 1),
            (1, 2),
        ],
        [
            (2, 0),
            (2, 1),
            (2, 2),
        ],

        # columns
        [
            (0, 0),
            (1, 0),
            (2, 0),
        ],
        [
            (0, 1),
            (1, 1),
            (2, 1),
        ],
        [
            (0, 2),
            (1, 2),
            (2, 2),
        ],

        # diagonals
        [
            (0, 0),
            (1, 1),
            (2, 2),
        ],
        [
            (2, 0),
            (1, 1),
            (0, 2),
        ]
    ]

    for check in checks:
        current_check_sum = get_cell_from_state(*check[0], state=state) + \
                            get_cell_from_state(*check[1], state=state) + \
                            get_cell_from_state(*check[2], state=state)

        if current_check_sum == 3:  # x is winning
            return True
        elif current_check_sum == -3:  # o is winning
            return True
        else:  # loop back for the next check
            continue

    # check for a full board
    for row in range(3):
        for column in range(3):
            if state[row * 3 + column] == 0:  # check for an empty cell
                return False

    # all cells are fill
    return True


class GameAPI:
    def __init__(self):
        self.state = (0,) * 9
        self.x_turn = True

    def get_current_reward(self):
        return reward_function(self.state, self.x_turn)

    def get_all_valid_actions(self) -> list[Action]:
        valid_actions: list[Action] = []

        for row in range(3):
            for column in range(3):
                if get_cell_from_state(row, column, self.state) == 0:  # if the cell is 0, that means that it is not occupied, so the action is valid
                    action = [0] * 9
                    action[row * 3 + column] = 1 if self.x_turn else -1

                    valid_actions.append(tuple(action))

        return valid_actions

    def get_random_action(self) -> Action:
        valid_actions = self.get_all_valid_actions()

        return valid_actions[random.randint(0, len(valid_actions) - 1)]

    def do_action(self, action):
        # returns True on success

        if get_cell_from_action_mask(action, self.state) != 0:  # if the cell is not 0, that means that it is occupied, so the action is invalid
            return False

        row, column = get_row_column_from_action(action)

        state = list(self.state)

        state[row * 3 + column] = 1 if self.x_turn else -1

        self.state = tuple(state)

        self.x_turn = not self.x_turn  # set next plater turn

        return True

    def is_someone_winning(self):
        checks = [
            # rows
            [
                (0, 0),
                (0, 1),
                (0, 2),
            ],
            [
                (1, 0),
                (1, 1),
                (1, 2),
            ],
            [
                (2, 0),
                (2, 1),
                (2, 2),
            ],

            # columns
            [
                (0, 0),
                (1, 0),
                (2, 0),
            ],
            [
                (0, 1),
                (1, 1),
                (2, 1),
            ],
            [
                (0, 2),
                (1, 2),
                (2, 2),
            ],

            # diagonals
            [
                (0, 0),
                (1, 1),
                (2, 2),
            ],
            [
                (2, 0),
                (1, 1),
                (0, 2),
            ]
        ]

        for check in checks:
            current_check_sum = get_cell_from_state(*check[0], state=self.state) + \
                                get_cell_from_state(*check[1], state=self.state) + \
                                get_cell_from_state(*check[2], state=self.state)

            if current_check_sum == 3:  # x is winning
                return 1
            elif current_check_sum == -3:  # o is winning
                return -1
            else:  # loop back for the next check
                continue

        return 0  # no one is winning

    def is_full(self):
        for row in range(3):
            for column in range(3):
                if self.state[row * 3 + column] == 0:
                    return False

        return True

    def is_state_terminal(self):
        return is_state_terminal(self.state)

    def clean(self):
        self.state = (0,) * 9

    def print_state(self):
        for row in range(3):
            for column in range(3):
                if self.state[row * 3 + column] == 0:
                    print("0", end="")
                elif self.state[row * 3 + column] == 1:
                    print("x", end="")
                elif self.state[row * 3 + column] == -1:
                    print("o", end="")
            print()
        print()
