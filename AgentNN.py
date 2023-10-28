import torch
import torch.nn as nn

import Consts
import TicTacToeGameAPI


# a State is just a list of cells
# with the value of each list cell been the thing that occupies that specific sell

# an Action is just a list of cells (bit mask), the marked cell (with a value of 1 for x turn or a value of -1 for o turn) in the list is the cell that the agent want to mark on the board

class AgentNN(nn.Module):
    # the agent network is the network that represent the q value of state
    def __init__(self, is_x: bool):
        super(AgentNN, self).__init__()
        self.module = torch.nn.Sequential(
            nn.Linear(18, 36),
            nn.Sigmoid(),
            nn.Linear(36, 36),
            nn.Sigmoid(),
            nn.Linear(36, 18),
            nn.Sigmoid(),
            nn.Linear(18, 9),
            nn.Sigmoid(),
            nn.Linear(9, 1),
        )
        self.is_x = is_x

    def forward(self, state: Consts.State, action: Consts.Action):
        input_list = state.copy()
        input_list.extend(action)

        return self.module(torch.tensor(input_list, dtype=torch.float))

    def get_max_q(self, game: TicTacToeGameAPI.GameAPI):
        valid_actions = game.get_all_valid_actions()
        q_over_all_valid_actions = []

        with torch.no_grad():
            for action in valid_actions:
                q_over_all_valid_actions.append(self.forward(game.board, action))

        max_q = max(q_over_all_valid_actions)

        return max_q

    def get_max_q_action(self, game: TicTacToeGameAPI.GameAPI) -> Consts.Action:
        max_q = self.get_max_q(game)
        max_q_index = q_over_all_valid_actions.index(max_q)
        max_q_action = valid_actions[max_q_index]

        return max_q_action


class Experience:
    def __init__(self, current_state: Consts.State, is_current_turn_x: bool, action: Consts.Action, reward: float, next_state: Consts.State):
        self.current_state: Consts.State = current_state
        self.is_current_turn_x: bool = is_current_turn_x
        self.action: Consts.Action = action
        self.reward: float = reward
        self.next_state: Consts.State = next_state
