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
        input_list = state
        input_list.extend(action)

        return self.module(torch.tensor(input_list, dtype=torch.float))

    def get_max_q_action(self, game: TicTacToeGameAPI.GameAPI) -> Consts.Action:
        valid_actions = game.get_all_valid_actions()
        q_over_all_valid_actions = []

        with torch.no_grad():
            for action in valid_actions:
                q_over_all_valid_actions.append(self.forward(game.state, action))

        max_q = max(q_over_all_valid_actions)
        max_q_index = q_over_all_valid_actions.index(max_q)
        max_q_action = valid_actions[max_q_index]

        return max_q_action


class Experience:
    def __init__(self, current_state: State, action: Action, reward, next_state: State):
        self.current_state: State = current_state
        self.action: Action = action
        self.reward = reward
        self.next_state: State = next_state
