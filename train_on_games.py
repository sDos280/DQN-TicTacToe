# import copy
import pickle

import torch

import AgentNN
import Consts
import TicTacToeGameAPI


def train_on_games_experience_list(agent_x=AgentNN.AgentNN(True),
                                   agent_y=AgentNN.AgentNN(False),
                                   out_file_path: str = 'games_experience'):
    with open(out_file_path, 'rb') as opener:
        games_list: list[list[AgentNN.Experience]] = pickle.load(opener)

    agent_x_train = AgentNN.AgentNN(True)
    agent_y_train = AgentNN.AgentNN(False)

    agent_x_train.load_state_dict(agent_x.state_dict())
    agent_y_train.load_state_dict(agent_y.state_dict())

    criterion = torch.nn.MSELoss()
    optimizer_x = torch.optim.SGD(agent_x_train.parameters(), lr=0.01, momentum=0.2)
    optimizer_y = torch.optim.SGD(agent_y_train.parameters(), lr=0.01, momentum=0.2)

    game = TicTacToeGameAPI.GameAPI()
    losses = []

    for i in range(Consts.episodes):
        for turn in range(len(games_list[i])):
            game.board = games_list[i][turn].next_state.copy()

            if game.is_terminal():
                target = torch.tensor([games_list[i][turn].reward])
            else:
                if games_list[i][turn].is_current_turn_x:
                    target = torch.tensor([games_list[i][turn].reward + Consts.discount_factor * agent_x.get_max_q(game)])
                else:
                    target = torch.tensor([games_list[i][turn].reward + Consts.discount_factor * agent_y.get_max_q(game)])

            game.board = games_list[i][turn].current_state.copy()

            if games_list[i][turn].is_current_turn_x:
                optimizer_x.zero_grad()
                prediction = agent_x_train.forward(game.board, games_list[i][turn].action)
            else:
                optimizer_y.zero_grad()
                prediction = agent_y_train.forward(game.board, games_list[i][turn].action)

            loss = criterion(prediction, target)

            """if abs(loss.item()) > 10:
                print()
                Consts.print_board(games_list[i][turn].current_state)
                Consts.print_board(games_list[i][turn].next_state)
                print()"""

            loss.backward()

            losses.append(loss.item())

            if games_list[i][turn].is_current_turn_x:
                optimizer_x.step()
            else:
                optimizer_y.step()

    agent_x.load_state_dict(agent_x_train.state_dict())
    agent_y.load_state_dict(agent_y_train.state_dict())

    return losses
