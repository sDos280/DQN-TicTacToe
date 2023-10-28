import torch

import AgentNN
import Consts
import TicTacToeGameAPI

agent_y = AgentNN.AgentNN(False)
agent_y.load_state_dict(torch.load("agents_modules.txt")['agent_y_module'])
game = TicTacToeGameAPI.GameAPI()
game.clear()

print()
Consts.print_board(game.board)
print()

while True:
    row = int(input("Enter Row: "))
    column = int(input("Enter Column: "))

    player_action = [0] * 9
    player_action[row * 3 + column] = 1
    game.do_action(player_action)

    print()
    Consts.print_board(game.board)
    print()

    if game.is_terminal():
        break

    agent_action = agent_y.get_max_q_action(game)
    game.do_action(agent_action)

    print()
    Consts.print_board(game.board)
    print()

    if game.is_terminal():
        game.clear()
        break
