"""

a script which gather experience (AKA games which are just a list of experiences) and output it as pickled format into a file

"""

import pickle
import random

import AgentNN
import Consts
import TicTacToeGameAPI


def gather_games_experience(agent_x=AgentNN.AgentNN(True), agent_y=AgentNN.AgentNN(False), out_file_path: str = 'games_experience'):
    games_list: list[list[AgentNN.Experience]] = []

    game = TicTacToeGameAPI.GameAPI()

    for i in range(Consts.episodes):  # the number of episodes is the number of games to be played
        games_list.append([])  # add a new game
        turn = 1

        while True:
            epsilon = Consts.calculate_epsilon(turn)

            current_board = game.board.copy()
            current_turn = game.x_turn

            if epsilon < random.random():  # execute the max action
                if game.x_turn:
                    execute_action = agent_x.get_max_q_action(game)
                else:
                    execute_action = agent_y.get_max_q_action(game)
            else:
                execute_action = random.choice(game.get_all_valid_actions())

            reward = game.do_action(execute_action)
            turn += 1

            next_board = game.board.copy()

            games_list[i].append(AgentNN.Experience(current_board, current_turn, execute_action, reward, next_board))

            if game.is_terminal():
                game.clear()
                turn = 1
                break

    with open(out_file_path, 'wb') as opener:
        pickle.dump(games_list, opener)


gather_games_experience()
