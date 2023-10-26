import random
import sys

import torch

import AgentNN
import TicTacToeGameAPI

# training consts
episodes = 100000
discount_factor = 1
epsilon = 0.05

D_x = set()
D_y = set()

game = TicTacToeGameAPI.GameAPI()

criterion = torch.nn.MSELoss()
# in this demo, we want to train the agent of the x turn (the first turn)

agent_x = AgentNN.AgentNN(True)
agent_y = AgentNN.AgentNN(False)

x_win = 0
o_win = 0
draw = 0

# training loop
for episode_num, episode in enumerate(range(episodes)):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write(f"%{(episode_num + 1) / episodes * 100}\n")
    sys.stdout.flush()

    game.do_action(game.get_random_action())

    # collect data
    while True:
        current_state = game.state

        if 1 - epsilon > random.random():  # execute the max action
            if game.x_turn:
                executed_action = agent_x.get_max_q_action(game)
            else:
                executed_action = agent_y.get_max_q_action(game)
        else:
            executed_action = game.get_random_action()
        game.do_action(executed_action)

        reward = game.get_current_reward()
        next_state = game.state

        if not game.x_turn:  # (after doing an action the turn is changed) x experience
            D_x.add(AgentNN.Experience(current_state, executed_action, reward, next_state))
        else:  # y experience
            D_y.add(AgentNN.Experience(current_state, executed_action, reward, next_state))

        if game.is_state_terminal():
            if game.is_someone_winning() == 1:
                x_win += 1
            elif game.is_someone_winning() == -1:
                o_win += 1
            elif game.is_full():
                draw += 1
            # print(f"X win: {x_win}, O win: {o_win}, Draw: {draw}", end="")
            # game.print_state()
            game.clean()
            break

    # train x
    for experience in D_x:
        game.state = experience.next_state

        if game.is_state_terminal():
            target = experience.reward
        else:
            game.state = experience.next_state
            target = experience.reward + discount_factor * agent_x.forward(game.state, agent_x.get_max_q_action(game))

        loss = criterion(agent_x.forward(experience.current_state, experience.action), torch.tensor([target], dtype=torch.float))
        loss.backward()

    D_x.clear()

    # train y
    for experience in D_y:
        game.state = experience.next_state

        if game.is_state_terminal():
            target = experience.reward
        else:
            game.state = experience.next_state
            target = experience.reward + discount_factor * agent_y.forward(game.state, agent_y.get_max_q_action(game))

        loss = criterion(agent_y.forward(experience.current_state, experience.action), torch.tensor([target], dtype=torch.float))
        loss.backward()

    D_y.clear()

    game.clean()

print(f"X win: {x_win}, O win: {o_win}, Draw: {draw}", end="")

torch.save({
    'agent_x_module': agent_x.state_dict(),
    'agent_y_module': agent_y.state_dict()
    },
    "./out.txt"
)
