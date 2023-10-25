import torch
import random
import AgentNN
import TicTacToeGameAPI

# training consts
episodes = 1000
discount_factor = 1
epsilon = 0.2

D_x = set()
D_y = set()

game = TicTacToeGameAPI.GameAPI()

criterion = torch.nn.MSELoss()
# in this demo, we want to train the agent of the x turn (the first turn)

agent_x = AgentNN.AgentNN(True)
agent_y = AgentNN.AgentNN(False)

# training loop
for episode in range(episodes):

    # collect data
    while True:
        current_state = game.state

        if random.random() > 1 - epsilon:  # execute the max action
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
            game.print_state()
            game.clean()
            break

    # train x
    for experience in D_x:
        if game.is_state_terminal():
            y = experience.reward
        else:
            game.state = experience.next_state
            y = experience.reward + discount_factor * agent_x.forward(game.state, agent_x.get_max_q_action(game))

        loss = criterion(agent_x.forward(experience.current_state, experience.action), torch.tensor([y], dtype=torch.float))
        loss.backward()
