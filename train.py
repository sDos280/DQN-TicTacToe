import matplotlib.pyplot as plt
import torch

import AgentNN
import gather_experience
import train_on_games

agent_x = AgentNN.AgentNN(is_x=True)
agent_o = AgentNN.AgentNN(is_x=False)
losses = []

for batche in range(100):
    print(f"Current Batch: {batche}")
    print("Do experience")
    gather_experience.gather_games_experience(agent_x, agent_o)
    print("Do Training")
    losses = train_on_games.train_on_games_experience_list(agent_x, agent_o)

fig, ax = plt.subplots()

ax.plot(range(len(losses)), losses)
plt.xlabel("loss iteration")
plt.ylabel("loss value")

plt.show()

torch.save({
    'agent_x_module': agent_x.state_dict(),
    'agent_y_module': agent_o.state_dict()
},
    "./agents_modules.txt"
)
