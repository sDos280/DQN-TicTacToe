import math
import random
from collections import namedtuple
from itertools import count

import torch

import Consts
import TicTacToeGameAPI
from AgentNN import AgentNN
from ReplayMemory import ReplayMemory

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4
TAU = 0.005
LR = 1e-4

env = TicTacToeGameAPI.GameAPI()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

x_wins = 0
o_wins = 0
draws = 0
episode_durations = []

"""def plot_durations():
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated"""


def peek_action(steps_done: Consts.Action, state: torch.Tensor) -> int:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            actions_list = env.get_all_valid_actions()

            # Convert the list of lists to a PyTorch tensor
            actions = torch.tensor(actions_list)

            curated_actions = actions.unsqueeze(1)

            out = policy_net.forward(
                state.repeat(len(actions_list), 1),
                curated_actions
            ).argmax()

            return out
    else:
        return random.randint(0, 8)


"""def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)

    state_action_values = policy_net(state_batch, action_batch)

    actions_list = torch.cat((torch.ones(non_final_next_states.shape[0], 1), torch.zeros(non_final_next_states.shape[0], 1)), dim=1)
    go_to_the_right_actions = torch.cat((torch.zeros(non_final_next_states.shape[0], 1), torch.ones(non_final_next_states.shape[0], 1)), dim=1)

    next_state_values = torch.zeros(BATCH_SIZE, 1)
    with torch.no_grad():
        value_for_left = target_net(non_final_next_states, go_to_the_left_actions)
        value_for_right = target_net(non_final_next_states, go_to_the_right_actions)

        my_max = torch.max(value_for_left, value_for_right)
        next_state_values[non_final_mask] = my_max

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = AgentNN().to(device)
target_net = AgentNN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

if torch.cuda.is_available():
    episodes = 600
else:
    episodes = 600

for episode in range(episodes):
    env.clear()
    last_observation = env.board.copy()
    # curate last observation
    last_observation = torch.tensor(last_observation, dtype=torch.float).unsqueeze(0)
    steps_done = 0

    for t in count():
        action = peek_action(steps_done, last_observation)

        observation, reward, terminal = env.step(action)

        # curate reward and action
        reward = torch.tensor([[reward]], dtype=torch.float)
        action = torch.tensor([[action]], dtype=torch.float)

        if terminal:
            next_state = None
        else:
            # curate observation
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Store the transition in memory
        memory.push(last_observation, action, next_state, reward)

        # Move to the next state
        state = next_state

        """# Perform one step of the optimization (on the policy network)
        optimize_model()"""

        """# Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)"""
