import math
import random
from collections import namedtuple
from itertools import count

import torch

import Consts
import TicTacToeGameAPI
from AgentNN import AgentNN
from ReplayMemory import ReplayMemory

# BATCH_SIZE = 128
BATCH_SIZE = 64
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
            actions = torch.tensor(actions_list, device=device)

            # curate actions
            actions = actions.unsqueeze(1)

            out = policy_net.forward(
                state.repeat(len(actions_list), 1),
                actions
            ).argmax()

            return actions_list[out]
    else:
        return random.choice(env.get_all_valid_actions())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    helper_game = TicTacToeGameAPI.GameAPI()

    for i in range(BATCH_SIZE):
        batch: Transition = memory.choice()

        if batch.next_state is None:
            expected_state_action_value = batch.reward
            state_action_value = policy_net.forward(batch.state, batch.action)
        else:
            helper_game.board = batch.next_state.squeeze(0).tolist()

            possible_action_in_next_state = helper_game.get_all_valid_actions()

            # curate next states batch
            non_final_next_states = batch.next_state.repeat(len(possible_action_in_next_state), 1)

            # curate possible_action_in_next_state
            possible_action_in_next_state = torch.tensor(possible_action_in_next_state, dtype=torch.float, device=device).unsqueeze(1)
            with torch.no_grad():
                next_state_value = torch.max(
                    target_net.forward(non_final_next_states, possible_action_in_next_state)
                ).view((1, 1))

            # expected_state_action_value = (next_state_value * GAMMA) + batch.reward
            expected_state_action_value = next_state_value

            state_action_value = policy_net.forward(batch.state, batch.action)

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        # loss = criterion(state_action_value, expected_state_action_value)

        loss = criterion(state_action_value, expected_state_action_value)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = AgentNN().to(device)
target_net = AgentNN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

if torch.cuda.is_available():
    episodes = 600
    # episodes = 100
else:
    episodes = 600

for episode in range(episodes):
    env.clear()
    state = env.board.copy()
    # curate last observation
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    steps_done = 0

    for t in count():
        action = peek_action(steps_done, state)

        observation, reward, terminal = env.step(action)

        # curate reward and action
        reward = torch.tensor([[reward]], dtype=torch.float, device=device)
        action = torch.tensor([[action]], dtype=torch.float, device=device)

        if terminal:
            next_state = None
        else:
            # curate observation
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        """status = env.get_board_situation()
        if status == TicTacToeGameAPI.BoardSituationKind.DRAW:
            breakpoint()"""

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        # Consts.print_board(observation)

        if terminal:
            board_status = env.get_board_situation()
            if board_status == TicTacToeGameAPI.BoardSituationKind.X_WIN:
                x_wins += 1
            elif board_status == TicTacToeGameAPI.BoardSituationKind.O_WIN:
                o_wins += 1
            elif board_status == TicTacToeGameAPI.BoardSituationKind.DRAW:
                draws += 1

            break

    print(f"Episode: {episode}, memory size: {len(memory)}, Draw prob: {draws / episodes}")
