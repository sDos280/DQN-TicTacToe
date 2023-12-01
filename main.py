import math
import random
from collections import namedtuple
from itertools import count

import torch

import Consts
import TicTacToeGameAPI
from AgentNN import AgentNN
from ReplayMemory import ReplayMemory, Transition

# BATCH_SIZE_NO_TERMINAL = 128
BATCH_SIZE_TERMINAL = 15
BATCH_SIZE_NO_TERMINAL = BATCH_SIZE_TERMINAL * 9
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4
TAU = 0.005
LR = 1e-4

env = TicTacToeGameAPI.GameAPI()

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
    if len(memory_terminal) < BATCH_SIZE_TERMINAL:
        return

    if len(memory_no_terminal) < BATCH_SIZE_NO_TERMINAL:
        return

    helper_game = TicTacToeGameAPI.GameAPI()
    criterion = torch.nn.SmoothL1Loss()

    # batch terminal states
    # ----------------------------------------------------------------------
    transitions: Transition = memory_terminal.sample(BATCH_SIZE_TERMINAL)

    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net.forward(state_batch, action_batch)

    # Compute Huber loss
    loss = criterion(state_action_values, reward_batch)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # ----------------------------------------------------------------------

    # batch not terminal states
    # ----------------------------------------------------------------------
    transitions: Transition = memory_no_terminal.sample(BATCH_SIZE_NO_TERMINAL)

    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    state_next_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    allowed_actions_batch = torch.cat(batch.allowed_actions)

    state_action_values = policy_net.forward(state_batch, action_batch)

    with torch.no_grad():
        next_state_values = target_net.forward_not_terminal_batch(state_next_batch, allowed_actions_batch)

        my_max = torch.max(next_state_values, 1)[0]
        next_state_values = torch.mul(my_max, -1.0)

    expected_state_action_values = torch.mul(next_state_values, GAMMA)

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # ----------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = AgentNN().to(device)
target_net = AgentNN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory_no_terminal = ReplayMemory(10000)
memory_terminal = ReplayMemory(10000)

if torch.cuda.is_available():
    episodes = 600
    # episodes = 100
else:
    episodes = 6000

for episode in range(episodes):
    env.clear()
    state = env.board.copy()
    # curate last observation
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    steps_done = 0

    for t in count():
        allowed_actions = env.get_all_valid_actions()
        action = peek_action(steps_done, state)

        observation, reward, terminal = env.step(action)

        # curate reward and action
        reward = torch.tensor([[reward]], dtype=torch.float, device=device)
        action = torch.tensor([[action]], dtype=torch.float, device=device)

        if terminal:
            next_state = None

            memory_terminal.push(state, action, None, reward, None)
        else:
            # pad allowed_actions
            allowed_actions = list(Consts.pad(allowed_actions, 9, -1))

            # curate observation and allowed_actions
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            allowed_actions = torch.tensor([allowed_actions], dtype=torch.float, device=device)

            # Store the transition in memory_no_terminal
            memory_no_terminal.push(state, action, next_state, reward, allowed_actions)

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

    print(f"Episode: {episode}, memory size: {len(memory_no_terminal)}, Draw prob: {draws / episodes}")
