import torch
import Consts
import TicTacToeGameAPI
from AgentNN import AgentNN
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peek_action(state: torch.Tensor) -> int:
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


policy_net = AgentNN().to(device)
policy_net.load_state_dict(torch.load("agents.pt")["policy_net"])

env = TicTacToeGameAPI.GameAPI()

for episode in range(100):
    print("--------------------------------------------------------------")
    env.clear()
    state = env.board.copy()
    # curate last observation
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    steps_done = 0

    Consts.print_board(env.board)

    for t in count():
        if not env.x_turn:  # player turn
            action = input("Enter Action, 0-8: (for quiting enter q)")
            if action == "q":
                exit()

            action = int(action)
        else:  # bot turn
            action = peek_action(state)

        observation, reward, terminal = env.step(action)

        Consts.print_board(env.board)

        # curate observation
        observation = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)

        # Move to the next state
        state = observation

        if terminal:
            print("--------------------------------------------------------------")
            break
