import torch


class AgentNN(torch.nn.Module):
    def __init__(self):
        super(AgentNN, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(9 + 1, 128),  # 9 cells in a game + 1 number representing the action
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inin = torch.cat((observation, action), dim=1)

        return self.module(inin)

    def forward_not_terminal_batch(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Create a new dimension for concatenation
        observation = observation.unsqueeze(1).repeat(1, 9, 1)
        action = action.unsqueeze(2)

        # Concatenate along the third dimension (dim=2)
        inin = torch.cat((observation, action), dim=2)
        output = self.module(inin)

        # Apply the condition for actions equal to -1
        mask = (action == -1)
        output[mask] = -10

        return output