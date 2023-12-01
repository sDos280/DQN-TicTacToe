import random
from collections import deque
from typing import NamedTuple

import torch


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor | None
    reward: torch.Tensor
    allowed_actions: torch.Tensor | None


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def choice(self):
        return random.choice(self.memory)

    def __len__(self):
        return len(self.memory)
