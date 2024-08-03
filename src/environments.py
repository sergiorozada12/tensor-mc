import torch
from typing import Tuple, List

class MarkovChainSimulator:
    def __init__(self, transition_probs: torch.Tensor):
        self.transition_probs: torch.Tensor = transition_probs
        self.state_dims: Tuple[int, ...] = transition_probs.shape[:transition_probs.ndimension() // 2]
        self.current_state: Tuple[int, ...] = None

    def reset(self) -> None:
        self.current_state = tuple(torch.randint(0, dim, (1,)).item() for dim in self.state_dims)

    def step(self) -> Tuple[int, ...]:
        if self.current_state is None:
            raise ValueError("The Markov chain has not been initialized. Call reset() first.")

        current_state_probs: torch.Tensor = self.transition_probs[self.current_state]
        flat_probs: torch.Tensor = current_state_probs.flatten()
        next_state_index: int = torch.multinomial(flat_probs, 1).item()

        next_state: List[int] = []
        for dim in reversed(self.state_dims):
            next_state.append(next_state_index % dim)
            next_state_index //= dim

        self.current_state = tuple(reversed(next_state))
        return self.current_state

    def simulate(self, num_steps: int) -> List[Tuple[int, ...]]:
        if self.current_state is None:
            self.reset()

        trajectory: List[Tuple[int, ...]] = [self.current_state]
        for _ in range(num_steps):
            next_state = self.step()
            trajectory.append(next_state)

        return trajectory
