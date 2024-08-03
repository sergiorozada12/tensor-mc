import torch
from typing import List, Tuple


class ProbabilityTensorEstimator:
    def __init__(self, state_dims: Tuple[int, ...]):
        self.state_dims = state_dims
        self.transition_counts = torch.zeros(state_dims + state_dims, dtype=torch.int32)

    def fit(self, trajectories: List[List[Tuple[int, ...]]]) -> None:
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                current_state = trajectory[i]
                next_state = trajectory[i + 1]
                self.transition_counts[current_state + next_state] += 1

    def get_transition_tensor(self) -> torch.Tensor:
        total_counts = self.transition_counts.sum(dim=tuple(range(len(self.state_dims), 2*len(self.state_dims))), keepdim=True)
        transition_probs = self.transition_counts.float() / total_counts
        mask = (total_counts == 0).expand_as(transition_probs)
        transition_probs[mask] = 0
        return transition_probs

    def get_marginal_origin_state(self) -> torch.Tensor:
        marginal_counts = self.transition_counts.sum(dim=tuple(range(len(self.state_dims), 2*len(self.state_dims))))
        marginal_probs = marginal_counts.float() / marginal_counts.sum()
        return marginal_probs

    def get_joint_distribution(self) -> torch.Tensor:
        marginal_counts = self.transition_counts.sum(dim=tuple(range(len(self.state_dims), 2*len(self.state_dims))))
        total_transitions = marginal_counts.sum()
        joint_probs = self.transition_counts.float() / total_transitions
        return joint_probs


def reconstruct_transition_tensor(
        joint_distribution: torch.Tensor,
        marginal_origin_state: torch.Tensor,
        dims: List[int]
    ) -> torch.Tensor:
    reshaped_marginal_origin_state = marginal_origin_state.view(*(dims + (1,) * len(dims)))
    recomputed_transition_tensor = joint_distribution / reshaped_marginal_origin_state
    mask = (reshaped_marginal_origin_state == 0).expand_as(recomputed_transition_tensor)
    recomputed_transition_tensor[mask] = 0
    return recomputed_transition_tensor
