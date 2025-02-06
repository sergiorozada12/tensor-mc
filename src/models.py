from typing import List, Tuple

import numpy as np
import torch


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
        total_counts = self.transition_counts.sum(
            dim=tuple(range(len(self.state_dims), 2 * len(self.state_dims))),
            keepdim=True,
        )
        transition_probs = self.transition_counts.float() / total_counts
        mask = (total_counts == 0).expand_as(transition_probs)
        transition_probs[mask] = 0
        return transition_probs

    def get_marginal_origin_state(self) -> torch.Tensor:
        marginal_counts = self.transition_counts.sum(
            dim=tuple(range(len(self.state_dims), 2 * len(self.state_dims)))
        )
        marginal_probs = marginal_counts.float() / marginal_counts.sum()
        return marginal_probs

    def get_joint_distribution(self) -> torch.Tensor:
        marginal_counts = self.transition_counts.sum(
            dim=tuple(range(len(self.state_dims), 2 * len(self.state_dims)))
        )
        total_transitions = marginal_counts.sum()
        joint_probs = self.transition_counts.float() / total_transitions
        return joint_probs


def reconstruct_transition_tensor(
    joint_distribution: torch.Tensor,
    marginal_origin_state: torch.Tensor,
    dims: List[int],
) -> torch.Tensor:
    reshaped_marginal_origin_state = marginal_origin_state.view(
        *(dims + (1,) * len(dims))
    )
    recomputed_transition_tensor = joint_distribution / reshaped_marginal_origin_state
    mask = (reshaped_marginal_origin_state == 0).expand_as(recomputed_transition_tensor)
    recomputed_transition_tensor[mask] = 0
    return recomputed_transition_tensor


class MarkovChainTensor:
    def __init__(self, P):
        self.D = P.ndim // 2
        self.P = P
        self.N = torch.tensor(P.shape[: P.ndimension() // 2])
        self.Ntot = torch.prod(self.N).item()
        self.P_1D = P.reshape(self.Ntot, self.Ntot)
        self.pi = None
        self.pi_1D = None
        self.Q = None
        self.Q_1D = None

        self.get_stationary_distribution()
        self.get_joint_matrix()

        self.current_state = None

    def reset(self):
        self.current_state = tuple(
            [torch.randint(0, self.N[d], (1,)).item() for d in range(self.D)]
        )

    def step(self):
        if self.current_state is None:
            raise ValueError(
                "The Markov chain has not been initialized. Call reset() first."
            )

        transition_prob = self.P[self.current_state].flatten()
        next_state = torch.multinomial(transition_prob, 1).item()
        self.current_state = np.unravel_index(next_state, tuple(self.N))
        return self.current_state

    def simulate(self, num_steps: int):
        if self.current_state is None:
            self.reset()

        X = [self.current_state]
        for _ in range(1, num_steps):
            next_state = self.step()
            X.append(next_state)

        return X

    def get_transition_tensor(self):
        return self.P

    def get_stationary_distribution(self):
        if self.pi is None:
            evals, evecs = torch.linalg.eig(self.P_1D.T)
            assert any(
                torch.abs(evals - 1) <= 1e-5
            ), "Invalid probability tensor. There should be at least one eigenvalue with value 1."
            idx_pi = torch.where(torch.abs(evals - 1) <= 1e-5)[0][0]
            pi_1D = torch.abs(torch.real(evecs[:, idx_pi]))
            self.pi_1D = pi_1D / pi_1D.sum()
            self.pi = pi_1D.reshape(tuple(self.N))
        return self.pi

    def get_joint_matrix(self):
        if self.Q is None:
            if self.pi is None:
                self.get_stationary_distribution()

            self.Q_1D = torch.diag(self.pi_1D) @ self.P_1D
            self.Q = self.Q_1D.reshape(tuple(self.N.repeat(2)))
        return self.Q


class MarkovChainMatrix:
    def __init__(self, P):
        self.P = P
        assert P.shape[0] == P.shape[1]
        self.N = P.shape[0]
        self.pi = None
        self.Q = None

        self.get_stationary_distribution()
        self.get_joint_matrix()

        self.current_state = None

    def reset(self):
        self.current_state = torch.randint(0, self.N, (1,)).item()

    def step(self):
        if self.current_state is None:
            raise ValueError(
                "The Markov chain has not been initialized. Call reset() first."
            )

        transition_prob = self.P[self.current_state]
        self.current_state = torch.multinomial(transition_prob, 1).item()
        return self.current_state

    def simulate(self, num_steps: int):
        if self.current_state is None:
            self.reset()

        X = [self.current_state]
        for _ in range(1, num_steps):
            next_state = self.step()
            X.append(next_state)

        return X

    def get_transition_matrix(self):
        return self.P

    def get_stationary_distribution(self):
        if self.pi is None:
            evals, evecs = torch.linalg.eig(self.P.T)
            assert any(
                torch.abs(evals - 1) <= 1e-5
            ), "Invalid probability tensor. There should be at least one eigenvalue with value 1."
            idx_pi = torch.where(torch.abs(evals - 1) <= 1e-5)[0][0]
            pi = torch.abs(torch.real(evecs[:, idx_pi]))
            self.pi = pi / pi.sum()
        return self.pi

    def get_joint_matrix(self):
        if self.Q is None:
            if self.pi is None:
                self.get_stationary_distribution()
            self.Q = torch.diag(self.pi) @ self.P
        return self.Q

    def set_tensor(self, dims):
        self.P_ND = self.P.reshape(tuple(dims.repeat(2)))

    def get_tensor(self):
        if not hasattr(self, "P_ND"):
            print("No tensor version of transition matrix set.")
        else:
            return self.P_ND


class MarkovChainSimulator:
    def __init__(self, transition_probs: torch.Tensor):
        self.transition_probs: torch.Tensor = transition_probs
        self.state_dims: Tuple[int, ...] = transition_probs.shape[
            : transition_probs.ndimension() // 2
        ]
        self.current_state: Tuple[int, ...] = None

    def reset(self) -> None:
        self.current_state = tuple(
            torch.randint(0, dim, (1,)).item() for dim in self.state_dims
        )

    def step(self) -> Tuple[int, ...]:
        if self.current_state is None:
            raise ValueError(
                "The Markov chain has not been initialized. Call reset() first."
            )

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
        for _ in range(1, num_steps):
            next_state = self.step()
            trajectory.append(next_state)

        return trajectory
