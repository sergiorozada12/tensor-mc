from typing import Optional, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.sparse.linalg import svds

from src.models import MarkovChainMatrix, MarkovChainTensor
from src.optimizers import (
    ADMMDCLRMFromCondPMF,
    ADMMLRTFromJointPMF,
    ADMMNNLRMFromCondPMF,
    cp_to_tensor,
)
from src.utils import color_schemes, madimshow


def frob_err(Ph: torch.Tensor, P: torch.Tensor) -> float:
    return torch.norm(Ph - P, "fro").item()


def normfrob_err(Ph: torch.Tensor, P: torch.Tensor) -> float:
    norm_term = torch.norm(P, "fro") + int(P.abs().max() > 0)
    return (torch.norm(Ph - P, "fro") / norm_term).item()


def l1_err(Ph: torch.Tensor, P: torch.Tensor) -> float:
    return torch.norm(Ph - P, 1).item()


def norml1_err(Ph: torch.Tensor, P: torch.Tensor) -> float:
    norm_term = torch.norm(P, 1) + int(P.abs().max() > 0)
    return (torch.norm(Ph - P, 1) / norm_term).item()


def sin_err(
    Ph: torch.Tensor,
    P: torch.Tensor,
    K: Optional[int] = None,
    sv_type: Literal["both", "left", "right"] = "both",
) -> float:
    if K is None:
        K = P.shape[0]

    if K < P.shape[0]:
        U, _, V = svds(P.numpy().astype(float), K)
        Uh, _, Vh = svds(Ph.numpy().astype(float), K)
    else:
        U, _, VT = np.linalg.svd(P.numpy().astype(float))
        V = VT.T
        Uh, _, VhT = np.linalg.svd(Ph.numpy().astype(float))
        Vh = VhT.T

    if sv_type == "right":
        return float(
            np.sqrt(np.abs(K - np.linalg.norm(Vh @ V.T, "fro") ** 2)) / np.sqrt(K)
        )
    elif sv_type == "left":
        return float(
            np.sqrt(np.abs(K - np.linalg.norm(Uh.T @ U, "fro") ** 2)) / np.sqrt(K)
        )
    else:
        left_term = np.sqrt(np.abs(K - np.linalg.norm(Uh.T @ U, "fro") ** 2))
        right_term = np.sqrt(np.abs(K - np.linalg.norm(Vh @ V.T, "fro") ** 2))
        return float(
            torch.tensor([left_term, right_term]).to(torch.float).max().item()
            / np.sqrt(K)
        )


def erank(A: torch.Tensor) -> torch.Tensor:
    svs = torch.linalg.svdvals(A)
    p = svs / (torch.linalg.norm(svs, 1) + int(svs.abs().max() == 0))
    return torch.exp(-torch.sum(p * torch.log(p)))


def mat2lowtri(
    A: Union[np.ndarray, torch.Tensor], N: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    N = A.shape[0] if N is None else N
    low_tri_indices = np.triu_indices(N, 1)
    return A[low_tri_indices[1], low_tri_indices[0]]


def lowtri2mat(a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    N = int((2 * len(a) + 0.25) ** 0.5 + 0.5)
    A = np.zeros((N, N)) if isinstance(a, np.ndarray) else torch.zeros((N, N))
    low_tri_indices = np.triu_indices(N, 1)
    A[low_tri_indices[1], low_tri_indices[0]] = a
    return A + A.T


def generate_erdosrenyi_matrix_model(
    N: int, edge_prob: float = 0.3, eps: float = 0.02, beta: float = 1.0
) -> "MarkovChainMatrix":
    A = lowtri2mat(np.random.binomial(1, edge_prob, int(N * (N - 1) / 2)))
    L = np.diag(A.sum(0)) - A
    while np.sum(np.abs(np.linalg.eigvalsh(L)) < 1e-9) > 1:
        A = lowtri2mat(np.random.binomial(1, edge_prob, int(N * (N - 1) / 2)))
        L = np.diag(A.sum(0)) - A
    At = A + beta * np.eye(N)
    P0 = At * (1 - eps) + eps
    P = torch.tensor(P0 / P0.sum(1, keepdims=True)).to(torch.float)
    mc = MarkovChainMatrix(P)
    return mc


def generate_lowranktensor_model(N: torch.Tensor, K: int) -> "MarkovChainTensor":
    D = len(N)
    Ntot = torch.prod(N).item()
    U = [torch.randn(N[d % D], K) for d in range(2 * D)]
    U = [
        (U[d] * U[d]) / (torch.linalg.norm(U[d], dim=0, keepdim=True) ** 2)
        for d in range(2 * D)
    ]
    w = torch.FloatTensor(np.random.rand(K))
    w = w / w.sum()
    P = cp_to_tensor((w, U))
    P_mat = P.reshape(Ntot, Ntot)
    P_mat = P_mat / P_mat.sum(dim=1, keepdim=True)
    P = P_mat.reshape(tuple(N.repeat(2)))
    mc = MarkovChainTensor(P)
    return mc


def generate_lowrankmatrix_model(N: int, K: int) -> "MarkovChainMatrix":
    U0 = torch.randn(N, K)
    U0 = (U0 * U0) / (torch.linalg.norm(U0, dim=1, keepdim=True) ** 2)
    V0 = torch.randn(N, K)
    V0 = (V0 * V0) / (torch.linalg.norm(V0, dim=0, keepdim=True) ** 2)
    B = torch.diag(torch.FloatTensor(np.random.beta(0.5, 0.5, N)))
    P = (U0 @ V0.T) @ B
    P = P / P.sum(dim=1, keepdim=True)
    mc = MarkovChainMatrix(P)
    return mc


def generate_blocktensor_model(N_per_block: int, D: int, K: int):
    state_mat = torch.rand((K,) * (2 * D))
    Q = torch.kron(state_mat, torch.ones((N_per_block // K,) * (2 * D)))
    Q /= torch.linalg.norm(Q.reshape(-1), 1)
    P = Q / Q.sum(dim=tuple(range(D, 2 * D)), keepdim=True)
    return MarkovChainTensor(P)


def generate_blockmatrix_model(N_per_block: int, K: int):
    return generate_blocktensor_model(N_per_block, 1, K)


def estimate_empirical_matrix(X, N: int):
    return _estimate_empirical(X, (N, N))


def estimate_empirical_tensor(X, N):
    return _estimate_empirical(X, tuple(N.repeat(2)))


def _estimate_empirical(X, shape):
    transition_counts = torch.zeros(shape, dtype=int)
    X_steps = np.array([np.array(X)[:-1], np.array(X)[1:]])
    X_steps, counts = np.unique(X_steps, axis=1, return_counts=True)
    transition_counts[tuple(X_steps)] = torch.tensor(counts)

    total_counts = transition_counts.sum(
        dim=tuple(range(len(shape) // 2, len(shape))), keepdim=True
    )
    Ph = transition_counts.float() / total_counts
    Mask = (total_counts == 0).expand_as(Ph)
    Ph[Mask] = 1 / np.prod(shape[: len(shape) // 2])

    marginal_counts = transition_counts.sum(
        dim=tuple(range(len(shape) // 2, len(shape)))
    )
    total_transitions = marginal_counts.sum()
    Qh = transition_counts.float() / total_transitions

    return Ph, Qh


class LowRankTensorEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        attributes = [
            "mc",
            "P",
            "Q",
            "P_1D",
            "Q_1D",
            "pi",
            "pi_1D",
            "lmbda",
            "Qfact",
            "admm_err",
            "admm_res",
            "admm_var",
            "admm_iters",
            "Qh",
            "K",
            "beta",
            "eps_abs",
            "eps_rel",
            "eps_diff",
            "max_itr",
            "verbose",
            "MARG_CONST",
            "ACCEL",
        ]
        for attr in attributes:
            setattr(self, attr, None)

    def estimate(self, Q_obs, args):
        self.reset()
        D = Q_obs.ndim // 2
        N = torch.tensor(Q_obs.shape[:D])
        Ntot = torch.prod(N).item()

        Q_LRT, (lmbda, Qfact), err_LRT, (rp_LRT, rd_LRT, ep_LRT, ed_LRT), var_diff = (
            ADMMLRTFromJointPMF(Q_obs, **args).optimize()
        )

        assert not torch.isnan(Q_LRT).any(), "Nan values in estimated Q."
        assert not torch.isinf(Q_LRT).any(), "Inf. values in estimated Q."

        marg_lrt = Q_LRT.sum(dim=tuple(range(D, 2 * D)), keepdim=True)
        P_LRT = Q_LRT / (marg_lrt + (marg_lrt == 0).to(int))
        P_LRT[(marg_lrt == 0).expand_as(P_LRT)] = 1 / Ntot

        self._store_results(
            Q_obs,
            P_LRT,
            Q_LRT,
            marg_lrt,
            lmbda,
            Qfact,
            err_LRT,
            rp_LRT,
            rd_LRT,
            ep_LRT,
            ed_LRT,
            var_diff,
            args,
        )
        return self.mc, {
            attr: getattr(self, attr)
            for attr in [
                "mc",
                "lmbda",
                "Qfact",
                "admm_obj",
                "admm_res",
                "admm_var",
                "admm_iters",
            ]
        }

    def _store_results(
        self,
        Q_obs,
        P_LRT,
        Q_LRT,
        marg_lrt,
        lmbda,
        Qfact,
        err_LRT,
        rp_LRT,
        rd_LRT,
        ep_LRT,
        ed_LRT,
        var_diff,
        args,
    ):
        D = Q_obs.ndim // 2
        N = torch.tensor(Q_obs.shape[:D])
        Ntot = torch.prod(N).item()

        self.mc = MarkovChainTensor(P_LRT)
        self.P, self.Q = P_LRT, Q_LRT
        self.P_1D, self.Q_1D = P_LRT.reshape(Ntot, Ntot), Q_LRT.reshape(Ntot, Ntot)
        self.pi, self.pi_1D = marg_lrt, marg_lrt.reshape(Ntot)
        self.lmbda, self.Qfact = lmbda, Qfact
        self.admm_obj, self.admm_res = err_LRT, (rp_LRT, rd_LRT, ep_LRT, ed_LRT)
        self.admm_var, self.admm_iters = var_diff, len(err_LRT)
        self.Qh = Q_obs
        for key, value in args.items():
            setattr(self, key, value)

    def _plot_helper(self, data, ylabel, start_idx=0):
        assert data is not None, f"{ylabel} not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.plot(
            np.arange(start_idx, self.admm_iters),
            data[start_idx:],
            "-",
            c=color_schemes["vib_qual"]["red"],
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        fig.tight_layout()

    def plot_admm_objective(self, start_idx=0):
        self._plot_helper(self.admm_obj, "Objective", start_idx)

    def plot_admm_var_diff(self, start_idx=0):
        self._plot_helper(self.admm_var, "Variable difference", start_idx)

    def plot_admm_convergence(self, start_idx=0):
        assert self.admm_res is not None, "ADMM residuals not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        rp, rd, ep, ed = self.admm_res

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a in ax:
            a.grid(True)
            a.set_axisbelow(True)
        ax[0].plot(
            np.arange(start_idx, self.admm_iters),
            rp[start_idx:],
            "-",
            c=color_schemes["vib_qual"]["red"],
        )
        ax[0].plot(
            np.arange(start_idx, self.admm_iters),
            ep[start_idx:],
            ":",
            c=color_schemes["vib_qual"]["red"],
            alpha=0.3,
        )
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Primal residual")
        ax[1].plot(
            np.arange(start_idx, self.admm_iters),
            rd[start_idx:],
            "-",
            c=color_schemes["vib_qual"]["red"],
        )
        ax[1].plot(
            np.arange(start_idx, self.admm_iters),
            ed[start_idx:],
            ":",
            c=color_schemes["vib_qual"]["red"],
            alpha=0.3,
        )
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Dual residual")
        fig.tight_layout()

    def plot_P_estimate(self, normalize=True):
        assert self.P_1D is not None, "Estimate not found."
        madimshow(
            self.P_1D, "plasma", axis=False, vmin=0, vmax=1 if normalize else None
        )

    def plot_Q_estimate(self, normalize=True):
        assert self.Q_1D is not None, "Estimate not found."
        madimshow(
            self.Q_1D, "plasma", axis=False, vmin=0, vmax=1 if normalize else None
        )


class NucNormMatrixEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        attributes = [
            "mc",
            "P",
            "Q",
            "pi",
            "admm_err",
            "admm_res",
            "admm_var",
            "admm_iters",
            "Ph",
            "gamma",
            "beta",
            "eps_abs",
            "eps_rel",
            "eps_diff",
            "max_itr",
            "verbose",
        ]
        for attr in attributes:
            setattr(self, attr, None)

    def estimate(self, P_obs, args):
        assert (
            P_obs.ndim == 2 and P_obs.shape[0] == P_obs.shape[1]
        ), "Invalid size of input matrix."
        self.reset()
        Ntot = P_obs.shape[0]
        P_LRM, err_LRM, (rp_LRM, rd_LRM, ep_LRM, ed_LRM), var_diff = (
            ADMMNNLRMFromCondPMF(P_obs, **args).optimize()
        )

        assert not torch.isnan(P_LRM).any(), "Nan values in estimated P."
        assert not torch.isinf(P_LRM).any(), "Inf. values in estimated P."

        mc_LRM = MarkovChainMatrix(P_LRM)
        self._store_results(
            mc_LRM, err_LRM, (rp_LRM, rd_LRM, ep_LRM, ed_LRM), var_diff, P_obs, args
        )

        return self.mc, {
            attr: getattr(self, attr)
            for attr in ["mc", "admm_obj", "admm_res", "admm_var", "admm_iters"]
        }

    def _store_results(self, mc_LRM, err_LRM, res, var_diff, P_obs, args):
        self.mc, self.P, self.Q, self.pi = (
            mc_LRM,
            mc_LRM.P,
            mc_LRM.Q,
            mc_LRM.Q.sum(dim=1, keepdim=True),
        )
        self.admm_obj, self.admm_res, self.admm_var, self.admm_iters = (
            err_LRM,
            res,
            var_diff,
            len(err_LRM),
        )
        self.Ph = P_obs
        for key, value in args.items():
            setattr(self, key, value)

    def plot_admm_objective(self, start_idx=0):
        self._plot_helper(self.admm_obj, "Objective", start_idx)

    def plot_admm_var_diff(self, start_idx=0):
        self._plot_helper(self.admm_var, "Variable difference", start_idx)

    def _plot_helper(self, data, ylabel, start_idx):
        assert data is not None, f"{ylabel} not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.plot(
            np.arange(start_idx, self.admm_iters),
            data[start_idx:],
            "-",
            c=color_schemes["vib_qual"]["red"],
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        fig.tight_layout()

    def plot_P_estimate(self, normalize=True):
        assert self.P is not None, "Estimate not found."
        madimshow(self.P, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)

    def plot_Q_estimate(self, normalize=True):
        assert self.Q is not None, "Estimate not found."
        madimshow(self.Q, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)


class DCLowRankMatrixEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        attributes = [
            "mc",
            "P",
            "Q",
            "pi",
            "admm_err",
            "admm_var",
            "admm_iters",
            "Ph",
            "K",
            "c",
            "alpha",
            "beta",
            "eta",
            "eps_abs",
            "eps_rel",
            "eps_diff",
            "max_itr",
            "admm_itr",
            "verbose",
        ]
        for attr in attributes:
            setattr(self, attr, None)

    def estimate(self, P_obs, args):
        assert (
            P_obs.ndim == 2 and P_obs.shape[0] == P_obs.shape[1]
        ), "Invalid size of input matrix."
        self.reset()
        Ntot = P_obs.shape[0]
        P_LRM, err_LRM, var_diff = ADMMDCLRMFromCondPMF(P_obs, **args).optimize()

        assert not torch.isnan(P_LRM).any(), "Nan values in estimated P."
        assert not torch.isinf(P_LRM).any(), "Inf. values in estimated P."

        mc_LRM = MarkovChainMatrix(P_LRM)
        self._store_results(mc_LRM, err_LRM, var_diff, P_obs, args)

        return self.mc, {
            attr: getattr(self, attr)
            for attr in ["mc", "admm_obj", "admm_var", "admm_iters"]
        }

    def _store_results(self, mc_LRM, err_LRM, var_diff, P_obs, args):
        self.mc, self.P, self.Q, self.pi = (
            mc_LRM,
            mc_LRM.P,
            mc_LRM.Q,
            mc_LRM.Q.sum(dim=1, keepdim=True),
        )
        self.admm_obj, self.admm_var, self.admm_iters = err_LRM, var_diff, len(err_LRM)
        self.Ph = P_obs
        for key, value in args.items():
            setattr(self, key, value)

    def plot_admm_objective(self, start_idx=0):
        self._plot_helper(self.admm_obj, "Objective", start_idx)

    def plot_admm_var_diff(self, start_idx=0):
        self._plot_helper(self.admm_var, "Variable difference", start_idx)

    def _plot_helper(self, data, ylabel, start_idx):
        assert data is not None, f"{ylabel} not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.plot(
            np.arange(start_idx, self.admm_iters),
            data[start_idx:],
            "-",
            c=color_schemes["vib_qual"]["red"],
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        fig.tight_layout()

    def plot_P_estimate(self, normalize=True):
        assert self.P is not None, "Estimate not found."
        madimshow(self.P, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)

    def plot_Q_estimate(self, normalize: bool = True):
        assert self.Q is not None, "Estimate not found."
        madimshow(self.Q, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)


class SpecLowRankMatrixEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        attributes = ["mc", "P", "Q", "pi", "K", "Qh"]
        for attr in attributes:
            setattr(self, attr, None)

    def estimate(self, Q_obs, K, prob_min: float = 0.0):
        assert (
            Q_obs.ndim == 2 and Q_obs.shape[0] == Q_obs.shape[1]
        ), "Invalid size of input matrix."
        self.reset()
        Ntot = Q_obs.shape[0]

        Q_LRM = self._compute_low_rank_matrix(Q_obs, K, prob_min)
        P_LRM, mc_slrm = self._normalize_and_create_mc(Q_LRM, Ntot)

        self._store_results(mc_slrm, Q_obs, K)
        return self.mc, {attr: getattr(self, attr) for attr in ["mc"]}

    def _compute_low_rank_matrix(self, Q_obs, K, prob_min):
        UK, svK, VhK = svds(Q_obs.numpy().astype(float), k=K)
        Q_LRM = torch.maximum(
            torch.FloatTensor(UK @ np.diag(svK) @ VhK),
            prob_min * torch.ones_like(Q_obs),
        )
        return Q_LRM / torch.linalg.norm(Q_LRM, 1)

    def _normalize_and_create_mc(self, Q_LRM, Ntot):
        marg_slrm = Q_LRM.sum(dim=1, keepdim=True)
        P_LRM = Q_LRM / (marg_slrm + (marg_slrm == 0).to(int))
        P_LRM[(marg_slrm == 0).expand_as(P_LRM)] = 1 / Ntot
        mc_slrm = MarkovChainMatrix(P_LRM)
        return P_LRM, mc_slrm

    def _store_results(self, mc_slrm, Q_obs, K):
        self.mc, self.P, self.Q, self.pi = mc_slrm, mc_slrm.P, mc_slrm.Q, mc_slrm.pi
        self.Qh, self.K = Q_obs, K

    def plot_P_estimate(self, normalize: bool = True):
        assert self.P is not None, "Estimate not found."
        madimshow(self.P, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)

    def plot_Q_estimate(self, normalize: bool = True):
        assert self.Q is not None, "Estimate not found."
        madimshow(self.Q, "plasma", axis=False, vmin=0, vmax=1 if normalize else None)
