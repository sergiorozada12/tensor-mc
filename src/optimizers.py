import numpy as np

from tqdm import tqdm
import torch
import tensorly as tl
from tensorly import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import khatri_rao
from scipy.sparse.linalg import svds


tl.set_backend("pytorch")


def soft_thresh(x, l):
    return torch.maximum(torch.abs(x) - l, torch.zeros_like(x)) * x.sign()


def generate_probability_matrix(dims, K):
    matrix = torch.rand(dims, K)
    matrix = matrix / matrix.sum(dim=0, keepdim=True)
    return matrix


def generate_probability_vector(D):
    vector = torch.rand(D)
    vector = vector / vector.sum()
    return vector


def generate_tensor(D, K, p):
    factors = [generate_probability_matrix(p[d], K) for d in range(D)]
    weights = generate_probability_vector(K)
    return cp_to_tensor((weights, factors)), factors, weights


class ADMMLRTFromJointPMF:
    def __init__(
        self,
        Q_obs,
        K: int,
        beta: float = 1.0,
        alpha: float = 1e3,
        lmbda_min: float = 0.0,
        prob_min: float = 0.0,
        eps_abs: float = 1e-6,
        eps_rel: float = 0.0,
        eps_diff: float = 1e-6,
        max_itr: int = 5000,
        min_itr: int = 0,
        verbose: bool = False,
        MARG_CONST: bool = True,
        ACCEL: bool = True,
        disable_tqdm: bool = False,
    ):
        self.Q_obs = Q_obs
        self.K = K
        self.beta = beta
        self.alpha = alpha if not MARG_CONST else 0.0
        self.lmbda_min = min(lmbda_min, 1 / K)
        self.prob_min = prob_min
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.eps_diff = eps_diff
        self.max_itr = max_itr
        self.min_itr = min_itr
        self.verbose = verbose
        self.ACCEL = ACCEL
        self.disable_tqdm = disable_tqdm

        self.D = Q_obs.ndim // 2
        self.Pi = 0.5 * (
            Q_obs.sum(dim=tuple(range(self.D, 2 * self.D)))
            + Q_obs.sum(dim=tuple(range(self.D)))
        )
        self.N = torch.tensor(Q_obs.shape)

        self._initialize_variables()

    def _initialize_variables(self):
        _, self.Q, self.l = generate_tensor(2 * self.D, self.K, self.N)
        self.R = [self.Q[d].clone() for d in range(2 * self.D)]
        self.r = self.l.clone()
        self.u = torch.zeros_like(self.l)
        self.v = torch.zeros(1)
        self.U = [torch.zeros_like(self.Q[d]) for d in range(2 * self.D)]
        self.V = [torch.zeros(self.K) for d in range(2 * self.D)]
        self.Q1 = cp_to_tensor((self.l, self.Q[: self.D]))
        self.Q2 = cp_to_tensor((self.l, self.Q[self.D :]))

        self.rh = self.r.clone()
        self.uh = self.u.clone()
        self.vh = self.v.clone()
        self.Rh = [self.R[d].clone() for d in range(2 * self.D)]
        self.Uh = [self.U[d].clone() for d in range(2 * self.D)]
        self.Vh = [self.V[d].clone() for d in range(2 * self.D)]

        self.M = [unfold(self.Q_obs, mode=d).T for d in range(2 * self.D)]
        self.T = [unfold(self.Pi, mode=d).T for d in range(self.D)] * 2
        self.Ir = torch.linalg.inv(torch.eye(self.K) + 1)
        self.IR = [
            torch.linalg.inv(torch.eye(self.N[d]) + 1) for d in range(2 * self.D)
        ]

        self.t = 1

    def _update_l(self):
        S = khatri_rao(self.Q)
        q = self.Q_obs.reshape((-1))
        S1 = khatri_rao(self.Q[: self.D])
        S2 = khatri_rao(self.Q[self.D :])
        Sdiff = S1 - S2
        pi = self.Pi.reshape((-1))

        self.l_prev = self.l.clone()
        Mat1 = (
            S.T @ S
            + self.beta * torch.eye(self.K)
            + self.alpha * (S1.T @ S1 + S2.T @ S2 + Sdiff.T @ Sdiff)
        )
        Mat2 = S.T @ q + self.alpha * (S1 + S2).T @ pi + self.beta * (self.rh - self.uh)
        self.l = torch.maximum(
            torch.linalg.inv(Mat1) @ Mat2, self.lmbda_min * torch.ones_like(self.l)
        )

    def _update_Q(self):
        self.Q_prev = [self.Q[d].clone() for d in range(2 * self.D)]
        for d in range(2 * self.D):
            S = khatri_rao(self.Q, self.l, skip_matrix=d)
            S1 = (
                khatri_rao(self.Q[: self.D], self.l, skip_matrix=d % self.D)
                if d < self.D
                else khatri_rao(self.Q[self.D :], self.l, skip_matrix=d % self.D)
            )
            S2 = (
                unfold(self.Q2, mode=d % self.D).T
                if d < self.D
                else unfold(self.Q1, mode=d % self.D).T
            )

            Mat1 = (
                self.M[d].T @ S
                + self.beta * (self.Rh[d] - self.Uh[d])
                + self.alpha * (self.T[d] + S2).T @ S1
            )
            Mat2 = S.T @ S + 2 * self.alpha * S1.T @ S1 + self.beta * torch.eye(self.K)
            self.Q[d] = torch.maximum(
                Mat1 @ torch.linalg.inv(Mat2),
                self.prob_min * torch.ones_like(self.Q[d]),
            )

        self.Q1 = cp_to_tensor((self.l, self.Q[: self.D]))
        self.Q2 = cp_to_tensor((self.l, self.Q[self.D :]))

    def _update_auxiliary(self):
        self.rh_prev = self.rh.clone()
        self.r_prev = self.r.clone()
        self.r = self.Ir @ (self.l + self.uh + (1 - self.vh))

        self.Rh_prev = [self.Rh[d].clone() for d in range(2 * self.D)]
        self.R_prev = [self.R[d].clone() for d in range(2 * self.D)]
        self.R = [
            self.IR[d]
            @ (
                self.Q[d]
                + self.Uh[d]
                + torch.outer(torch.ones(self.N[d]), (1 - self.Vh[d]))
            )
            for d in range(2 * self.D)
        ]

    def _update_multipliers(self):
        self.u_prev = self.u.clone()
        self.uh_prev = self.uh.clone()
        self.u = self.uh + self.l - self.r

        self.Uh_prev = [self.Uh[d].clone() for d in range(2 * self.D)]
        self.U_prev = [self.U[d].clone() for d in range(2 * self.D)]
        self.U = [self.Uh[d] + self.Q[d] - self.R[d] for d in range(2 * self.D)]

        self.vh_prev = self.vh.clone()
        self.v_prev = self.v.clone()
        self.v = self.vh + self.r.sum() - 1

        self.Vh_prev = [self.Vh[d].clone() for d in range(2 * self.D)]
        self.V_prev = [self.V[d].clone() for d in range(2 * self.D)]
        self.V = [self.Vh[d] + self.R[d].sum(0) - 1 for d in range(2 * self.D)]

    def _compute_residuals(self):
        rp_l = torch.linalg.norm(self.l - self.l_prev, 2)
        rp_Q = sum(
            [
                torch.linalg.norm(self.Q[d] - self.Q_prev[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        rp_r = torch.linalg.norm(self.r - self.rh_prev, 2)
        rp_R = sum(
            [
                torch.linalg.norm(self.R[d] - self.Rh_prev[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        rd_u = torch.linalg.norm(self.u - self.uh_prev, 2)
        rd_v = torch.linalg.norm(self.v - self.vh_prev, 2)
        rd_U = sum(
            [torch.linalg.norm(self.U[d] - self.Uh_prev[d]) for d in range(2 * self.D)]
        )
        rd_V = sum(
            [torch.linalg.norm(self.V[d] - self.Vh_prev[d]) for d in range(2 * self.D)]
        )

        return rp_l, rp_Q, rp_r, rp_R, rd_u, rd_v, rd_U, rd_V

    def _compute_tolerances(self):
        ep_u = self.eps_abs * np.sqrt(self.K) + self.eps_rel * torch.maximum(
            torch.linalg.norm(self.l), torch.linalg.norm(self.r)
        )
        ep_v = self.eps_abs + self.eps_rel * torch.maximum(
            torch.abs(self.r.sum()), torch.tensor(1)
        )
        ep_U = sum(
            [
                self.eps_abs * np.sqrt(self.Q[d].numel())
                + self.eps_rel
                * torch.maximum(
                    torch.linalg.norm(self.Q[d], "fro"),
                    torch.linalg.norm(self.R[d], "fro"),
                )
                for d in range(2 * self.D)
            ]
        )
        ep_V = sum(
            [
                self.eps_abs * np.sqrt(self.K)
                + self.eps_rel
                * torch.maximum(
                    torch.linalg.norm(self.R[d].sum(0)), torch.tensor(np.sqrt(self.K))
                )
                for d in range(2 * self.D)
            ]
        )

        ed_u = self.eps_abs * np.sqrt(self.K) + self.eps_rel * torch.linalg.norm(
            self.u, 2
        )
        ed_v = self.eps_abs + self.eps_rel * torch.linalg.norm(self.v, 2)
        ed_U = sum(
            [
                self.eps_abs * np.sqrt(self.Q[d].numel())
                + self.eps_rel * torch.linalg.norm(self.U[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        ed_V = sum(
            [
                self.eps_abs * np.sqrt(self.K)
                + self.eps_rel
                * torch.linalg.norm(torch.outer(self.V[d], torch.ones(self.K)), "fro")
                for d in range(2 * self.D)
            ]
        )

        return (
            torch.tensor([ep_u + ep_v + ep_U + ep_V]).float(),
            torch.tensor([ed_u + ed_v + ed_U + ed_V]).float(),
        )

    def optimize(self):
        var_diff = torch.tensor([])
        err_hat = torch.tensor([])
        res_pri = torch.tensor([])
        res_dua = torch.tensor([])
        eps_pri = torch.tensor([])
        eps_dua = torch.tensor([])
        co_res = torch.tensor([0])

        for itr in tqdm(range(self.max_itr), disable=self.disable_tqdm):
            if self.ACCEL:
                t_prev = self.t
                self.t = 0.5 * (1 + np.sqrt(1 + 4 * self.t**2))
                w = (t_prev - 1) / self.t
            else:
                w = 0.0

            self._update_l()
            self._update_Q()
            self._update_auxiliary()
            self._update_multipliers()

            rp_l, rp_Q, rp_r, rp_R, rd_u, rd_v, rd_U, rd_V = self._compute_residuals()
            ep, ed = self._compute_tolerances()

            rp = torch.tensor([rd_u + rd_v + rd_U + rd_V]).float()
            rd = torch.tensor([rp_r + rp_R])
            vd = torch.tensor(
                [torch.tensor([rp_l, rp_Q, rp_r, rp_R, rd_u, rd_v, rd_U, rd_V]).mean()]
            ).float()

            cr = torch.tensor([rp_r + rp_R + rd_u + rd_v + rd_U + rd_V]).float()
            co_res = torch.cat((co_res, cr), 0)
            if co_res[-1] >= co_res[-2]:
                w = 0.0

            # Update auxiliary variables if accelerating
            self.rh = self.r + w * (self.r - self.r_prev)
            self.Rh = [
                self.R[d] + w * (self.R[d] - self.R_prev[d]) for d in range(2 * self.D)
            ]
            self.uh = self.u + w * (self.u - self.u_prev)
            self.vh = self.v + w * (self.v - self.v_prev)
            self.Uh = [
                self.U[d] + w * (self.U[d] - self.U_prev[d]) for d in range(2 * self.D)
            ]
            self.Vh = [
                self.V[d] + w * (self.V[d] - self.V_prev[d]) for d in range(2 * self.D)
            ]

            Q_est = cp_to_tensor((self.l, self.Q))
            eh = torch.tensor([torch.norm(Q_est - self.Q_obs, "fro")]).float()
            err_hat = torch.cat((err_hat, eh), 0)

            res_pri = torch.cat((res_pri, rp), 0)
            res_dua = torch.cat((res_dua, rd), 0)
            eps_pri = torch.cat((eps_pri, ep), 0)
            eps_dua = torch.cat((eps_dua, ed), 0)
            var_diff = torch.cat((var_diff, vd), 0)

            if itr >= self.min_itr and (
                ((res_pri <= eps_pri).sum() > 0 and (res_dua <= eps_dua).sum() > 0)
                or vd.item() < self.eps_diff
            ):
                break

        if self.verbose and itr < self.max_itr - 1:
            print("Terminated early")

        return (
            Q_est,
            (self.l, self.Q),
            err_hat,
            (res_pri, res_dua, eps_pri, eps_dua),
            var_diff,
        )


class ADMMDCLRMFromCondPMF:
    def __init__(
        self,
        P_obs: torch.Tensor,
        K: int,
        c: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 1e-6,
        prob_min: float = 0.0,
        eps_abs: float = 1e-6,
        eps_rel: float = 0.0,
        eps_diff: float = 1e-6,
        max_itr: int = 300,
        min_itr: int = 0,
        admm_itr: int = 500,
        verbose: bool = False,
        disable_tqdm: bool = False,
    ):
        self.P_obs = P_obs
        self.K = K
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.prob_min = prob_min
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.eps_diff = eps_diff
        self.max_itr = max_itr
        self.min_itr = min_itr
        self.admm_itr = admm_itr
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.p = P_obs.shape[0]

    def _initialize_variables(self) -> None:
        self.P = generate_probability_matrix(self.p, self.p)
        self.R = self.P.clone()
        self.S = self.P.clone()
        self.U = torch.zeros_like(self.P)
        self.v = torch.zeros(self.p)
        self.IR = torch.inverse(torch.eye(self.p) + 1)

    def _compute_Y(self) -> torch.Tensor:
        if not torch.allclose(self.P, torch.zeros_like(self.P)):
            try:
                up, svp, vhp = svds(self.P.numpy().astype(float), k=self.K)
                return torch.FloatTensor(up @ vhp)
            except:
                try:
                    up, svp, vhp = torch.linalg.svd(self.P)
                    return up[:, : self.K] @ vhp[: self.K, :]
                except:
                    up, svp, vhp = np.linalg.svd(self.P.numpy().astype(float))
                    return torch.FloatTensor(up[:, : self.K] @ vhp[: self.K, :])
        return self.P.clone()

    def _update_R(self, Y: torch.Tensor) -> None:
        prox_arg = (
            self.P_obs
            + self.alpha * self.P
            + self.c * Y
            + self.beta * (self.S - self.U)
        ) / (self.alpha + self.beta + 1)
        if not torch.allclose(prox_arg, torch.zeros_like(prox_arg)):
            try:
                UL, svs, URh = torch.linalg.svd(prox_arg)
            except:
                UL, svs, URh = np.linalg.svd(prox_arg.numpy().astype(float))
                UL = torch.FloatTensor(UL)
                svs = torch.FloatTensor(svs)
                URh = torch.FloatTensor(URh)
            trun_svs = soft_thresh(svs, self.c / (self.alpha + self.beta + 1))
            self.R = UL @ torch.diag(trun_svs) @ URh
        else:
            self.R = prox_arg.clone()

    def _update_S(self) -> None:
        self.S = torch.maximum(
            (self.P + self.U + torch.outer(1 - self.v, torch.ones(self.p))) @ self.IR,
            self.prob_min * torch.ones_like(self.S),
        )

    def _compute_residuals(
        self,
        R_prev: torch.Tensor,
        S_prev: torch.Tensor,
        U_prev: torch.Tensor,
        v_prev: torch.Tensor,
    ) -> tuple:
        vd = torch.tensor(
            [
                torch.tensor(
                    [
                        torch.norm(R_prev - self.R, "fro"),
                        torch.norm(S_prev - self.S, "fro"),
                        torch.norm(U_prev - self.U, "fro"),
                        torch.norm(v_prev - self.v, 2),
                    ]
                ).max()
            ]
        ).float()

        rp = torch.tensor(
            [torch.norm(self.R - self.S, "fro") + torch.norm(self.R.sum(1) - 1, 2)]
        ).float()
        rd = torch.tensor(
            [
                self.beta * torch.norm(self.S - S_prev, "fro")
                + self.beta * torch.norm(self.R - R_prev, "fro")
                + self.beta * torch.norm(self.R.sum(1) - R_prev.sum(1), 2)
            ]
        ).float()

        return vd, rp, rd

    def _compute_tolerances(self) -> tuple:
        ep = torch.tensor(
            [
                self.eps_abs * np.sqrt(self.R.numel())
                + self.eps_rel * torch.maximum(torch.norm(self.R), torch.norm(self.S))
                + self.eps_abs * np.sqrt(self.p)
                + self.eps_rel
                * torch.maximum(
                    torch.norm(self.R.sum(1)), torch.norm(torch.ones(self.p))
                )
            ]
        ).float()

        ed = torch.tensor(
            [
                self.eps_abs * np.sqrt(self.R.numel())
                + self.eps_rel * torch.norm(self.beta * self.U)
                + self.eps_abs * np.sqrt(self.p)
                + self.eps_rel
                * torch.norm(self.beta * torch.outer(self.v, torch.ones(self.p)))
            ]
        ).float()

        return ep, ed

    def optimize(self) -> tuple:
        self._initialize_variables()
        err_hat = torch.tensor([])
        var_diff_out = torch.tensor([])

        for itr1 in tqdm(range(self.max_itr), disable=self.disable_tqdm):
            Y = self._compute_Y()

            self.R = self.P.clone()
            self.S = self.P.clone()
            self.U = torch.zeros_like(self.R)
            self.v = torch.zeros(self.p)

            res_pri = torch.tensor([])
            res_dua = torch.tensor([])
            eps_pri = torch.tensor([])
            eps_dua = torch.tensor([])
            var_diff_in = torch.tensor([])

            for itr2 in range(self.admm_itr):
                R_prev = self.R.clone()
                S_prev = self.S.clone()
                U_prev = self.U.clone()
                v_prev = self.v.clone()

                self._update_R(Y)
                self._update_S()
                self.U = self.U + self.R - self.S
                self.v = self.v + self.S.sum(1) - 1

                vd, rp, rd = self._compute_residuals(R_prev, S_prev, U_prev, v_prev)
                ep, ed = self._compute_tolerances()

                res_pri = torch.cat((res_pri, rp), 0)
                eps_pri = torch.cat((eps_pri, ep), 0)
                res_dua = torch.cat((res_dua, rd), 0)
                eps_dua = torch.cat((eps_dua, ed), 0)
                var_diff_in = torch.cat((var_diff_in, vd), 0)

                if (
                    (res_pri <= eps_pri).sum() > 0 and (res_dua <= eps_dua).sum() > 0
                ) or (vd.item() < self.eps_diff):
                    break

            P_prev = self.P.clone()
            self.P = self.R.clone()

            P_est = self.P.clone()
            P_est = torch.maximum(P_est, self.prob_min * torch.ones_like(P_est))
            P_est = P_est / (
                P_est.sum(dim=1, keepdim=True)
                + (P_est.sum(dim=1, keepdim=True) == 0).to(int)
            )

            eh = torch.tensor([torch.norm(self.P - self.P_obs, "fro")]).float()
            err_hat = torch.cat((err_hat, eh), 0)

            vdo = torch.tensor([torch.norm(self.P - P_prev, "fro")]).float()
            var_diff_out = torch.cat((var_diff_out, vdo), 0)

            if itr1 >= self.min_itr and vdo < self.eta:
                break

        if self.verbose and itr1 < self.max_itr - 1:
            print("Terminated early")

        return P_est, err_hat, var_diff_out


class ADMMNNLRMFromCondPMF:
    def __init__(
        self,
        P_obs,
        gamma: float = 0.1,
        beta: float = 1.0,
        prob_min: float = 0.0,
        eps_abs: float = 1e-6,
        eps_rel: float = 0.0,
        eps_diff: float = 1e-6,
        max_itr: int = 5000,
        min_itr: int = 0,
        verbose: bool = False,
        disable_tqdm: bool = False,
    ):
        self.P_obs = P_obs
        self.gamma = gamma
        self.beta = beta
        self.prob_min = prob_min
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.eps_diff = eps_diff
        self.max_itr = max_itr
        self.min_itr = min_itr
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm

        self.p = P_obs.shape[0]
        self._initialize_variables()

    def _initialize_variables(self):
        self.P = generate_probability_matrix(self.p, self.p)
        self.Q = self.P.clone()
        self.U = torch.zeros_like(self.P)
        self.V = torch.zeros(self.p)
        self.IQ = torch.inverse(torch.eye(self.p) + torch.ones((self.p, self.p)))

    def _update_P(self):
        S = (self.P_obs + self.beta * (self.Q - self.U)) / (1 + self.beta)
        if not torch.allclose(S, torch.zeros_like(S)):
            try:
                UL, svs, UR = torch.linalg.svd(S)
            except:
                UL, svs, UR = np.linalg.svd(S.numpy().astype(float))
                UL = torch.FloatTensor(UL)
                svs = torch.FloatTensor(svs)
                UR = torch.FloatTensor(UR)
            self.P = (
                UL @ torch.diag(soft_thresh(svs, self.gamma / (1 + self.beta))) @ UR
            )
        else:
            self.P = S.clone()
        self.P = torch.maximum(self.P, self.prob_min * torch.ones_like(self.P))

    def _update_Q(self):
        self.Q = (
            self.P + self.U + torch.outer(1 - self.V, torch.ones(self.p))
        ) @ self.IQ

    def _compute_residuals(self, Q_prev):
        rp_U = torch.norm(self.P - self.Q, "fro")
        rp_V = torch.norm(self.Q.sum(1) - 1, 2)
        rp = torch.tensor([rp_U + rp_V]).float()
        rd = torch.tensor([2 * self.beta * torch.norm(self.Q - Q_prev, "fro")]).float()
        return rp, rd

    def _compute_tolerances(self):
        ep_U = self.eps_abs * np.sqrt(self.P.numel()) + self.eps_rel * torch.maximum(
            torch.norm(self.P), torch.norm(self.Q)
        )
        ep_V = self.eps_abs * np.sqrt(self.p) + self.eps_rel * torch.maximum(
            torch.norm(self.Q.sum(1)), torch.norm(torch.ones(self.p))
        )
        ep = torch.tensor([ep_U + ep_V]).float()

        ed_U = self.eps_abs * np.sqrt(self.P.numel()) + self.eps_rel * torch.norm(
            self.beta * self.U
        )
        ed_V = self.eps_abs * np.sqrt(self.p) + self.eps_rel * torch.norm(
            self.beta * torch.outer(self.V, torch.ones(self.p))
        )
        ed = torch.tensor([ed_U + ed_V]).float()
        return ep, ed

    def _compute_var_diff(self, P_prev, Q_prev, U_prev, V_prev):
        return torch.tensor(
            [
                torch.tensor(
                    [
                        torch.norm(self.Q - Q_prev, "fro"),
                        torch.norm(self.P - P_prev, "fro"),
                        torch.norm(self.U - U_prev, "fro"),
                        torch.norm(self.V - V_prev, "fro"),
                    ]
                ).max()
            ]
        ).float()

    def optimize(self):
        var_diff = torch.tensor([])
        err_hat = torch.tensor([])
        res_pri = torch.tensor([])
        res_dua = torch.tensor([])
        eps_pri = torch.tensor([])
        eps_dua = torch.tensor([])

        for itr in tqdm(range(self.max_itr), disable=self.disable_tqdm):
            P_prev = self.P.clone()
            Q_prev = self.Q.clone()
            U_prev = self.U.clone()
            V_prev = self.V.clone()

            self._update_P()
            self._update_Q()
            self.U = self.U + self.P - self.Q
            self.V = self.V + self.Q.sum(1) - 1

            vd = self._compute_var_diff(P_prev, Q_prev, U_prev, V_prev)
            rp, rd = self._compute_residuals(Q_prev)
            ep, ed = self._compute_tolerances()

            res_pri = torch.cat((res_pri, rp), 0)
            res_dua = torch.cat((res_dua, rd), 0)
            eps_pri = torch.cat((eps_pri, ep), 0)
            eps_dua = torch.cat((eps_dua, ed), 0)

            P_est = self.P.clone()
            P_est = torch.maximum(P_est, self.prob_min * torch.ones_like(P_est))
            P_est = P_est / (
                P_est.sum(dim=1, keepdim=True)
                + (P_est.sum(dim=1, keepdim=True) == 0).to(int)
            )
            eh = torch.tensor([torch.norm(self.P - self.P_obs, "fro")]).float()
            err_hat = torch.cat((err_hat, eh), 0)

            var_diff = torch.cat((var_diff, vd), 0)

            if itr >= self.min_itr and (
                ((res_pri <= eps_pri).sum() > 0 and (res_dua <= eps_dua).sum() > 0)
                or (vd.item() < self.eps_diff)
            ):
                break

        if self.verbose and itr < self.max_itr - 1:
            print("Terminated early")

        return P_est, err_hat, (res_pri, res_dua, eps_pri, eps_dua), var_diff


class ADMMLRTFromJointPMFMask:
    def __init__(
        self,
        Q_obs,
        K: int,
        Mask=None,
        beta: float = 1.0,
        alpha: float = 1e3,
        lmbda_min: float = 0.0,
        prob_min: float = 0.0,
        eps_abs: float = 1e-6,
        eps_rel: float = 0.0,
        eps_diff: float = 1e-6,
        max_itr: int = 5000,
        min_itr: int = 0,
        verbose: bool = False,
        MARG_CONST: bool = True,
        ACCEL: bool = True,
        disable_tqdm: bool = False,
    ):
        self.Q_obs = Q_obs
        self.K = K
        self.Mask = torch.ones_like(Q_obs) if Mask is None else Mask
        self.beta = beta
        self.alpha = 0.0 if not MARG_CONST else alpha
        self.lmbda_min = min(lmbda_min, 1 / K)
        self.prob_min = prob_min
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.eps_diff = eps_diff
        self.max_itr = max_itr
        self.min_itr = min_itr
        self.verbose = verbose
        self.ACCEL = ACCEL
        self.disable_tqdm = disable_tqdm

        self.D = Q_obs.ndim // 2
        self.R = 0.5 * (
            Q_obs.sum(dim=tuple(range(self.D, 2 * self.D)))
            + Q_obs.sum(dim=tuple(range(self.D)))
        )
        self.N = torch.tensor(Q_obs.shape)
        self.t = 1

        self._initialize_variables()
        self._initialize_matrices()

    def _initialize_variables(self):
        _, self.Q, self.l = generate_tensor(2 * self.D, self.K, self.N)
        self.S = [self.Q[d].clone() for d in range(2 * self.D)]
        self.s = self.l.clone()
        self.u = torch.zeros_like(self.l)
        self.v = torch.zeros(1)
        self.U = [torch.zeros_like(self.Q[d]) for d in range(2 * self.D)]
        self.V = [torch.zeros(self.K) for d in range(2 * self.D)]
        self.Q1 = cp_to_tensor((self.l, self.Q[: self.D]))
        self.Q2 = cp_to_tensor((self.l, self.Q[self.D :]))
        self.sh = self.s.clone()
        self.uh = self.u.clone()
        self.vh = self.v.clone()
        self.Sh = [self.S[d].clone() for d in range(2 * self.D)]
        self.Uh = [self.U[d].clone() for d in range(2 * self.D)]
        self.Vh = [self.V[d].clone() for d in range(2 * self.D)]

    def _initialize_matrices(self):
        self.Qd_obs = [unfold(self.Q_obs, mode=d).T for d in range(2 * self.D)]
        self.Maskd = [unfold(self.Mask, mode=d).T for d in range(2 * self.D)]
        self.Rd = [unfold(self.R, mode=d).T for d in range(self.D)] * 2
        self.Is = torch.linalg.inv(torch.eye(self.K) + 1)
        self.IS = [
            torch.linalg.inv(torch.eye(self.N[d]) + 1) for d in range(2 * self.D)
        ]

    def _update_l(self):
        T = khatri_rao(self.Q)
        q_obs = self.Q_obs.reshape((-1))
        mask = self.Mask.reshape((-1))
        T1 = khatri_rao(self.Q[: self.D])
        T2 = khatri_rao(self.Q[self.D :])
        Tdiff = T1 - T2
        r = self.R.reshape((-1))

        Tm = T[mask == 1]
        qm = q_obs[mask == 1]

        Mat1 = (
            Tm.T @ Tm
            + self.beta * torch.eye(self.K)
            + self.alpha * (T1.T @ T1 + T2.T @ T2 + Tdiff.T @ Tdiff)
        )
        Mat2 = (
            Tm.T @ qm + self.alpha * (T1 + T2).T @ r + self.beta * (self.sh - self.uh)
        )
        self.l = torch.maximum(
            torch.linalg.inv(Mat1) @ Mat2, self.lmbda_min * torch.ones_like(self.l)
        )

    def _update_Q(self):
        for d in range(2 * self.D):
            Td = khatri_rao(self.Q, self.l, skip_matrix=d)
            T1d = (
                khatri_rao(self.Q[: self.D], self.l, skip_matrix=d % self.D)
                if d < self.D
                else khatri_rao(self.Q[self.D :], self.l, skip_matrix=d % self.D)
            )
            T2d = (
                unfold(self.Q2, mode=d % self.D).T
                if d < self.D
                else unfold(self.Q1, mode=d % self.D).T
            )

            maskd = self.Maskd[d].T.reshape(-1)
            Td_kron = torch.kron(torch.eye(self.N[d % self.D]), Td)
            qd_obs = self.Qd_obs[d].T.reshape(-1)
            Tdm = Td_kron[maskd == 1]
            qdm = qd_obs[maskd == 1]

            T1d_kron = torch.kron(torch.eye(self.N[d % self.D]), T1d)
            t2d = T2d.T.reshape(-1)
            rd = self.Rd[d].T.reshape(-1)

            Mat1 = (
                Tdm.T @ Tdm
                + self.beta * torch.eye(self.N[d % self.D] * self.K)
                + 2 * self.alpha * T1d_kron.T @ T1d_kron
            )
            Mat2 = (
                Tdm.T @ qdm
                + self.beta * (self.Sh[d] - self.Uh[d]).reshape(-1)
                + self.alpha * T1d_kron.T @ (t2d + rd)
            )
            self.Q[d] = torch.maximum(
                torch.linalg.inv(Mat1) @ Mat2,
                self.prob_min * torch.ones(self.N[d % self.D] * self.K),
            ).reshape(self.Q[d].shape)
        self.Q1 = cp_to_tensor((self.l, self.Q[: self.D]))
        self.Q2 = cp_to_tensor((self.l, self.Q[self.D :]))

    def _update_auxiliary(self, w: float):
        self.sh = self.s + w * (self.s - self.s_prev)
        self.Sh = [
            self.S[d] + w * (self.S[d] - self.S_prev[d]) for d in range(2 * self.D)
        ]
        self.uh = self.u + w * (self.u - self.u_prev)
        self.vh = self.v + w * (self.v - self.v_prev)
        self.Uh = [
            self.U[d] + w * (self.U[d] - self.U_prev[d]) for d in range(2 * self.D)
        ]
        self.Vh = [
            self.V[d] + w * (self.V[d] - self.V_prev[d]) for d in range(2 * self.D)
        ]

    def _compute_residuals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rp_l = torch.linalg.norm(self.l - self.l_prev, 2)
        rp_Q = sum(
            [
                torch.linalg.norm(self.Q[d] - self.Q_prev[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        rp_s = torch.linalg.norm(self.s - self.sh_prev, 2)
        rp_S = sum(
            [
                torch.linalg.norm(self.S[d] - self.Sh_prev[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        rd_u = torch.linalg.norm(self.u - self.uh_prev, 2)
        rd_v = torch.linalg.norm(self.v - self.vh_prev, 2)
        rd_U = sum(
            [torch.linalg.norm(self.U[d] - self.Uh_prev[d]) for d in range(2 * self.D)]
        )
        rd_V = sum(
            [torch.linalg.norm(self.V[d] - self.Vh_prev[d]) for d in range(2 * self.D)]
        )

        rp = torch.tensor([rd_u + rd_v + rd_U + rd_V]).float()
        rd = torch.tensor([rp_s + rp_S])
        vd = torch.tensor(
            [torch.tensor([rp_l, rp_Q, rp_s, rp_S, rd_u, rd_v, rd_U, rd_V]).mean()]
        ).float()

        return rp, rd, vd, rp_l, rp_Q, rp_s, rp_S, rd_u, rd_v, rd_U, rd_V

    def _compute_tolerances(self) -> tuple[torch.Tensor, torch.Tensor]:
        ep_u = self.eps_abs * np.sqrt(self.K) + self.eps_rel * torch.maximum(
            torch.linalg.norm(self.l), torch.linalg.norm(self.s)
        )
        ep_v = self.eps_abs + self.eps_rel * torch.maximum(
            torch.abs(self.s.sum()), torch.tensor(1)
        )
        ep_U = sum(
            [
                self.eps_abs * np.sqrt(self.Q[d].numel())
                + self.eps_rel
                * torch.maximum(
                    torch.linalg.norm(self.Q[d], "fro"),
                    torch.linalg.norm(self.S[d], "fro"),
                )
                for d in range(2 * self.D)
            ]
        )
        ep_V = sum(
            [
                self.eps_abs * np.sqrt(self.K)
                + self.eps_rel
                * torch.maximum(
                    torch.linalg.norm(self.S[d].sum(0)), torch.tensor(np.sqrt(self.K))
                )
                for d in range(2 * self.D)
            ]
        )
        ep = torch.tensor([ep_u + ep_v + ep_U + ep_V]).float()

        ed_u = self.eps_abs * np.sqrt(self.K) + self.eps_rel * torch.linalg.norm(
            self.u, 2
        )
        ed_v = self.eps_abs + self.eps_rel * torch.linalg.norm(self.v, 2)
        ed_U = sum(
            [
                self.eps_abs * np.sqrt(self.Q[d].numel())
                + self.eps_rel * torch.linalg.norm(self.U[d], "fro")
                for d in range(2 * self.D)
            ]
        )
        ed_V = sum(
            [
                self.eps_abs * np.sqrt(self.K)
                + self.eps_rel
                * torch.linalg.norm(torch.outer(self.V[d], torch.ones(self.K)), "fro")
                for d in range(2 * self.D)
            ]
        )
        ed = torch.tensor([ed_u + ed_v + ed_U + ed_V]).float()

        return ep, ed

    def optimize(self):
        var_diff = torch.tensor([])
        err_hat = torch.tensor([])
        res_pri = torch.tensor([])
        res_dua = torch.tensor([])
        eps_pri = torch.tensor([])
        eps_dua = torch.tensor([])
        co_res = torch.tensor([0])

        for itr in tqdm(range(self.max_itr), disable=self.disable_tqdm):
            if self.ACCEL:
                t_prev = self.t
                self.t = 0.5 * (1 + np.sqrt(1 + 4 * self.t**2))
                w = (t_prev - 1) / self.t
            else:
                w = 0.0

            self.l_prev = self.l.clone()
            self.Q_prev = [self.Q[d].clone() for d in range(2 * self.D)]
            self.s_prev = self.s.clone()
            self.S_prev = [self.S[d].clone() for d in range(2 * self.D)]
            self.u_prev = self.u.clone()
            self.v_prev = self.v.clone()
            self.U_prev = [self.U[d].clone() for d in range(2 * self.D)]
            self.V_prev = [self.V[d].clone() for d in range(2 * self.D)]
            self.sh_prev = self.sh.clone()
            self.Sh_prev = [self.Sh[d].clone() for d in range(2 * self.D)]
            self.uh_prev = self.uh.clone()
            self.vh_prev = self.vh.clone()
            self.Uh_prev = [self.Uh[d].clone() for d in range(2 * self.D)]
            self.Vh_prev = [self.Vh[d].clone() for d in range(2 * self.D)]

            self._update_l()
            self._update_Q()

            self.s = self.Is @ (self.l + self.uh + (1 - self.vh))
            self.S = [
                self.IS[d]
                @ (
                    self.Q[d]
                    + self.Uh[d]
                    + torch.outer(torch.ones(self.N[d]), (1 - self.Vh[d]))
                )
                for d in range(2 * self.D)
            ]

            self.u = self.uh + self.l - self.s
            self.U = [self.Uh[d] + self.Q[d] - self.S[d] for d in range(2 * self.D)]

            self.v = self.vh + self.s.sum() - 1
            self.V = [self.Vh[d] + self.S[d].sum(0) - 1 for d in range(2 * self.D)]

            rp, rd, vd, rp_l, rp_Q, rp_s, rp_S, rd_u, rd_v, rd_U, rd_V = (
                self._compute_residuals()
            )
            ep, ed = self._compute_tolerances()

            cr = torch.tensor([rp_s + rp_S + rd_u + rd_v + rd_U + rd_V]).float()
            co_res = torch.cat((co_res, cr), 0)
            if co_res[-1] >= co_res[-2]:
                w = 0.0

            self._update_auxiliary(w)

            Q_est = cp_to_tensor((self.l, self.Q))
            eh = torch.tensor([torch.norm(Q_est - self.Q_obs, "fro")]).float()

            err_hat = torch.cat((err_hat, eh), 0)
            res_pri = torch.cat((res_pri, rp), 0)
            res_dua = torch.cat((res_dua, rd), 0)
            eps_pri = torch.cat((eps_pri, ep), 0)
            eps_dua = torch.cat((eps_dua, ed), 0)
            var_diff = torch.cat((var_diff, vd), 0)

            if itr >= self.min_itr and (
                ((res_pri <= eps_pri).sum() > 0 and (res_dua <= eps_dua).sum() > 0)
                or vd.item() < self.eps_diff
            ):
                break

        if self.verbose and itr < self.max_itr - 1:
            print("Terminated early")

        return (
            Q_est,
            (self.l, self.Q),
            err_hat,
            (res_pri, res_dua, eps_pri, eps_dua),
            var_diff,
        )
