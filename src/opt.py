import numpy as np

from tqdm import tqdm
import torch
import tensorly as tl
from tensorly import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import khatri_rao
from scipy.sparse.linalg import svds

tl.set_backend('pytorch')

def soft_thresh(x,l):
    return torch.maximum(torch.abs(x)-l,torch.zeros_like(x)) * x.sign()

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

def admm_lrt_from_jointpmf( Q_obs,
                            K:int, 
                            beta:float=1., 
                            alpha:float=1e3,
                            lmbda_min:float=0.,
                            prob_min:float=0.,
                            eps_abs:float=1e-6, 
                            eps_rel:float=0., 
                            eps_diff:float=1e-6,
                            max_itr:int=5000, 
                            min_itr:int=0,
                            verbose:bool=False,
                            MARG_CONST:bool=True,
                            ACCEL:bool=True,
                            disable_tqdm:bool=False
                            ):
    # ------------------------------
    D = Q_obs.ndim // 2
    Pi = .5 * ( Q_obs.sum(dim=tuple(range(D,2*D))) + Q_obs.sum(dim=tuple(range(D))) )
    N = torch.tensor(Q_obs.shape)
    t = 1

    lmbda_min = np.min([lmbda_min,1/K])

    if MARG_CONST:
        alpha = 0.

    # Initialize variables
    _, Q, l = generate_tensor(2*D,K,N)
    R = [Q[d].clone() for d in range(2*D)]
    r = l.clone()
    u = torch.zeros_like(l)
    v = torch.zeros(1)
    U = [torch.zeros_like(Q[d]) for d in range(2*D)]
    V = [torch.zeros(K) for d in range(2*D)]
    Q1 = cp_to_tensor((l,Q[:D]))
    Q2 = cp_to_tensor((l,Q[D:]))
    rh = r.clone()
    uh = u.clone()
    vh = v.clone()
    Rh = [R[d].clone() for d in range(2*D)]
    Uh = [U[d].clone() for d in range(2*D)]
    Vh = [V[d].clone() for d in range(2*D)]

    var_diff = torch.tensor([])
    err_hat = torch.tensor([]) # Objective
    res_pri = torch.tensor([]); res_dua = torch.tensor([])
    eps_pri = torch.tensor([]); eps_dua = torch.tensor([])

    co_res = torch.tensor([0])

    M = [unfold(Q_obs,mode=d).T for d in range(2*D)]
    T = [unfold(Pi,mode=d).T for d in range(D)]*2
    Ir = torch.linalg.inv( torch.eye(K) + 1 )
    IR = [ torch.linalg.inv( torch.eye(N[d])+1 ) for d in range(2*D) ]

    for itr in tqdm(range(max_itr),disable=disable_tqdm):
        if ACCEL:
            # t update
            t_prev = t
            t = .5 * (1 + np.sqrt(1 + 4*t**2))
            w = (t_prev-1)/t
        else:
            w = 0.

        # l update
        S = khatri_rao(Q)
        q = Q_obs.reshape((-1))
        S1 = khatri_rao(Q[:D])
        S2 = khatri_rao(Q[D:])
        Sdiff = S1-S2
        pi = Pi.reshape((-1))

        l_prev = l.clone()
        Mat1 = S.T@S + beta*torch.eye(K) + alpha*( S1.T@S1 + S2.T@S2 + Sdiff.T@Sdiff )
        Mat2 = S.T@q + alpha * ( S1 + S2 ).T@pi + beta*( rh-uh )
        l = torch.maximum( torch.linalg.inv(Mat1) @ Mat2, lmbda_min * torch.ones_like(l))
        # l = torch.maximum( torch.linalg.inv(Mat1) @ Mat2, torch.zeros_like(l))

        # Q update
        Q_prev = [Q[d].clone() for d in range(2*D)]
        for d in range(2*D):
            S = khatri_rao(Q,l,skip_matrix=d)
            S1 = khatri_rao(Q[:D],l,skip_matrix=d%D) if d<D else khatri_rao(Q[D:],l,skip_matrix=d%D)
            S2 = unfold(Q2,mode=d%D).T if d<D else unfold(Q1,mode=d%D).T

            Mat1 = M[d].T@S + beta*(Rh[d]-Uh[d]) + alpha * ( T[d]+S2 ).T@S1
            Mat2 = S.T@S + 2*alpha*S1.T@S1 + beta*torch.eye(K)
            Q[d] = torch.maximum(Mat1@torch.linalg.inv(Mat2), prob_min * torch.ones_like(Q[d]))
            # Q[d] = torch.maximum(Mat1@torch.linalg.inv(Mat2), torch.zeros_like(Q[d]))

        Q1 = cp_to_tensor((l,Q[:D]))
        Q2 = cp_to_tensor((l,Q[D:]))

        # r update
        rh_prev = rh.clone()
        r_prev = r.clone()
        r = Ir @ ( l+uh + (1-vh) )

        # R update
        Rh_prev = [Rh[d].clone() for d in range(2*D)]
        R_prev = [R[d].clone() for d in range(2*D)]
        R = [ IR[d] @ (Q[d]+Uh[d] + torch.outer( torch.ones(N[d]), (1-Vh[d]) )) for d in range(2*D)]

        # u update
        u_prev = u.clone()
        uh_prev = uh.clone()
        u = uh + l - r

        # U update
        Uh_prev = [Uh[d].clone() for d in range(2*D)]
        U_prev = [U[d].clone() for d in range(2*D)]
        U = [Uh[d] + Q[d] - R[d] for d in range(2*D)]

        # v update
        vh_prev = vh.clone()
        v_prev = v.clone()
        v = vh + r.sum()-1

        # V update
        Vh_prev = [Vh[d].clone() for d in range(2*D)]
        V_prev = [V[d].clone() for d in range(2*D)]
        V = [Vh[d] + R[d].sum(0)-1 for d in range(2*D)]

        # Residuals
        rp_l = torch.linalg.norm(l - l_prev,2)
        rp_Q = sum([torch.linalg.norm(Q[d]-Q_prev[d],'fro') for d in range(2*D)])
        rp_r = torch.linalg.norm(r - rh_prev,2)
        rp_R = sum([torch.linalg.norm(R[d] - Rh_prev[d],'fro') for d in range(2*D)])
        rd_u = torch.linalg.norm(u - uh_prev,2)
        rd_v = torch.linalg.norm(v - vh_prev,2)
        rd_U = sum([torch.linalg.norm(U[d] - Uh_prev[d]) for d in range(2*D)])
        rd_V = sum([torch.linalg.norm(V[d] - Vh_prev[d]) for d in range(2*D)])

        rp = torch.tensor([rd_u+rd_v+rd_U+rd_V]).float()
        # rd = torch.tensor([rp_r+rp_R+rp_l+rp_Q])
        rd = torch.tensor([rp_r+rp_R])
        # vd = torch.tensor([torch.tensor([rp_l,rp_Q,rp_r,rp_R,rd_u,rd_v,rd_U,rd_V]).max()]).float()
        vd = torch.tensor([torch.tensor([rp_l,rp_Q,rp_r,rp_R,rd_u,rd_v,rd_U,rd_V]).mean()]).float()

        ep_u = eps_abs*np.sqrt(K) + eps_rel*torch.maximum(torch.linalg.norm(l), torch.linalg.norm(r))
        ep_v = eps_abs + eps_rel*torch.maximum(torch.abs(r.sum()), torch.tensor(1))
        ep_U = sum([eps_abs*np.sqrt(Q[d].numel()) + eps_rel*torch.maximum( torch.linalg.norm(Q[d],'fro'), torch.linalg.norm(R[d],'fro') ) for d in range(2*D)])
        ep_V = sum([eps_abs*np.sqrt(K) + eps_rel*torch.maximum( torch.linalg.norm(R[d].sum(0)), torch.tensor(np.sqrt(K)) ) for d in range(2*D)])
        ep = torch.tensor([ep_u + ep_v + ep_U + ep_V]).float()

        ed_u = eps_abs*np.sqrt(K) + eps_rel*torch.linalg.norm(u,2) 
        ed_v = eps_abs + eps_rel*torch.linalg.norm(v,2)
        ed_U = sum([eps_abs*np.sqrt(Q[d].numel()) + eps_rel*torch.linalg.norm(U[d],'fro') for d in range(2*D)])
        ed_V = sum([eps_abs*np.sqrt(K) + eps_rel*torch.linalg.norm(torch.outer(V[d],torch.ones(K)),'fro') for d in range(2*D)])
        ed = torch.tensor([ed_u + ed_v + ed_U + ed_V]).float()

        # Combined residual
        cr = torch.tensor([rp_r + rp_R + rd_u + rd_v + rd_U + rd_V]).float()
        co_res = torch.cat(( co_res, cr ),0)
        if co_res[-1] >= co_res[-2]:
            w = 0.

        # Update auxiliary variables if accelerating
        rh = r + w * (r - r_prev)
        Rh = [R[d] + w * (R[d] - R_prev[d]) for d in range(2*D)]
        # rh = r + 0 * (r - r_prev)
        # Rh = [R[d] + 0 * (R[d] - R_prev[d]) for d in range(2*D)]
        uh = u + w * (u - u_prev)
        vh = v + w * (v - v_prev)
        Uh = [U[d] + w * (U[d] - U_prev[d]) for d in range(2*D)]
        Vh = [V[d] + w * (V[d] - V_prev[d]) for d in range(2*D)]

        Q_est = cp_to_tensor((l,Q))
        eh = torch.tensor([torch.norm(Q_est-Q_obs,'fro')]).float()
        err_hat = torch.cat((err_hat,eh),0)

        res_pri = torch.cat(( res_pri, rp ), 0)
        res_dua = torch.cat(( res_dua, rd ), 0)
        eps_pri = torch.cat(( eps_pri, ep ), 0)
        eps_dua = torch.cat(( eps_dua, ed ), 0)
        var_diff = torch.cat((var_diff,vd),0)

        if itr>=min_itr and (((res_pri<=eps_pri).sum()>0 and (res_dua<=eps_dua).sum()>0) or vd.item() < eps_diff):
            break

    if verbose and itr<max_itr-1:
        print("Terminated early")

    return Q_est, (l,Q), err_hat, (res_pri, res_dua, eps_pri, eps_dua), var_diff

def admm_dclrm_from_condpmf( P_obs,
                          K:int,
                          c:float=1.,
                          alpha:float=1.,
                          beta:float=1.,
                          eta:float=1e-6,
                          prob_min:float=0.,
                          eps_abs:float=1e-6, 
                          eps_rel:float=0., 
                          eps_diff:float=1e-6,
                          max_itr:int=300, 
                          min_itr:int=0,
                          admm_itr:int=500, 
                          verbose:bool=False,
                          disable_tqdm:bool=False
                          ):
    # ------------------------
    p,p = P_obs.shape

    # Initialize variables
    P = generate_probability_matrix(p,p)
    R = P.clone()
    S = P.clone()
    U = torch.zeros_like(P)
    W = torch.zeros_like(P)
    v = torch.zeros(p)

    # Initialize variables
    P = generate_probability_matrix(p,p)

    IR = torch.inverse( torch.eye(p) + 1 )
    err_hat = torch.tensor([])
    var_diff_out = torch.tensor([])

    for itr1 in tqdm(range(max_itr),disable=disable_tqdm):
        if not torch.allclose(P,torch.zeros_like(P)):
            try:
                up,svp,vhp = svds(P.numpy().astype(float),k=K)
                Y = torch.FloatTensor(up@vhp)
            except:
                try:
                    up,svp,vhp = torch.linalg.svd(P)
                    Y = up[:,:K]@vhp[:K,:]
                except:
                    up,svp,vhp = np.linalg.svd(P.numpy.astype(float))
                    Y = torch.FloatTensor(up[:,:K]@vhp[:K,:])
        else:
            Y = P.clone()

        R = P.clone()
        S = P.clone()
        U = torch.zeros_like(R)
        v = torch.zeros(p)

        res_pri = torch.tensor([])
        res_dua = torch.tensor([])
        eps_pri = torch.tensor([])
        eps_dua = torch.tensor([])
        var_diff_in = torch.tensor([])

        for itr2 in range(admm_itr):
            R_prev = R.clone()
            S_prev = S.clone()
            U_prev = U.clone()
            v_prev = v.clone()
            prox_arg = (P_obs + alpha*P + c*Y + beta*(S-U)) / (alpha+beta+1)
            if not torch.allclose(prox_arg,torch.zeros_like(prox_arg)):
                try:
                    UL,svs,URh = torch.linalg.svd(prox_arg)
                except:
                    UL,svs,URh = np.linalg.svd(prox_arg.numpy().astype(float))
                    UL = torch.FloatTensor(UL)
                    svs = torch.FloatTensor(svs)
                    URh = torch.FloatTensor(URh)
                trun_svs = soft_thresh(svs, c/(alpha+beta+1))
                R = UL @ torch.diag(trun_svs) @ URh
            else:
                R = prox_arg.clone()
            # S = torch.maximum((P + U + torch.outer(1-v,torch.ones(p))) @ IR, torch.zeros_like(S))
            S = torch.maximum((P + U + torch.outer(1-v,torch.ones(p))) @ IR, prob_min * torch.ones_like(S))
            U = U + R - S
            v = v + S.sum(1) - 1

            vd = torch.tensor([torch.tensor([
                torch.norm(R_prev-R,'fro'),
                torch.norm(S_prev-S,'fro'),
                torch.norm(U_prev-U,'fro'),
                torch.norm(v_prev-v,2)
            ]).max()]).float()

            rp = torch.tensor([ torch.norm(R-S,'fro') + torch.norm(R.sum(1)-1,2) ]).float()
            ep = torch.tensor([ eps_abs*np.sqrt(R.numel()) + eps_rel*torch.maximum(torch.norm(R),torch.norm(S)) + eps_abs*np.sqrt(p) + eps_rel*torch.maximum(torch.norm(R.sum(1)),torch.norm(torch.ones(p))) ]).float()
            rd = torch.tensor([ beta * torch.norm(S-S_prev,'fro') + beta * torch.norm(R-R_prev,'fro') + beta * torch.norm(R.sum(1)-R_prev.sum(1),2) ]).float()
            ed = torch.tensor([ eps_abs*np.sqrt(R.numel()) + eps_rel*torch.norm(beta*U) + eps_abs*np.sqrt(p) + eps_rel*torch.norm(beta*torch.outer(v,torch.ones(p))) ]).float()

            res_pri = torch.cat((res_pri,rp),0)
            eps_pri = torch.cat((eps_pri,ep),0)
            res_dua = torch.cat((res_dua,rd),0)
            eps_dua = torch.cat((eps_dua,ed),0)
            var_diff_in = torch.cat((var_diff_in,vd),0)

            if ((res_pri<=eps_pri).sum()>0 and (res_dua<=eps_dua).sum()>0) or (vd.item() < eps_diff):
                break

        P_prev = P.clone()
        P = R.clone()

        P_est = P.clone()
        # P_est = torch.maximum(P_est,torch.zeros_like(P_est))
        P_est = torch.maximum(P_est,prob_min * torch.ones_like(P_est))
        P_est = P_est / (P_est.sum(dim=1,keepdim=True) + (P_est.sum(dim=1,keepdim=True)==0).to(int))
        eh = torch.tensor([torch.norm(P-P_obs,'fro')]).float()
        err_hat = torch.cat((err_hat,eh),0)

        vdo = torch.tensor([torch.norm(P-P_prev,'fro')]).float()
        var_diff_out = torch.cat((var_diff_out,vdo),0)
        if itr1>=min_itr and vdo<eta:
            break

    if verbose and itr1<max_itr-1:
        print("Terminated early")

    return P_est, err_hat, var_diff_out

def admm_nnlrm_from_condpmf( P_obs,
                           gamma:float=.1,
                           beta:float=1., 
                           prob_min:float=0.,
                           eps_abs:float=1e-6, 
                           eps_rel:float=0., 
                           eps_diff:float=1e-6,
                           max_itr:int=5000, 
                           min_itr:int=0, 
                           verbose:bool=False,
                           disable_tqdm:bool=False
                           ):
    # ------------------------
    p,p = P_obs.shape

    # Initialize variables
    P = generate_probability_matrix(p,p)
    Q = P.clone()
    U = torch.zeros_like(P)
    V = torch.zeros(p)

    IQ = torch.inverse(torch.eye(p) + torch.ones((p,p)))

    var_diff = torch.tensor([])
    err_hat = torch.tensor([]) # Objective
    res_pri = torch.tensor([]); res_dua = torch.tensor([])
    eps_pri = torch.tensor([]); eps_dua = torch.tensor([])

    for itr in tqdm(range(max_itr),disable=disable_tqdm):
        # P update
        P_prev = P.clone()
        S = (P_obs + beta * (Q-U))/(1+beta)
        if not torch.allclose(S,torch.zeros_like(S)):
            try:
                UL,svs,UR = torch.linalg.svd(S)
            except:
                UL, svs, UR = np.linalg.svd(S.numpy().astype(float))
                UL = torch.FloatTensor(UL)
                svs = torch.FloatTensor(svs)
                UR = torch.FloatTensor(UR)
            P = UL @ torch.diag(soft_thresh(svs,gamma/(1+beta))) @ UR
        else:
            P = S.clone()
        # P = torch.maximum( P, torch.zeros_like(P) )
        P = torch.maximum( P, prob_min * torch.ones_like(P) )

        # Q update
        Q_prev = Q.clone()
        Q = (P + U + torch.outer(1-V,torch.ones(p)) ) @ IQ

        # Dual updates
        U_prev = U.clone()
        V_prev = V.clone()
        U = U + P - Q
        V = V + Q.sum(1)-1

        vd = torch.tensor([torch.tensor([
            torch.norm(Q-Q_prev,'fro'),
            torch.norm(P-P_prev,'fro'),
            torch.norm(U-U_prev,'fro'),
            torch.norm(V-V_prev,'fro')
        ]).max()]).float()

        # Primal residual
        rp_U = torch.norm( P-Q,'fro' )
        rp_V = torch.norm( Q.sum(1)-1,2 )
        rp = torch.tensor([rp_U + rp_V]).float()

        # Dual residual
        rd = torch.tensor([2 * beta * torch.norm( Q - Q_prev,'fro' )]).float()

        # Primal threshold
        ep_U = eps_abs * np.sqrt(P.numel()) + eps_rel * torch.maximum( torch.norm(P), torch.norm(Q) )
        ep_V = eps_abs * np.sqrt(p) + eps_rel * torch.maximum( torch.norm(Q.sum(1)), torch.norm(torch.ones(p)) )
        ep = torch.tensor([ep_U + ep_V]).float()

        # Dual threshold
        ed_U = eps_abs * np.sqrt(P.numel()) + eps_rel * torch.norm( beta * U )
        ed_V = eps_abs * np.sqrt(p) + eps_rel * torch.norm( beta * torch.outer(V,torch.ones(p)) )
        ed = torch.tensor([ed_U + ed_V]).float()

        res_pri = torch.cat(( res_pri, rp ), 0)
        res_dua = torch.cat(( res_dua, rd ), 0)
        eps_pri = torch.cat(( eps_pri, ep ), 0)
        eps_dua = torch.cat(( eps_dua, ed ), 0)

        P_est = P.clone()
        # P_est = torch.maximum(P_est,torch.zeros_like(P_est))
        P_est = torch.maximum(P_est,prob_min * torch.ones_like(P_est))
        P_est = P_est / (P_est.sum(dim=1,keepdim=True) + (P_est.sum(dim=1,keepdim=True)==0).to(int))
        eh = torch.tensor([torch.norm(P-P_obs,'fro')]).float()
        err_hat = torch.cat((err_hat,eh),0)

        var_diff = torch.cat(( var_diff, vd ), 0)

        # if (((res_pri<=eps_pri).sum()>0 and (res_dua<=eps_dua).sum()>0) or (vd.item() < eps_diff)) and itr>=min_itr:
        if itr>=min_itr and (((res_pri<=eps_pri).sum()>0 and (res_dua<=eps_dua).sum()>0) or (vd.item() < eps_diff)):
            break

    if verbose and itr<max_itr-1:
        print("Terminated early")

    return P_est, err_hat, (res_pri, res_dua, eps_pri, eps_dua), var_diff

def admm_lrt_from_jointpmf_mask( Q_obs,
                                 K:int, 
                                 Mask=None,
                                 beta:float=1., 
                                 alpha:float=1e3,
                                 lmbda_min:float=0.,
                                 prob_min:float=0.,
                                 eps_abs:float=1e-6, 
                                 eps_rel:float=0., 
                                 eps_diff:float=1e-6,
                                 max_itr:int=5000, 
                                 min_itr:int=0,
                                 verbose:bool=False,
                                 MARG_CONST:bool=True,
                                 ACCEL:bool=True,
                                 disable_tqdm:bool=False
                                ):
    # ------------------------------
    D = Q_obs.ndim // 2
    R = .5 * ( Q_obs.sum(dim=tuple(range(D,2*D))) + Q_obs.sum(dim=tuple(range(D))) )
    N = torch.tensor(Q_obs.shape)
    t = 1
    lmbda_min = np.min([lmbda_min,1/K])
    if not MARG_CONST:
        alpha = 0.
    if Mask is None:
        Mask = torch.ones_like(Q_obs)

    # Initialize variables
    _, Q, l = generate_tensor(2*D,K,N)
    S = [Q[d].clone() for d in range(2*D)]
    s = l.clone()
    u = torch.zeros_like(l)
    v = torch.zeros(1)
    U = [torch.zeros_like(Q[d]) for d in range(2*D)]
    V = [torch.zeros(K) for d in range(2*D)]
    Q1 = cp_to_tensor((l,Q[:D]))
    Q2 = cp_to_tensor((l,Q[D:]))
    sh = s.clone()
    uh = u.clone()
    vh = v.clone()
    Sh = [S[d].clone() for d in range(2*D)]
    Uh = [U[d].clone() for d in range(2*D)]
    Vh = [V[d].clone() for d in range(2*D)]

    var_diff = torch.tensor([])
    err_hat = torch.tensor([]) # Objective
    res_pri = torch.tensor([]); res_dua = torch.tensor([])
    eps_pri = torch.tensor([]); eps_dua = torch.tensor([])
    co_res = torch.tensor([0])

    Qd_obs = [unfold(Q_obs,mode=d).T for d in range(2*D)]
    Maskd = [unfold(Mask,mode=d).T for d in range(2*D)]
    Rd = [unfold(R,mode=d).T for d in range(D)]*2
    Is = torch.linalg.inv( torch.eye(K) + 1 )
    IS = [ torch.linalg.inv( torch.eye(N[d])+1 ) for d in range(2*D) ]

    for itr in tqdm(range(max_itr),disable=disable_tqdm):
        if ACCEL:
            # t update
            t_prev = t
            t = .5 * (1 + np.sqrt(1 + 4*t**2))
            w = (t_prev-1)/t
        else:
            w = 0.

        # l update
        T = khatri_rao(Q)
        q_obs = Q_obs.reshape((-1))
        mask = Mask.reshape((-1))
        T1 = khatri_rao(Q[:D])
        T2 = khatri_rao(Q[D:])
        Tdiff = T1-T2
        r = R.reshape((-1))

        Tm = T[mask==1]
        qm = q_obs[mask==1]

        l_prev = l.clone()
        Mat1 = Tm.T@Tm + beta*torch.eye(K) + alpha*( T1.T@T1 + T2.T@T2 + Tdiff.T@Tdiff )
        Mat2 = Tm.T@qm + alpha*( T1+T2 ).T@r + beta*(sh-uh)
        l = torch.maximum( torch.linalg.inv(Mat1) @ Mat2, lmbda_min* torch.ones_like(l) )

        # Q update
        Q_prev = [Q[d].clone() for d in range(2*D)]
        for d in range(2*D):
            Td = khatri_rao(Q,l,skip_matrix=d)
            T1d = khatri_rao(Q[:D],l,skip_matrix=d%D) if d<D else khatri_rao(Q[D:],l,skip_matrix=d%D)
            T2d = unfold(Q2,mode=d%D).T if d<D else unfold(Q1,mode=d%D).T
            
            maskd = Maskd[d].T.reshape(-1)
            Td_kron = torch.kron(torch.eye(N[d%D]),Td)
            qd_obs = Qd_obs[d].T.reshape(-1)
            Tdm = Td_kron[maskd==1]
            qdm = qd_obs[maskd==1]
            
            T1d_kron = torch.kron(torch.eye(N[d%D]),T1d)
            t2d = T2d.T.reshape(-1)
            rd = Rd[d].T.reshape(-1)
            
            Mat1 = Tdm.T@Tdm + beta*torch.eye(N[d%D]*K) + 2*alpha*T1d_kron.T@T1d_kron
            Mat2 = Tdm.T@qdm + beta*(Sh[d]-Uh[d]).reshape(-1) + alpha*T1d_kron.T@(t2d+rd)
            Q[d] = torch.maximum( torch.linalg.inv(Mat1)@Mat2, prob_min*torch.ones(N[d%D]*K) ).reshape(Q[d].shape)
        Q1 = cp_to_tensor((l,Q[:D]))
        Q2 = cp_to_tensor((l,Q[D:]))

        # s update
        sh_prev = sh.clone()
        s_prev = s.clone()
        s = Is @ ( l+uh + (1-vh) )

        # S update
        Sh_prev = [Sh[d].clone() for d in range(2*D)]
        S_prev = [S[d].clone() for d in range(2*D)]
        S = [ IS[d] @ (Q[d]+Uh[d] + torch.outer( torch.ones(N[d]), (1-Vh[d]) )) for d in range(2*D)]

        # u update
        u_prev = u.clone()
        uh_prev = uh.clone()
        u = uh + l - s

        # U update
        Uh_prev = [Uh[d].clone() for d in range(2*D)]
        U_prev = [U[d].clone() for d in range(2*D)]
        U = [Uh[d] + Q[d] - S[d] for d in range(2*D)]

        # v update
        vh_prev = vh.clone()
        v_prev = v.clone()
        v = vh + s.sum()-1

        # V update
        Vh_prev = [Vh[d].clone() for d in range(2*D)]
        V_prev = [V[d].clone() for d in range(2*D)]
        V = [Vh[d] + S[d].sum(0)-1 for d in range(2*D)]

        # Residuals
        rp_l = torch.linalg.norm(l - l_prev,2)
        rp_Q = sum([torch.linalg.norm(Q[d]-Q_prev[d],'fro') for d in range(2*D)])
        rp_s = torch.linalg.norm(s - sh_prev,2)
        rp_S = sum([torch.linalg.norm(S[d] - Sh_prev[d],'fro') for d in range(2*D)])
        rd_u = torch.linalg.norm(u - uh_prev,2)
        rd_v = torch.linalg.norm(v - vh_prev,2)
        rd_U = sum([torch.linalg.norm(U[d] - Uh_prev[d]) for d in range(2*D)])
        rd_V = sum([torch.linalg.norm(V[d] - Vh_prev[d]) for d in range(2*D)])

        rp = torch.tensor([rd_u+rd_v+rd_U+rd_V]).float()
        # rd = torch.tensor([rp_s+rp_S+rp_l+rp_Q])
        rd = torch.tensor([rp_s+rp_S])
        # vd = torch.tensor([torch.tensor([rp_l,rp_Q,rp_s,rp_S,rd_u,rd_v,rd_U,rd_V]).max()]).float()
        vd = torch.tensor([torch.tensor([rp_l,rp_Q,rp_s,rp_S,rd_u,rd_v,rd_U,rd_V]).mean()]).float()

        ep_u = eps_abs*np.sqrt(K) + eps_rel*torch.maximum(torch.linalg.norm(l), torch.linalg.norm(s))
        ep_v = eps_abs + eps_rel*torch.maximum(torch.abs(s.sum()), torch.tensor(1))
        ep_U = sum([eps_abs*np.sqrt(Q[d].numel()) + eps_rel*torch.maximum( torch.linalg.norm(Q[d],'fro'), torch.linalg.norm(S[d],'fro') ) for d in range(2*D)])
        ep_V = sum([eps_abs*np.sqrt(K) + eps_rel*torch.maximum( torch.linalg.norm(S[d].sum(0)), torch.tensor(np.sqrt(K)) ) for d in range(2*D)])
        ep = torch.tensor([ep_u + ep_v + ep_U + ep_V]).float()

        ed_u = eps_abs*np.sqrt(K) + eps_rel*torch.linalg.norm(u,2) 
        ed_v = eps_abs + eps_rel*torch.linalg.norm(v,2)
        ed_U = sum([eps_abs*np.sqrt(Q[d].numel()) + eps_rel*torch.linalg.norm(U[d],'fro') for d in range(2*D)])
        ed_V = sum([eps_abs*np.sqrt(K) + eps_rel*torch.linalg.norm(torch.outer(V[d],torch.ones(K)),'fro') for d in range(2*D)])
        ed = torch.tensor([ed_u + ed_v + ed_U + ed_V]).float()

        # Combined residual
        cr = torch.tensor([rp_s + rp_S + rd_u + rd_v + rd_U + rd_V]).float()
        co_res = torch.cat(( co_res, cr ),0)
        if co_res[-1] >= co_res[-2]:
            w = 0.

        # Update auxiliary variables if accelerating
        # sh = s + w * (s - s_prev)
        # Sh = [S[d] + w * (S[d] - S_prev[d]) for d in range(2*D)]
        sh = s + 0 * (s - s_prev)
        Sh = [S[d] + 0 * (S[d] - S_prev[d]) for d in range(2*D)]
        uh = u + w * (u - u_prev)
        vh = v + w * (v - v_prev)
        Uh = [U[d] + w * (U[d] - U_prev[d]) for d in range(2*D)]
        Vh = [V[d] + w * (V[d] - V_prev[d]) for d in range(2*D)]

        Q_est = cp_to_tensor((l,Q))
        eh = torch.tensor([torch.norm(Q_est-Q_obs,'fro')]).float()
        err_hat = torch.cat((err_hat,eh),0)

        res_pri = torch.cat(( res_pri, rp ), 0)
        res_dua = torch.cat(( res_dua, rd ), 0)
        eps_pri = torch.cat(( eps_pri, ep ), 0)
        eps_dua = torch.cat(( eps_dua, ed ), 0)
        var_diff = torch.cat((var_diff,vd),0)

        if itr>=min_itr and (((res_pri<=eps_pri).sum()>0 and (res_dua<=eps_dua).sum()>0) or vd.item() < eps_diff):
            break

    if verbose and itr<max_itr-1:
        print("Terminated early")

    return Q_est, (l,Q), err_hat, (res_pri, res_dua, eps_pri, eps_dua), var_diff