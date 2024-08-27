from src.environments import *
from src.opt import *
from plt_utils import *

import os
from joblib import Parallel, delayed
from time import perf_counter

def frob_err(Ph,P):
    return torch.norm(Ph-P,'fro').item()
def normfrob_err(Ph,P):
    return (torch.norm(Ph-P,'fro') / (torch.norm(P,'fro') + int(P.abs().max()>0))).item()
def l1_err(Ph,P):
    return torch.norm(Ph-P,1).item()
def norml1_err(Ph,P):
    return (torch.norm(Ph-P,1) / (torch.norm(P,1) + int(P.abs().max()>0))).item()
def sin_err(Ph,P,K:int=None,sv_type:str='both'):
    assert sv_type in ['both','left','right'], "Invalid SV type."
    if K is None:
        K,K = P.shape
    if K<P.shape[0]:
        U,_,V = svds(P.numpy().astype(float),K)
        Uh,_,Vh = svds(Ph.numpy().astype(float),K)
    else:
        U,_,VT = np.linalg.svd(P.numpy().astype(float))
        V = VT.T
        Uh,_,VhT = np.linalg.svd(Ph.numpy().astype(float))
        Vh = VhT.T
    if sv_type=='right':
        return float(np.sqrt( np.abs(K - np.linalg.norm(Vh@V.T,'fro')**2) ) / np.sqrt(K))
    elif sv_type=='left':
        return float(np.sqrt( np.abs(K - np.linalg.norm(Uh.T@U,'fro')**2) ) / np.sqrt(K))
    else:
        return float(torch.tensor([ np.sqrt( np.abs(K - np.linalg.norm(Uh.T@U,'fro')**2) ), np.sqrt( np.abs(K - np.linalg.norm(Vh@V.T,'fro')**2) ) ]).to(torch.float).max().item() / np.sqrt(K))

def erank(A):
    svs = torch.linalg.svdvals(A)
    p = svs / (torch.linalg.norm(svs,1) + int(svs.abs().max()==0))
    return torch.exp(-torch.sum(p * torch.log(p)))
    
def mat2lowtri(A,N=None):
    N=A.shape[0]
    low_tri_indices=np.triu_indices(N,1)
    return A[low_tri_indices[1],low_tri_indices[0]]

def lowtri2mat(a):
    N=int((2*len(a)+.25)**.5+.5)
    A=np.zeros((N,N)) if type(a)!=torch.Tensor else torch.zeros((N,N))
    low_tri_indices=np.triu_indices(N,1)
    A[low_tri_indices[1],low_tri_indices[0]]=a
    A=A+A.T
    return A

# ------------------------------------------------------------------------

def generate_erdosrenyi_matrix_model(N,edge_prob:float=.3,eps:float=.02,beta:float=1.):
    A = lowtri2mat(np.random.binomial(1,edge_prob,int(N*(N-1)/2)))
    L = np.diag(A.sum(0)) - A
    while np.sum(np.abs(np.linalg.eigvalsh(L))<1e-9)>1:
        A = lowtri2mat(np.random.binomial(1,edge_prob,int(N*(N-1)/2)))
        L = np.diag(A.sum(0)) - A
    At = A + beta * np.eye(N)
    P0 = At * (1-eps) + eps
    P = torch.tensor(P0 / P0.sum(1,keepdims=True)).to(torch.float)
    mc = MarkovChainMatrix(P)

    return mc

def generate_lowranktensor_model(N,K:int):
    D = len(N)
    Ntot = torch.prod(N).item()

    # Zhu et al., 2022, Operations Research, "Learning Markov"
    U = [torch.randn(N[d%D],K) for d in range(2*D)]
    U = [(U[d]*U[d])/(torch.linalg.norm(U[d],dim=0,keepdim=True)**2) for d in range(2*D)]
    # w = torch.FloatTensor(np.random.beta(.5,.5,K))
    w = torch.FloatTensor(np.random.rand(K))
    # w = (w / np.linalg.norm(w))**2
    w = w / w.sum()
    P = cp_to_tensor((w,U))
    P_mat = P.reshape(Ntot,Ntot)
    P_mat = P_mat / P_mat.sum(dim=1,keepdim=True)
    P = P_mat.reshape(tuple(N.repeat(2)))
    mc = MarkovChainTensor(P)

    return mc

def generate_lowrankmatrix_model(N:int,K:int):
    # Zhu et al., 2022, Operations Research, "Learning Markov"
    U0 = torch.randn(N,K)
    U0 = (U0*U0)/(torch.linalg.norm(U0,dim=1,keepdim=True)**2)
    V0 = torch.randn(N,K)
    V0 = (V0*V0)/(torch.linalg.norm(V0,dim=0,keepdim=True)**2)
    B = torch.diag(torch.FloatTensor(np.random.beta(.5,.5,N)))
    P0 = (U0@V0.T)@B
    P_1D = P0 / P0.sum(dim=1,keepdim=True)
    mc = MarkovChainMatrix(P_1D)

    return mc

def generate_blocktensor_model(N_per_block:int,D:int,K:int):
    state_mat = torch.rand((K,)*2*D)
    Q = torch.kron(state_mat, torch.ones((N_per_block//K,)*2*D))
    Q = Q / torch.linalg.norm(Q.reshape(-1),1)
    P = Q / Q.sum(dim=tuple(range(D,2*D)),keepdim=True)
    mc = MarkovChainTensor(P)
    return mc

def generate_blockmatrix_model(N_per_block:int,K:int):
    state_mat = torch.rand((K,K))
    Q = torch.kron(state_mat, torch.ones((N_per_block//K,N_per_block//K)))
    Q = Q / torch.linalg.norm(Q.reshape(-1),1)
    P = Q / Q.sum(dim=1,keepdim=True)
    mc = MarkovChainMatrix(P)
    return mc

def estimate_empirical_matrix(X,N:int):
    transition_counts = torch.zeros((N,N),dtype=int)
    X_steps = np.array([np.array(X)[:-1],np.array(X)[1:]])
    X_steps,counts = np.unique(X_steps,axis=1,return_counts=True)
    transition_counts[(X_steps[0],X_steps[1])] = torch.tensor(counts)

    total_counts = transition_counts.sum(1,keepdim=True)
    Ph = transition_counts.float() / total_counts
    Mask = (total_counts==0).expand_as(Ph)
    Ph[Mask] = 1/N

    marginal_counts = transition_counts.sum(1)
    marginal_probs = marginal_counts.float() / marginal_counts.sum()

    marginal_counts = transition_counts.sum(1)
    total_transitions = marginal_counts.sum()
    Qh = transition_counts.float() / total_transitions

    return Ph, Qh, Mask

def estimate_empirical_tensor(X,N):
    Ntot = N.prod().item()
    D = len(N)

    transition_counts = torch.zeros(tuple(N.repeat(2)),dtype=int)
    X_steps = np.concatenate([np.array(X)[:-1],np.array(X)[1:]],axis=1).T
    X_steps, counts = np.unique(X_steps,axis=1,return_counts=True)
    transition_counts[*list(map(tuple,X_steps))] = torch.tensor(counts)

    total_counts = transition_counts.sum(tuple(range(D,2*D)),keepdim=True)
    Ph = transition_counts.float() / total_counts
    Mask = (total_counts==0).expand_as(Ph)
    Ph[Mask] = 1/Ntot

    marginal_counts = transition_counts.sum(tuple(range(D,2*D)))
    marginal_probs = marginal_counts.float() / marginal_counts.sum()

    marginal_counts = transition_counts.sum(tuple(range(D,2*D)))
    total_transitions = marginal_counts.sum()
    Qh = transition_counts.float() / total_transitions

    return Ph, Qh, Mask

# ------------------------------------------------------------------------

# Low-rank tensor estimator
class LowRankTensorEstimator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        for a in ['mc','P','Q','P_1D','Q_1D','pi','pi_1D','lmbda','Qfact','admm_err','admm_res','admm_var','admm_iters']:
            setattr(self,a,None)
        for a in ['Qh','K','beta','eps_abs','eps_rel','eps_diff','max_itr','verbose','MARG_CONST','ACCEL']:
            setattr(self,a,None)

    # def estimate(self, Q_obs, args):
    def estimate(self, Q_obs, Mask, args):
        self.reset()

        D = Q_obs.ndim // 2
        N = torch.tensor(Q_obs.shape[:D])
        Ntot = torch.prod(N).item()
        # args['Mask'] = Mask

        # Estimate low-rank joint PMF tensor
        Q_LRT, (lmbda, Qfact), err_LRT, (rp_LRT, rd_LRT, ep_LRT, ed_LRT), var_diff = admm_lrt_from_jointpmf(Q_obs,**args)
        # Q_LRT, (lmbda, Qfact), err_LRT, (rp_LRT, rd_LRT, ep_LRT, ed_LRT), var_diff = admm_lrt_from_jointpmf_mask(Q_obs,**args)

        assert not torch.isnan(Q_LRT).any(), "Nan values in estimated Q."
        assert not torch.isinf(Q_LRT).any(), "Inf. values in estimated Q."

        marg_lrt = Q_LRT.sum(dim=tuple(range(D,2*D)),keepdim=True)
        P_LRT = Q_LRT / (marg_lrt + (marg_lrt==0).to(int))
        mask = (marg_lrt==0).expand_as(P_LRT)
        P_LRT[mask] = 1/Ntot

        mc_LRT = MarkovChainTensor(P_LRT)

        self.mc = mc_LRT
        self.P = P_LRT
        self.Q = Q_LRT
        self.P_1D = P_LRT.reshape(Ntot,Ntot)
        self.Q_1D = Q_LRT.reshape(Ntot,Ntot)
        self.pi = marg_lrt
        self.pi_1D = marg_lrt.reshape(Ntot)
        self.lmbda = lmbda
        self.Qfact = Qfact
        self.admm_obj = err_LRT
        self.admm_res = (rp_LRT, rd_LRT, ep_LRT, ed_LRT)
        self.admm_var = var_diff
        self.admm_iters = len(err_LRT)

        self.Qh = Q_obs
        for key in args:
            setattr(self,key,args[key])

        res = {a:getattr(self,a) for a in ['mc','lmbda','Qfact','admm_obj','admm_res','admm_var','admm_iters']}

        return self.mc, res

    def plot_admm_objective(self,start_idx:int=0):
        assert self.admm_obj is not None, "ADMM objective not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_obj[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Objective')
        fig.tight_layout()

    def plot_admm_var_diff(self,start_idx:int=0):
        assert self.admm_var is not None, "ADMM variable difference not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_var[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Variable difference')
        fig.tight_layout()

    def plot_admm_convergence(self,start_idx:int=0):
        assert self.admm_res is not None, "ADMM residuals not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        (rp,rd,ep,ed) = self.admm_res

        clr = vib_qual['red']
        fig = plt.figure(figsize=(2*5,4)); ax = fig.subplots(1,2); _ = [a.grid(1) for a in ax]; _ = [a.set_axisbelow(1) for a in ax]
        _ = ax[0].plot(np.arange(start_idx,self.admm_iters),rp[start_idx:], '-',c=clr)
        _ = ax[0].plot(np.arange(start_idx,self.admm_iters),ep[start_idx:], ':',c=clr,alpha=.3)
        ax[0].set_xlabel("Iteration"); ax[0].set_ylabel('Primal residual')
        _ = ax[1].plot(np.arange(start_idx,self.admm_iters),rd[start_idx:], '-',c=clr)
        _ = ax[1].plot(np.arange(start_idx,self.admm_iters),ed[start_idx:], ':',c=clr,alpha=.3)
        ax[1].set_xlabel("Iteration"); ax[1].set_ylabel('Dual residual')
        fig.tight_layout()

    def plot_P_estimate(self,normalize:bool=True):
        assert self.P_1D is not None, "Estimate not found."
        if normalize:
            madimshow(self.P_1D,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.P_1D,'plasma',axis=False)

    def plot_Q_estimate(self,normalize:bool=True):
        assert self.Q_1D is not None, "Estimate not found."
        if normalize:
            madimshow(self.Q_1D,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.Q_1D,'plasma',axis=False)

# Nuclear norm matrix estimator
class NucNormMatrixEstimator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        for a in ['mc','P','Q','pi','admm_err','admm_res','admm_var','admm_iters']:
            setattr(self,a,None)
        for a in ['Ph','gamma','beta','eps_abs','eps_rel','eps_diff','max_itr','verbose']:
            setattr(self,a,None)

    def estimate(self, P_obs, args):
        assert P_obs.ndim==2 and P_obs.shape[0]==P_obs.shape[1], "Invalid size of input matrix."
        self.reset()

        Ntot = P_obs.shape[0]

        # Estimate low-rank conditional PMF matrix
        P_LRM, err_LRM, (rp_LRM, rd_LRM, ep_LRM, ed_LRM), var_diff = admm_nnlrm_from_condpmf(P_obs,**args)

        assert not torch.isnan(P_LRM).any(), "Nan values in estimated P."
        assert not torch.isinf(P_LRM).any(), "Inf. values in estimated P."

        mc_LRM = MarkovChainMatrix(P_LRM)

        marg_lrm = mc_LRM.Q.sum(dim=1,keepdim=True)
        mc_LRM.P = mc_LRM.Q / (marg_lrm + (marg_lrm==0).to(int))
        mask = (marg_lrm==0).expand_as(mc_LRM.P)
        mc_LRM.P[mask] = 1/Ntot

        self.mc = mc_LRM
        self.P = mc_LRM.P
        self.Q = mc_LRM.Q
        self.pi = marg_lrm
        self.admm_obj = err_LRM
        self.admm_res = (rp_LRM, rd_LRM, ep_LRM, ed_LRM)
        self.admm_var = var_diff
        self.admm_iters = len(err_LRM)

        self.Ph = P_obs
        for key in args:
            setattr(self,key,args[key])

        res = {a:getattr(self,a) for a in ['mc','admm_obj','admm_res','admm_var','admm_iters']}

        # return self.mc
        return self.mc, res

    def plot_admm_objective(self,start_idx:int=0):
        assert self.admm_obj is not None, "ADMM objective not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_obj[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Objective')
        fig.tight_layout()

    def plot_admm_var_diff(self,start_idx:int=0):
        assert self.admm_var is not None, "ADMM variable difference not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_var[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Variable difference')
        fig.tight_layout()

    def plot_admm_convergence(self,start_idx:int=0):
        assert self.admm_res is not None, "ADMM residuals not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."
        (rp,rd,ep,ed) = self.admm_res

        clr = vib_qual['red']
        fig = plt.figure(figsize=(2*5,4)); ax = fig.subplots(1,2); _ = [a.grid(1) for a in ax]; _ = [a.set_axisbelow(1) for a in ax]
        _ = ax[0].plot(np.arange(start_idx,self.admm_iters),rp[start_idx:], '-',c=clr)
        _ = ax[0].plot(np.arange(start_idx,self.admm_iters),ep[start_idx:], ':',c=clr,alpha=.3)
        ax[0].set_xlabel("Iteration"); ax[0].set_ylabel('Primal residual')
        _ = ax[1].plot(np.arange(start_idx,self.admm_iters),rd[start_idx:], '-',c=clr)
        _ = ax[1].plot(np.arange(start_idx,self.admm_iters),ed[start_idx:], ':',c=clr,alpha=.3)
        ax[1].set_xlabel("Iteration"); ax[1].set_ylabel('Dual residual')
        fig.tight_layout()

    def plot_P_estimate(self,normalize:bool=True):
        assert self.P is not None, "Estimate not found."
        if normalize:
            madimshow(self.P,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.P,'plasma',axis=False)

    def plot_Q_estimate(self,normalize:bool=True):
        assert self.Q is not None, "Estimate not found."
        if normalize:
            madimshow(self.Q,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.Q,'plasma',axis=False)

# DC low-rank matrix estimator
class DCLowRankMatrixEstimator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        for a in ['mc','P','Q','pi','admm_err','admm_var','admm_iters']:
            setattr(self,a,None)
        for a in ['Ph','K','c','alpha','beta','eta','eps_abs','eps_rel','eps_diff','max_itr','admm_itr','verbose']:
            setattr(self,a,None)

    def estimate(self, P_obs, args):
        assert P_obs.ndim==2 and P_obs.shape[0]==P_obs.shape[1], "Invalid size of input matrix."
        self.reset()

        Ntot = P_obs.shape[0]

        # Estimate low-rank conditional PMF matrix
        P_LRM, err_LRM, var_diff = admm_dclrm_from_condpmf(P_obs,**args)

        assert not torch.isnan(P_LRM).any(), "Nan values in estimated P."
        assert not torch.isinf(P_LRM).any(), "Inf. values in estimated P."

        mc_LRM = MarkovChainMatrix(P_LRM)

        marg_lrm = mc_LRM.Q.sum(dim=1,keepdim=True)
        mc_LRM.P = mc_LRM.Q / (marg_lrm + (marg_lrm==0).to(int))
        mask = (marg_lrm==0).expand_as(mc_LRM.P)
        mc_LRM.P[mask] = 1/Ntot

        self.mc = mc_LRM
        self.P = mc_LRM.P
        self.Q = mc_LRM.Q
        self.pi = marg_lrm
        self.admm_obj = err_LRM
        self.admm_var = var_diff
        self.admm_iters = len(err_LRM)

        self.Ph = P_obs
        for key in args:
            setattr(self,key,args[key])

        res = {a:getattr(self,a) for a in ['mc','admm_obj','admm_var','admm_iters']}

        # return self.mc
        return self.mc, res

    def plot_admm_objective(self,start_idx:int=0):
        assert self.admm_obj is not None, "ADMM objective not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_obj[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Objective')
        fig.tight_layout()

    def plot_admm_var_diff(self,start_idx:int=0):
        assert self.admm_var is not None, "ADMM variable difference not found."
        assert start_idx <= self.admm_iters, "Starting index out of range."

        clr = vib_qual['red']
        fig = plt.figure(); ax = fig.subplots(); ax.grid(1); ax.set_axisbelow(1)
        ax.plot(np.arange(start_idx,self.admm_iters),self.admm_var[start_idx:], '-',c=clr)
        ax.set_xlabel("Iteration"); ax.set_ylabel('Variable difference')
        fig.tight_layout()

    def plot_P_estimate(self,normalize:bool=True):
        assert self.P is not None, "Estimate not found."
        if normalize:
            madimshow(self.P,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.P,'plasma',axis=False)

    def plot_Q_estimate(self,normalize:bool=True):
        assert self.Q is not None, "Estimate not found."
        if normalize:
            madimshow(self.Q,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.Q,'plasma',axis=False)

# Spectral low-rank matrix estimator
class SpecLowRankMatrixEstimator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        for a in ['mc','P','Q','pi','K','Qh']:
            setattr(self,a,None)

    def estimate(self, Q_obs, K, prob_min:float=0.):
        assert Q_obs.ndim==2 and Q_obs.shape[0]==Q_obs.shape[1], "Invalid size of input matrix."
        self.reset()

        Ntot = Q_obs.shape[0]

        # Estimate low-rank conditional PMF matrix
        UK,svK,VhK = svds(Q_obs.numpy().astype(float),k=K)
        # Q_LRM = torch.maximum( torch.FloatTensor(UK @ np.diag(svK) @ VhK),torch.zeros_like(Q_obs) )
        Q_LRM = torch.maximum( torch.FloatTensor(UK @ np.diag(svK) @ VhK),prob_min*torch.ones_like(Q_obs) )
        Q_LRM = Q_LRM / torch.linalg.norm(Q_LRM,1)

        marg_slrm = Q_LRM.sum(dim=1,keepdim=True)
        P_LRM = Q_LRM / (marg_slrm + (marg_slrm==0).to(int))
        mask = (marg_slrm==0).expand_as(P_LRM)
        P_LRM[mask] = 1/Ntot
        mc_slrm = MarkovChainMatrix(P_LRM)

        assert not torch.isnan(P_LRM).any(), "Nan values in estimated P."
        assert not torch.isinf(P_LRM).any(), "Inf. values in estimated P."
        assert not torch.isnan(Q_LRM).any(), "Nan values in estimated Q."
        assert not torch.isinf(Q_LRM).any(), "Inf. values in estimated Q."

        self.mc = mc_slrm
        self.P = mc_slrm.P
        self.Q = mc_slrm.Q
        self.pi = mc_slrm.pi

        self.Qh = Q_obs
        self.K = K

        res = {a:getattr(self,a) for a in ['mc']}

        # return self.mc
        return self.mc, res

    def plot_P_estimate(self,normalize:bool=True):
        assert self.P is not None, "Estimate not found."
        if normalize:
            madimshow(self.P,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.P,'plasma',axis=False)

    def plot_Q_estimate(self,normalize:bool=True):
        assert self.Q is not None, "Estimate not found."
        if normalize:
            madimshow(self.Q,'plasma',axis=False,vmin=0,vmax=1)
        else:
            madimshow(self.Q,'plasma',axis=False)
