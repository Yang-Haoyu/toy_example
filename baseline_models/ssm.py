from distutils.util import strtobool
from baseline_models.base import Model
from baseline_models.utils import *
from baseline_models.inference import RNN_STInf
from baseline_models.iefs.att_iefs import AttentionIEFTransition
from baseline_models.iefs.moe import MofE
import torch
from pyro.distributions import Normal, Independent
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
def calc_MI(x, y, bins=10):
    mi_a = adjusted_mutual_info_score(x, y)
    mi_n = normalized_mutual_info_score(x, y)
    return float(mi_a), float(mi_n)


class SSM(Model):
    def __init__(self,hparams, **kwargs):
        super(SSM, self).__init__(hparams)
        self.save_hyperparameters()

    def init_model(self):
        # ttype       = self.hparams['ttype']
        # etype = self.hparams['etype']
        dim_hidden  = self.hparams['dim_hidden']
        num_heads   = self.hparams['nheads']
        dim_stochastic = self.hparams['dim_stochastic']
        d_gene = self.hparams['d_gene']
        d_x    = self.hparams['d_x']
        d_treat   = self.hparams['d_treat'] + 3
        # post_approx = self.hparams['post_approx']
        # inftype     = self.hparams['inftype']
        etype = self.hparams['etype']
        ttype = self.hparams['ttype']
        # augmented   = self.hparams['augmented']
        # alpha1_type = self.hparams['alpha1_type']

        combiner_type = self.hparams['combiner_type']
        # nheads = self.hparams['nheads']
        # add_stochastic = self.hparams['add_stochastic']


        # Inference Network
        # self.inf_noise = np.abs(self.hparams['inf_noise'])

        self.inf_network    = RNN_STInf(self.hparams, d_gene, d_x, d_treat, dim_hidden,
                                            dim_stochastic, nl='relu', combiner_type = combiner_type)


        # Emission Function
        if etype == 'lin':
            self.e_mu    = nn.Linear(dim_stochastic, d_x)
            self.e_sigma = nn.Linear(dim_stochastic, d_x)
        elif etype  == 'nl':
            dim_hidden   = self.trial.suggest_int('dim_hidden',100,500)
            emodel       = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden), nn.ReLU(True))
            self.e_mu    = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
            self.e_sigma = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
        else:
            raise ValueError('bad etype')

        # Transition Function

        # self.transition_fxn = TransitionFunction(self.hparams,dim_stochastic, d_x, d_treat, dim_hidden, ttype,
        #                                          num_heads=num_heads)
        # if self.hparams['ttype'] == "ief_g":
            
        # else: 
            
        
        # Prior over Z1
        if self.hparams['ttype'] == "ief_g":
            self.transition_fxn = TransitionFunction(self.hparams, dim_stochastic, d_x, d_treat+d_gene, dim_hidden, ttype, num_heads=num_heads)
            self.prior_W        = nn.Linear(d_treat+d_x+d_gene, dim_stochastic)
            self.prior_sigma    = nn.Linear(d_treat+d_x+d_gene, dim_stochastic)
        elif self.hparams['ttype'] == "ief":
            self.transition_fxn = TransitionFunction(self.hparams, dim_stochastic, d_x, d_treat, dim_hidden, ttype, num_heads=num_heads)
            self.prior_W        = nn.Linear(d_treat+d_x, dim_stochastic)
            self.prior_sigma    = nn.Linear(d_treat+d_x, dim_stochastic)
        else:
            raise NotImplementedError

    def p_Z1(self, B, X0, A0):
        inp_cat = torch.cat([B, X0, A0], -1)
        mu      = self.prior_W(inp_cat)
        sigma   = torch.nn.functional.softplus(self.prior_sigma(inp_cat))
        p_z_bxa = Independent(Normal(mu, sigma), 1)
        return p_z_bxa

    def p_X_Z(self, Zt, Tval):

        mu          = self.e_mu(Zt)
        sigma       = torch.nn.functional.softplus(self.e_sigma(Zt))

        return mu, sigma

    def p_Zt_Ztm1(self, Zt, A, B, X, A0, eps = 0.):
        X0 = X[:,0,:]
        Xt = X[:,1:,:]

        if self.hparams['ttype'] == "ief_g":
            inp_cat = torch.cat([B, X0, A0], -1)
        elif self.hparams['ttype'] == "ief":
            inp_cat = torch.cat([X0, A0], -1)
        else:
            raise NotImplementedError

        mu1      = self.prior_W(inp_cat)[:,None,:]
        sig1     = torch.nn.functional.softplus(self.prior_sigma(inp_cat))[:,None,:]
#         mu1      = torch.zeros_like(sig1).to(sig1.device)

        Tmax     = Zt.shape[1]

        Zinp = Zt[:, :-1, :]
        if self.hparams['ttype'] == "ief_g":
            Aval = A[:,1:Tmax,:]
            # Aval = torch.cat([Aval, B[:,None,:].repeat(1,Aval.shape[1],1), torch.zeros_like(Aval).repeat(1,1,3)], -1)
            Aval = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            # Acat = torch.cat([Aval[...,[0]],B[:,None,:].repeat(1,Aval.shape[1],1), Aval[...,1:]],-1)
            mu2T, sig2T = self.transition_fxn(Zinp, Aval, eps = eps)
        elif self.hparams['ttype'] == "ief":
            mu2T, sig2T = self.transition_fxn(Zinp, A[:, 1:Tmax, :], eps=eps)
        else:
            raise NotImplementedError

        mu, sig     = torch.cat([mu1,mu2T],1), torch.cat([sig1,sig2T],1)
        return Independent(Normal(mu, sig), 1)

    def get_loss(self, B, X, A, M, anneal = 1.):
        _, _, lens         = get_masks(M)
        B, X, A, M = B[lens>1], X[lens>1], A[lens>1], M[lens>1]
        m_t, m_g_t, _      = get_masks(M[:,1:,:])
        # Xnew = X + torch.randn(X.shape).to(X.device)
        Xnew = X
        Z_t, q_zt, _          = self.inf_network(Xnew, A, M, B)
        Tmax               = Z_t.shape[1]
        p_x_mu, p_x_std    = self.p_X_Z(Z_t, A[:,1:Tmax+1,[0]])
        p_zt               = self.p_Zt_Ztm1(Z_t, A, B, X, A[:,0,:])
        masked_nll         = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, M[:,1:Tmax+1,:])
        full_masked_nll    = masked_nll
        masked_nll         = masked_nll.sum(-1).sum(-1)
        self.masked_nll_mean = ((masked_nll/full_masked_nll.shape[-1])/m_t.sum(-1)).mean()


        kl_t       = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t= (m_t[:,:Tmax]*kl_t).sum(-1)
        self.masked_kl_mean = (masked_kl_t/m_t.sum(-1)).mean()
        neg_elbo   = masked_nll + anneal*masked_kl_t



        return neg_elbo, masked_nll, masked_kl_t, Z_t, M
    def _infer_z(self, B, X, A, M):
        _, _, lens         = get_masks(M)
        B, X, A, M = B[lens>1], X[lens>1], A[lens>1], M[lens>1]
        m_t, m_g_t, _   = get_masks(M[:,1:,:])
        z_mu          = self.inf_network.infer_forward(X, A, M, B)
        return z_mu, M

    def predict(self, X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask):
        Trt = torch.cat([Trt, torch.zeros_like(Trt).repeat(1,1,3)], -1)
        Z_t, Mask = self._infer_z(Genetic, X, Trt,  Mask)

        d_z = Z_t.shape[-1]
        z_lst_sel = Z_t.masked_select(Mask[:, 1:, 0:1].expand(-1, -1, d_z).bool()).reshape(-1, d_z)
        kmeans = KMeans(n_clusters=self.hparams['n_ggroup'], random_state=1, n_init="auto").fit( z_lst_sel.detach().cpu().numpy())
        cluster_pred = kmeans.labels_

        y_sel = Z_gt[:, 1:].masked_select(Mask[:, 1:, 0].bool()).detach().cpu().numpy()

        result = np.zeros((len(set(y_sel)), len(set(y_sel))))
        for i in range(len(y_sel)):
            result[int(y_sel[i]), int(cluster_pred[i])] += 1
        chi2 = chi2_contingency(result)[0]

        return Z_t, chi2, result

    def forward(self, X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask, anneal = 1.):
        # M sequence mask [N x t]
        B = Genetic
        A = torch.cat([Trt, torch.zeros_like(Trt).repeat(1,1,3)], -1)

        M = Mask

        neg_elbo, masked_nll, kl, Z_t, M  = self.get_loss(B, X, A, M, anneal = anneal)
        reg_loss   = torch.mean(neg_elbo)


        # for name,param in self.named_parameters():
            # if self.reg_all == 'all':
            #     # reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
            #     reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
            # elif self.reg_all == 'except_multi_head':
            #     # regularize everything except the multi-headed attention weights?
            #     if 'attn' not in name:
            #         reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
            # elif self.reg_all == 'except_multi_head_ief':
            #     if 'attn' not in name and 'logcell' not in name \
            #         and 'treatment_exp' not in name and 'control_layer' not in name:
            #         reg_loss += self.C*apply_reg(param, reg_type=self.reg_type)
        loss = torch.mean(reg_loss)
        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), Z_t, M), loss



class TransitionFunction(nn.Module):
    def __init__(self, hparams,
                 dim_stochastic,
                 dim_data,
                 dim_treat,
                 dim_hidden,
                 ttype,
                 augmented: bool = False,
                 alpha1_type: str = 'linear',
                 otype: str = 'linear',
                 add_stochastic: bool = False,
                 num_heads: int = 1,
                 zmatrix: str = 'identity'):
        super(TransitionFunction, self).__init__()
        self.hparams = hparams
        self.dim_stochastic  = dim_stochastic
        self.dim_treat       = dim_treat
        self.dim_hidden      = dim_hidden
        self.dim_data        = dim_data
        # Number of different lines of therapy to multiplex on (only for heterogenous ief_models)
        self.K               = 3
        self.ttype           = ttype



        self.t_mu               = AttentionIEFTransition( dim_stochastic,
                                                        dim_treat,  dim_hidden = self.dim_hidden ,
                                                            alpha1_type=alpha1_type,
                                                            num_heads=num_heads, dim_output=dim_stochastic)


        self.t_sigma            = nn.Linear(dim_stochastic+dim_treat, dim_stochastic)




    def apply_fxn(self, fxn, z, u, eps=0.):
        if 'Monotonic' in fxn.__class__.__name__ or 'LogCellTransition' in fxn.__class__.__name__ or 'LogCellKill' in fxn.__class__.__name__ \
            or 'TreatmentExp' in fxn.__class__.__name__ or 'GatedTransition' in fxn.__class__.__name__ or 'Synthetic' in fxn.__class__.__name__ \
            or 'MofE' in fxn.__class__.__name__ or 'Ablation1' in fxn.__class__.__name__ or 'Ablation2' in fxn.__class__.__name__ \
            or 'AttentionIEFTransition' in fxn.__class__.__name__:
            return fxn(z, u, eps)
        else:
            return fxn(torch.cat([z, u],-1))

    def forward(self, z, u, eps=0.):

        mu  = self.apply_fxn(self.t_mu, z, u, eps)
        sig = torch.nn.functional.softplus(self.apply_fxn(self.t_sigma, z, u))
        return mu, sig
