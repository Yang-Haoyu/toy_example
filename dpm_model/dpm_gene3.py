from distutils.util import strtobool
from baseline_models.base import Model
from baseline_models.utils import *
from baseline_models.inference import RNN_STInf
from pyro.distributions import Normal, Independent
from sklearn.cluster import KMeans

from dpm_model.vae import MultiVAE, loss_function, MultiVAE2
from baseline_models.multi_head_att import MultiHeadedAttention
from scipy.stats import chi2_contingency


class DPM_vae3(Model):
    def __init__(self, trial, **kwargs):
        super(DPM_vae3, self).__init__(trial)
        self.save_hyperparameters()

    def init_model(self):
        dim_hidden  = self.hparams['dim_hidden']


        dim_stochastic = self.hparams['dim_stochastic']
        d_gene = self.hparams['d_gene']
        d_x    = self.hparams['d_x']
        d_treat   = self.hparams['d_treat']


        etype = self.hparams['etype']
        combiner_type = self.hparams['combiner_type']


        # Inference Network

        self.inf_vae = MultiVAE([self.hparams['n_ggroup'], dim_hidden, d_gene])


        # self.inf_noise = np.abs(self.hparams['inf_noise'])
        self.inf_network    = RNN_STInf(self.hparams, d_gene, d_x, d_treat, dim_hidden, dim_stochastic,
                                        nl='relu', combiner_type = combiner_type)

        # Emission Function
        if etype == 'lin':
            self.e_mu    = nn.Linear(dim_stochastic, d_x)
            self.e_sigma = nn.Linear(dim_stochastic, d_x)
        elif etype  == 'nl':
            emodel       = nn.Sequential(nn.Linear(dim_stochastic, dim_hidden), nn.ReLU(True))
            self.e_mu    = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
            self.e_sigma = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
        else:
            raise ValueError('bad etype')

        # Transition Function
        self.transition_fxn = TransitionFunction3(self.hparams, dim_stochastic, d_x, d_treat, dim_hidden, d_gene)


        self.prior_W        = nn.Linear(d_treat+d_x+d_gene, dim_stochastic)
        self.prior_sigma    = nn.Linear(d_treat+d_x+d_gene, dim_stochastic)



    def p_Z1(self, B, X0, A0):
        inp_cat = torch.cat([B, X0, A0], -1)
        mu      = self.prior_W(inp_cat)
        sigma   = torch.nn.functional.softplus(self.prior_sigma(inp_cat))
        p_z_bxa = Independent(Normal(mu, sigma), 1)
        return p_z_bxa

    def p_X_Z(self, Zt):

        mu          = self.e_mu(Zt)
        sigma       = torch.nn.functional.softplus(self.e_sigma(Zt))

        return mu, sigma

    def p_Zt_Ztm1(self, X, Zt, Trt, Genetic, V=None):
        X0 = X[:,0,:]
        Trt0 = Trt[:,0,:]

        Tmax     = Zt.shape[1]
        prev_Z = Zt[:,:-1,:]
        cur_Trt = Trt[:,1:Tmax,:]

        # use x0, t0, demographic to generate z0
        inp0_cat  = torch.cat([Genetic, X0, Trt0], -1)

        mu1      = self.prior_W(inp0_cat)[:,None,:]
        sig1     = torch.nn.functional.softplus(self.prior_sigma(inp0_cat))[:,None,:]


        G_sequence = Genetic[:,None,:].repeat(1,cur_Trt.shape[1],1)
        mu2T, sig2T = self.transition_fxn(prev_Z, cur_Trt, G_sequence, v = V, s = self.inf_vae.p_layers.weight.T)

        mu, sig     = torch.cat([mu1,mu2T],1), torch.cat([sig1,sig2T],1)
        return Independent(Normal(mu, sig), 1)


    def get_loss(self, X, Trt,  Genetic, Mask, anneal = 1.):
        _, _, lens         = get_masks(Mask)
        Genetic, X, Trt, Mask = Genetic[lens>1], X[lens>1], Trt[lens>1], Mask[lens>1]
        m_t, _, _      = get_masks(Mask[:,1:,:])
        
        """==================== inference network ===================="""
        
        # VAE part, only for perdpm use
        recon_batch, mu_v, logvar_v = self.inf_vae(Genetic)
        loss_vae = loss_function(recon_batch, Genetic, mu_v, logvar_v, anneal)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar_v - mu_v.pow(2) - logvar_v.exp(), dim=1))
        loss_vae = nn.MSELoss()(recon_batch, Genetic) + self.hparams['_beta']*KLD
        

        # SSM part
        Z_t, q_zt, _          = self.inf_network(X, Trt, Mask, Genetic)
        Tmax               = Z_t.shape[1]

        """==================== generative network ===================="""
        # emission
        p_x_mu, p_x_std    = self.p_X_Z(Z_t)

        # trainsition
        p_zt               = self.p_Zt_Ztm1(X, Z_t, Trt, Genetic, V=mu_v)

        """compute loss"""
        masked_nll         = masked_gaussian_nll_3d(X[:,1:Tmax+1,:], p_x_mu, p_x_std, Mask[:,1:Tmax+1,:])
        full_masked_nll    = masked_nll
        masked_nll         = masked_nll.sum(-1).sum(-1)
        self.masked_nll_mean = ((masked_nll/full_masked_nll.shape[-1])/m_t.sum(-1)).mean()


        kl_t       = q_zt.log_prob(Z_t)-p_zt.log_prob(Z_t)
        masked_kl_t= (m_t[:,:Tmax]*kl_t).sum(-1)
        self.masked_kl_mean = (masked_kl_t/m_t.sum(-1)).mean()
        # neg_elbo   = masked_nll + anneal*masked_kl_t + loss_vae

        if self.hparams["add_vae_loss"]:
            neg_elbo   = masked_nll + anneal*masked_kl_t + loss_vae
        else:
            neg_elbo   = masked_nll + anneal*masked_kl_t

        return neg_elbo, masked_nll, masked_kl_t, Z_t, Mask


    def _infer_z(self, B, X, A, M):
        _, _, lens         = get_masks(M)
        B, X, A, M = B[lens>1], X[lens>1], A[lens>1], M[lens>1]
        m_t, m_g_t, _   = get_masks(M[:,1:,:])
        z_mu          = self.inf_network.infer_forward(X, A, M, B)
        return z_mu, M

    def predict(self, X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask):
        Z_t, Mask = self._infer_z(Genetic, X, Trt,  Mask)


        # mi_a, mi_n = self.mi( Z_t.detach(), Mask, Z_gt)


        # model.eval()
        # for test_data in dm.test_dataloader():
        #     break
        # X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask = test_data
        # mi_a, mi_n, Z_t, mu_v, Z_gt = model.predict(*test_data)

        d_z = Z_t.shape[-1]
        z_lst_sel = Z_t.masked_select(Mask[:, 1:, 0:1].expand(-1, -1, d_z).bool()).reshape(-1, d_z)
        kmeans = KMeans(n_clusters=self.hparams['n_ggroup'], random_state=1, n_init="auto").fit( z_lst_sel.detach().cpu().numpy())
        cluster_pred = kmeans.labels_

        y_sel = Z_gt[:, 1:].masked_select(Mask[:, 1:, 0].bool()).detach().cpu().numpy()



        result = np.zeros((len(set(y_sel)), len(set(y_sel))))
        for i in range(len(y_sel)):
            result[int(y_sel[i]), int(cluster_pred[i])] += 1
        chi2 = chi2_contingency(result)[0]

        return  Z_t, chi2, result



    def forward(self, X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask, anneal = 1.):
        # M sequence mask [N x t]
        neg_elbo, masked_nll, kl, Z_t, Mask = self.get_loss( X, Trt, Genetic,  Mask, anneal = anneal)
        
        reg_loss   = torch.mean(neg_elbo)

        for name,param in self.named_parameters():

            reg_loss += self.hparams['C']*apply_reg(param, reg_type=self.hparams['reg_type'])
                
        loss = torch.mean(reg_loss)
        if "pdpm" in self.hparams['ttype']:
            loss += 0.01 * torch.norm(self.inf_vae.p_layers.weight,  p = 1)

        return (torch.mean(neg_elbo), torch.mean(masked_nll), torch.mean(kl), Z_t, Mask), loss



class TransitionFunction3(nn.Module):
    def __init__(self, hparams, dim_z, dim_data, dim_treat, dim_hidden, d_g,
                 otype: str = 'linear'):
        super(TransitionFunction3, self).__init__()
        self.hparams = hparams

        self.dim_z  = dim_z
        self.dim_treat       = dim_treat
        self.dim_hidden      = dim_hidden
        self.dim_data        = dim_data
        self.d_g = d_g
        # Number of different lines of therapy to multiplex on (only for heterogenous ief_models)

        if self.hparams["setting"] == 1:
            print("Using setting 1")
            self.t_mu = GeneTransition1(hparams, dim_z, dim_treat, d_g, dim_hidden = self.dim_hidden,
                                    otype=otype)
        elif self.hparams["setting"] == 2:
            print("Using setting 2")
            self.t_mu = GeneTransition2(hparams, dim_z, dim_treat, d_g, dim_hidden = self.dim_hidden,
                                    otype=otype)
        elif self.hparams["setting"] == 3:
            print("Using setting 3")
            self.t_mu = GeneTransition3(hparams, dim_z, dim_treat, d_g, dim_hidden = self.dim_hidden,
                                    otype=otype)
        else:
            raise NotImplementedError
        self.t_sigma = nn.Linear(dim_z + dim_treat, dim_z)

    def forward(self, prev_z, cur_Trt, g_seq, v = None, s = None):
        mu  = self.t_mu(prev_z, cur_Trt, g_seq, v = v,s = s)
        sig = torch.nn.functional.softplus(self.t_sigma(torch.cat([prev_z, cur_Trt],-1)))
        return mu, sig

class GeneTransition1(nn.Module):
    def __init__(self, hparams, dim_z, dim_treat, dim_gene,
                 dim_hidden = 300,  otype = 'linear', zmatrix=None):
        super(GeneTransition1, self).__init__()
        self.hparams = hparams
        self.dim_z       = dim_z
        self.dim_treat       = dim_treat
        self.dim_gene        = dim_gene
        self.zmatrix          = zmatrix


        if self.hparams['gpu']:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.out_layer    = nn.Linear(dim_hidden, dim_z)


        self.control_layer    = nn.Linear(dim_gene + dim_treat, dim_hidden)
        self.inp_layer        = nn.Linear(dim_z , dim_hidden)

        if self.hparams['ttype'] == "ief_g" or self.hparams['ttype'] == "ief":
            self.cluster_fn = [nn.Sequential(nn.ReLU(True), nn.Linear(2*dim_hidden, dim_hidden)).to(device)  for i in range(self.hparams['n_ggroup'])]
            # self.progression_fn = [nn.Sequential(nn.ReLU(True), nn.Linear(2*dim_hidden, dim_hidden))  for i in range(self.hparams['n_ggroup'])]

        else:
            self.cluster_fn = [nn.Sequential(nn.ReLU(True), nn.Linear(2*dim_hidden + dim_gene, dim_hidden)).to(device)  for i in range(self.hparams['n_ggroup'])]
            # self.progression_fn = [nn.Sequential(nn.ReLU(True), nn.Linear(2*dim_hidden, dim_hidden))  for i in range(self.hparams['n_ggroup'])]
            self.noisy = torch.randn(self.hparams['n_ggroup'], self.dim_gene )
        self.attn = MultiHeadedAttention(self.hparams["nheads"], dim_hidden)
        
    def forward(self, prev_z, cur_Trt, g_seq, v = None, s = None):
        inp        = self.inp_layer(prev_z)

        inp_con =  self.control_layer(torch.cat([g_seq, cur_Trt],-1))


        if "pdpm" in self.hparams['ttype']:

            gene_map_tmp = [self.cluster_fn[i](
                torch.cat([inp, inp_con, s[i][None,None,:].repeat(inp_con.shape[0], inp_con.shape[1],1)], -1)
                ) for i in range(self.hparams['n_ggroup'])]


        elif self.hparams['ttype'] == "ief" or self.hparams['ttype'] == "ief_g":
            gene_map_tmp = [self.cluster_fn[i](torch.cat([inp, inp_con], -1)) for i in range(self.hparams['n_ggroup'])]
            
        else:
            raise NotImplementedError
        
        gene_map = torch.stack(gene_map_tmp)
        value = gene_map.transpose(1,0).transpose(2,1)
        key = gene_map.transpose(1,0).transpose(2,1)
        query = inp[...,None,:]
        if self.hparams['ttype'] == "pdpm_attn":
            # value = torch.einsum("ijkl, ik -> ijkl", value, torch.softmax(v, dim=-1))
            out   = self.attn.forward(query, key, value, use_matmul=False).squeeze()

        elif self.hparams['ttype'] == "ief" :
            out   = self.attn.forward(query, key, value, use_matmul=False).squeeze()
        
        elif self.hparams['ttype'] == "pdpm_hybrid":
            value = torch.einsum("ijkl, ik -> ijkl", value, torch.softmax(v, dim=-1))
            out   = self.attn.forward(query, key, value, use_matmul=False).squeeze()
        elif self.hparams['ttype'] == "pdpm_hybrid2":
            value = torch.einsum("ijkl, ik -> ijkl", value, torch.softmax(v, dim=-1))
            out   = self.attn.forward(query, value, value, use_matmul=False).squeeze()
        elif self.hparams['ttype'] == "pdpm_vae":
            out = torch.einsum("ijkl, ik -> ijl", value, torch.softmax(v, dim=-1))
        elif self.hparams['ttype'] == "ief_g":
            out = torch.einsum("ijkl, ik -> ijl", value, torch.softmax(v, dim=-1))
        else:
            raise NotImplementedError
        



        return self.out_layer(out)
