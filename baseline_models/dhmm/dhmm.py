"""
An implementation of a Deep Markov Model in Pyro based on reference [1].

Adopted from https://github.com/uber/pyro/tree/dev/examples/dmm  
         and https://github.com/clinicalml/structuredinference

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""
import sys
sys.path.append("baseline_models/dhmm")
import argparse
import time
from os.path import exists
import numpy as np
from baseline_models.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from modules import GatedTransition, PostNet, Encoder
from helper import reverse_sequence, sequence_mask
from sklearn.cluster import KMeans
from baseline_models.base import Model
from scipy.stats import chi2_contingency

class DHMM(Model):
    """
    The Deep Markov Model
    """
    def __init__(self, hparams , **kwargs):
        super(DHMM, self).__init__(hparams)
        self.save_hyperparameters()
    def init_model(self):
        dim_hidden  = self.hparams['dim_hidden']
        num_heads   = self.hparams['nheads']
        z_dim = self.hparams['dim_stochastic']
        d_gene = self.hparams['d_gene']
        d_x    = self.hparams['d_x']
        d_treat   = self.hparams['d_treat']
        # post_approx = self.hparams['post_approx']
        # inftype     = self.hparams['inftype']
        etype = self.hparams['etype']
        ttype = self.hparams['ttype']
        # augmented   = self.hparams['augmented']
        # alpha1_type = self.hparams['alpha1_type']

        combiner_type = self.hparams['combiner_type']

        # self.emitter = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
        #     nn.Linear(z_dim, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, d_x),
        #     # nn.Sigmoid()
        # )

        if etype == 'lin':
            self.e_mu    = nn.Linear(z_dim, d_x)
            self.e_sigma = nn.Linear(z_dim, d_x)
        elif etype  == 'nl':
            dim_hidden   = self.trial.suggest_int('dim_hidden',100,500)
            emodel       = nn.Sequential(nn.Linear(z_dim, dim_hidden), nn.ReLU(True))
            self.e_mu    = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
            self.e_sigma = nn.Sequential(emodel, nn.Linear(dim_hidden, d_x))
        else:
            raise ValueError('bad etype')
    
        self.trans = GatedTransition(z_dim, dim_hidden)
        self.postnet = PostNet(z_dim, dim_hidden)
        self.rnn = Encoder(None, d_x, dim_hidden, False, 1)
                   #nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu', \
                   #batch_first=True, bidirectional=False, num_layers=1)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, dim_hidden))

    def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
        one = torch.ones(1, device=mu1.device)
        if mu2 is None: mu2=torch.zeros(1, device=mu1.device)
        if logvar2 is None: logvar2=torch.zeros(1, device=mu1.device)
        return torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)
    def p_X_Z(self, Zt):

        mu          = self.e_mu(Zt)
        sigma       = torch.nn.functional.softplus(self.e_sigma(Zt))

        return mu, sigma

    def infer(self, x, x_rev, x_lens, M):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()

        _, rnn_out = self.rnn(x_rev, x_lens, h_0) # push the observed x's through the rnn;
        rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        for t in range(T_max):
            z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            z_t, z_mu, z_logvar = self.postnet(z_prev, rnn_out[:,t,:]) #q(z_t | z_{t-1}, x_{t:T})
            kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            # logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)
            p_x_mu, p_x_std = self.p_X_Z(z_t) # p(x_t|z_t)

            masked_nll         = masked_gaussian_nll_3d(x[:,t,:], p_x_mu, p_x_std, M[:,t,:])
            full_masked_nll    = masked_nll
            masked_nll         = masked_nll.sum(-1).sum(-1)
            # self.masked_nll_mean = ((masked_nll/full_masked_nll.shape[-1])/m_t.sum(-1)).mean()
            
            # rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
            # rec_losses[:,t] = rec_loss.mean(dim=1)
            z_prev = z_t
        x_mask = sequence_mask(x_lens)
        x_mask = x_mask.gt(0).view(-1)
        # rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
        return masked_nll, kl_loss

    def forward(self,  X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask, anneal = 1.):
        x = X
        x_rev = torch.flip(x, dims=[1])
        x_lens = torch.IntTensor([x.shape[1] for i in range(len(x))])


        masked_nll, kl_loss = self.infer(x, x_rev, x_lens, Mask)
        loss = masked_nll + anneal*kl_loss

        return (loss, masked_nll, kl_loss, None, None), loss

    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss
    def infer_z(self, x, x_rev, x_lens):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()

        _, rnn_out = self.rnn(x_rev, x_lens, h_0) # push the observed x's through the rnn;
        rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        z_lst = []
        for t in range(T_max):
            z_t, z_mu, z_logvar = self.postnet(z_prev, rnn_out[:,t,:]) #q(z_t | z_{t-1}, x_{t:T})
            z_lst.append(z_mu.detach())
            z_prev = z_mu
        z_lst = torch.stack(z_lst).transpose(1,0)
        return z_lst


    def predict(self, X, Trt, V_gene_cluster, S_gene_info, Genetic, Z_gt, Mask):
        x = X
        x_rev = torch.flip(x, dims=[1])
        x_lens = torch.IntTensor([x.shape[1] for i in range(len(x))])

        Z_t = self.infer_z(x, x_rev, x_lens)
        d_z = Z_t.shape[-1]
        z_lst_sel = Z_t[:, 1:, 0:1].masked_select(Mask[:, 1:, 0:1].expand(-1, -1, d_z).bool()).reshape(-1, d_z)
        kmeans = KMeans(n_clusters=self.hparams['n_ggroup'], random_state=1, n_init="auto").fit( z_lst_sel.detach().cpu().numpy())
        cluster_pred = kmeans.labels_

        y_sel = Z_gt[:, 1:].masked_select(Mask[:, 1:, 0].bool()).detach().cpu().numpy()

        result = np.zeros((len(set(y_sel)), len(set(y_sel))))
        for i in range(len(y_sel)):
            result[int(y_sel[i]), int(cluster_pred[i])] += 1
        chi2 = chi2_contingency(result)[0]

        return Z_t, chi2, result

class DHMM_archived(nn.Module):
    """
    The Deep Markov Model
    """
    def __init__(self, config ):
        super(DHMM, self).__init__()
        self.input_dim = config['input_dim']
        self.z_dim = config['z_dim']
        self.emission_dim = config['emission_dim']
        self.trans_dim = config['trans_dim']
        self.rnn_dim = config['rnn_dim']
        self.clip_norm = config['clip_norm']

        self.emitter = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.z_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.input_dim),
            nn.Sigmoid()
        )
        self.trans = GatedTransition(self.z_dim, self.trans_dim)
        self.postnet = PostNet(self.z_dim, self.rnn_dim)
        self.rnn = Encoder(None, self.input_dim, self.rnn_dim, False, 1)
                   #nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu', \
                   #batch_first=True, bidirectional=False, num_layers=1)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))

    def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
        one = torch.ones(1, device=mu1.device)
        if mu2 is None: mu2=torch.zeros(1, device=mu1.device)
        if logvar2 is None: logvar2=torch.zeros(1, device=mu1.device)
        return torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)

    def infer(self, x, x_rev, x_lens):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()

        _, rnn_out = self.rnn(x_rev, x_lens, h_0) # push the observed x's through the rnn;
        rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        for t in range(T_max):
            z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            z_t, z_mu, z_logvar = self.postnet(z_prev, rnn_out[:,t,:]) #q(z_t | z_{t-1}, x_{t:T})
            kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)
            rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
            rec_losses[:,t] = rec_loss.mean(dim=1)
            z_prev = z_t
        x_mask = sequence_mask(x_lens)
        x_mask = x_mask.gt(0).view(-1)
        rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
        return rec_loss, kl_loss

    def train_AE(self, x, x_rev, x_lens, kl_anneal):
        self.rnn.train() # put the RNN back into training mode (i.e. turn on drop-out if applicable)

        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_anneal*kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()

        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}

    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss

    def generate(self, x, x_rev, x_lens):
        """
        generation model p(x_{1:T} | z_{1:T}) p(z_{1:T})
        """
        batch_size, _, x_dim = x.size() # number of time steps we need to process in the mini-batch
        T_max = x_lens.max()
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0)) # set z_prev=z_0 to setup the recursive conditioning in p(z_t|z_{t-1})
        for t in range(1, T_max + 1):
            # sample z_t ~ p(z_t | z_{t-1}) one time step at a time
            z_t, z_mu, z_logvar = self.trans(z_prev) # p(z_t | z_{t-1})
            p_x_t = F.sigmoid(self.emitter(z_t))  # compute the probabilities that parameterize the bernoulli likelihood
            x_t = torch.bernoulli(p_x_t) #sample observe x_t according to the bernoulli distribution p(x_t|z_t)
            z_prev = z_t

