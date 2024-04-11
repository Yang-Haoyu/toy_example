import torch
import math
import numpy as np
import pickle 
from torch import nn
import torch.functional as F
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import sys, os
from torch.autograd import grad
from baseline_models.iefs.logcellkill import LogCellKill
from baseline_models.iefs.treatexp import TreatmentExponential
from baseline_models.multi_head_att import MultiHeadedAttention

def te_matrix():
    te_matrix = np.array([[-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 0, -1, 0, 1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  0], 
                          [-1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    return te_matrix


class AttentionIEFTransition(nn.Module):
    def __init__(self, dim_stochastic,
                 dim_treat,
                 dim_hidden=300,
                dim_output = -1, 
                dim_subtype = -1, 
                dim_input = -1, 
                 response_only=True,
                 alpha1_type='linear',
                 add_stochastic=False,
                 num_heads=1,
                 zmatrix=None):
        super(AttentionIEFTransition, self).__init__()
        self.response_only = response_only

        self.dim_subtype = dim_subtype

        self.dim_treat = dim_treat
        self.zmatrix = zmatrix

        if dim_input == -1: 
            dim_input = dim_stochastic
        if dim_output == -1: 
            dim_output = dim_stochastic
        self.out_layer = nn.Linear(dim_hidden, dim_output)

        # self.control_layers   = nn.ModuleList([nn.Linear(dim_treat, dim_stochastic) for _ in range(2)])
        self.control_layer = nn.Linear(dim_treat, dim_hidden)
        self.inp_layer = nn.Linear(dim_input, dim_hidden)

        self.attn = MultiHeadedAttention(num_heads, dim_hidden)
        self.logcell = LogCellKill(dim_hidden, dim_treat, mtype='logcellkill_1', response_only=True,
                                   alpha1_type=alpha1_type)
        self.treatment_exp = TreatmentExponential(dim_hidden, dim_treat, response_only=True,
                                                  alpha1_type=alpha1_type, add_stochastic=add_stochastic)

    def forward(self, inpx, con, eps=0.):
        inp = self.inp_layer(inpx)
        # out_linears= [torch.tanh(l(con))[...,None] for l in self.control_layers]
        # out_te     = [t(inp,con,eps=eps)[...,None] for t in self.treatment_exps]
        out_linear = inp * torch.tanh(self.control_layer(con))
        #         out_linear = torch.tanh(self.control_layer(con))
        out_te = self.treatment_exp(inp, con, eps=eps)
        out_logcell = self.logcell(inp, con)
        # f   = tuple(out_linears + [out_te, out_logcell])
        value = torch.cat((out_linear[..., None], out_te[..., None], out_logcell[..., None]), dim=-1).transpose(-2, -1)
        key = torch.cat((out_linear[..., None], out_te[..., None], out_logcell[..., None]), dim=-1).transpose(-2, -1)
        query = inp[..., None, :]
        out = self.attn.forward(query, key, value, use_matmul=False).squeeze()
        #         if len(out.shape) == 2:
        #             out = out[:,None,:]
        return self.out_layer(out)
        # if self.response_only:
        #     return out
        # else:
        #     return self.out_layer(torch.matmul(inp, self.linear_layer)+out)


class AttentionIEFTransition2(nn.Module):
    def __init__(self, hparams,
                 dim_stochastic,
                       dim_treat, 
                       dim_hidden = 300, 
                       dim_output = -1, 
                       dim_subtype = -1, 
                       dim_input = -1, 

                       response_only = True, 

                       otype = 'linear', 
                       alpha1_type = 'linear', 
                       add_stochastic = False, 
                       num_heads = 1, 
                       zmatrix=None):
        super(AttentionIEFTransition, self).__init__()
        self.response_only    = response_only

        self.dim_output       = dim_output
        self.dim_subtype      = dim_subtype
        self.dim_input        = dim_input
        self.dim_treat        = dim_treat
        self.zmatrix          = zmatrix
        self.hparams = hparams

        if otype == 'linear': 
            self.out_layer    = nn.Linear(dim_hidden, dim_stochastic)
        elif otype == 'nl': 
            omodel            = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),nn.ReLU(True))
            self.out_layer    = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_stochastic))
        elif otype == 'identity': # useful for FOMM
            self.out_layer    = nn.Sequential()
        # self.control_layers   = nn.ModuleList([nn.Linear(dim_treat, dim_stochastic) for _ in range(2)]) 
        # self.control_layer    = nn.Linear(dim_treat, dim_stochastic)

        self.control_layer    = nn.Linear(dim_treat, dim_hidden)

        if self.hparams['ttype'] in ["pdpm_attn","ief_g","pdpm_vae", "pdpm_hybrid"]:
            self.inp_layer = nn.Linear(self.hparams['d_gene'] + dim_stochastic, dim_hidden)
            self.treatment_exp = TreatmentExponential(dim_hidden, dim_treat, response_only=True,
                                                      alpha1_type=alpha1_type, add_stochastic=False)

            # self.control_layer    = nn.Linear(dim_gene + dim_stochastic, dim_hidden)
        elif self.hparams['ttype'] == "ief":
            self.inp_layer        = nn.Linear(dim_stochastic, dim_hidden)
            self.treatment_exp = TreatmentExponential(dim_hidden, dim_treat, response_only=True,
                                                      alpha1_type=alpha1_type, add_stochastic=False)

        else:
            raise NotImplementedError



        self.attn = MultiHeadedAttention(num_heads, dim_hidden)
        self.logcell      = LogCellKill(dim_hidden, dim_treat, mtype='logcellkill_1', response_only = True, alpha1_type=alpha1_type)

    def forward(self, inpx, con, eps=0.):
        inp        = self.inp_layer(inpx)
        # out_linears= [torch.tanh(l(con))[...,None] for l in self.control_layers]
        # out_te     = [t(inp,con,eps=eps)[...,None] for t in self.treatment_exps]
        out_linear = inp*torch.tanh(self.control_layer(con))
#         out_linear = torch.tanh(self.control_layer(con))
        out_te     = self.treatment_exp(inp, con.repeat(1,1,4), eps=eps)
        out_logcell= self.logcell(inp, con)
        # f   = tuple(out_linears + [out_te, out_logcell])
        value = torch.cat((out_linear[...,None], out_te[...,None], out_logcell[...,None]), dim=-1).transpose(-2,-1)
        key   = torch.cat((out_linear[...,None], out_te[...,None], out_logcell[...,None]), dim=-1).transpose(-2,-1)
        query = inp[...,None,:]
        out   = self.attn.forward(query, key, value, use_matmul=False).squeeze()
#         if len(out.shape) == 2: 
#             out = out[:,None,:]
        return self.out_layer(out)
        # if self.response_only:
        #     return out
        # else: 
        #     return self.out_layer(torch.matmul(inp, self.linear_layer)+out)

