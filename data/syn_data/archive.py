# import torch
# from methods.nn_utils import MLP
# from collections import Counter
# from numpy.polynomial.polynomial import polyval
# self = SynDataDPM(0, 3, 10, 4, 20, max_seq_len=10, trans_type="nn")
# nsample = 1000
# nlayer_trans = 2
# scaled_trans_matrix = True
# self.nsample = nsample
# # sample the prior parameters
# self.G = self._sample_gene()
# self.x_orig = self._get_longitudinal_data()
# self.trt = self._set_trt()
# self.x_after_trt = self._apply_trt()
#
# import matplotlib.pyplot as plt
# i = 2
# k = 4
# for i in range(10):
#     plt.figure()
#     plt.plot(self.x_after_trt[i,:,k])
#     plt.plot(self.x_orig[i,:,k])
#     plt.show()
#
# self.pi = self._create_init_state_dist()
# self.P = self._create_trans(nlayer_trans, scaled_trans_matrix)
#
#
# class SynDataDPM:
#     def __init__(self, seed, n_gene_group, d_gene, d_z, d_x, max_seq_len=10, trans_type="nn", discrete_v=True,num_trt = 1):
#         """
#
#         :param seed: control the randomness
#         :param ncluster: number of genetic clusters
#         :param d_gene: dimension of genetic information
#         :param d_z: dimension of hidden state
#         :param trans_type: type of transition function, choose from ["nn", "linear", "individual_nn"]
#         """
#
#         self.seed = seed
#         self.rng = np.random.default_rng(self.seed)
#         self.n_gene_group = n_gene_group
#         self.d_gene = d_gene
#         self.d_z = d_z
#         self.d_x = d_x
#         self.trans_type = trans_type
#         self.max_seq_len = max_seq_len
#         self.num_trt = num_trt
#
#         self.discrete_v = discrete_v
#
#         self.mu_x = self.rng.normal(size=(self.n_gene_group, self.d_z, self.d_x))
#         self.var_x = np.abs(self.rng.normal(size=(self.n_gene_group, self.d_z, self.d_x)) / 10)
#
#         self.mu_gene = None  # ncluster x d_gene
#         self.cov_gene = None  # ncluster x d_gene x d_gene
#         self.pi = None
#         self.P = None
#         self.G = None
#         self.V = None
#         self.nsample = None
#         self.gene_cluster = None
#         self.trt = None
#
#     def _sample_gene(self):
#         """
#
#         :param nsample: number of sample
#         :return: gene_cluster, a ncluster x n (#patient) x g (dimension of gene) tensor
#         """
#         # set the mean and covariance of genetic information
#         self.mu_gene = self.rng.uniform(-10, 10, (self.n_gene_group, self.d_gene))  # mu_g
#         self.cov_gene = np.ones_like(self.mu_gene)  # sigma_g
#
#         self.V = self.rng.normal(size=(self.nsample, self.n_gene_group))
#         self.V_discrete = (self.V == np.max(self.V, axis=1)[:, None]).astype(int)
#         self.V_continuous = np.exp(self.V) / np.sum(np.exp(self.V), axis=1, keepdims=True)
#
#         gene_cluster = []
#
#         for c in range(self.n_gene_group):
#             # get the parameter for c-th genetic group
#             mu_gene_cluster = self.mu_gene[c]
#             cov_gene_cluster = self.cov_gene[c]
#
#             # genetic info is sampled from multivariate Gaussian distribution
#             gene_info = self.rng.multivariate_normal(mu_gene_cluster, np.diag(cov_gene_cluster))
#             gene_cluster.append(gene_info)
#         self.gene_cluster = np.array(gene_cluster)
#
#         if self.discrete_v:
#             G = self.V_discrete @ self.gene_cluster
#         else:
#             G = self.V_continuous @ self.gene_cluster
#
#         # gene_cluster is a ncluster x n x g tensor
#         return G
#
#     def _get_longitudinal_data(self):
#         Xvals = np.arange(self.max_seq_len)
#         up = [-2, 0.0002, 0.1]
#         down = [4, -0.1, -0.1]
#
#         # if not lrandom:
#         #     up = [-2, 0.0002, 0.1]
#         #     down = [4, -0.1, -0.1]
#         # else:
#         #     a = np.random.uniform(0.05, 0.15)
#         #     bu = np.random.uniform(0.0001, 0.01)
#         #     bd = np.random.uniform(-0.3, -0.05)
#         #     cu = np.random.uniform(-5, -1)
#         #     cd = np.random.uniform(1, 5)
#         #     up = [cu, bu, a];
#         #     down = [cd, bd, -a]
#
#         subtype_fxn = {}
#         subtype_fxn[0] = [down, down]
#         subtype_fxn[1] = [down, up]
#         subtype_fxn[2] = [up, down]
#         subtype_fxn[3] = [up, up]
#         add_feats = self.d_x - 2
#         for k in subtype_fxn.keys():
#             add = []
#             for num in range(add_feats):
#                 a = np.random.uniform(0.05, 0.15)
#                 bu = np.random.uniform(0.0001, 0.01)
#                 bd = np.random.uniform(-0.3, -0.05)
#                 cu = np.random.uniform(-5, -1)
#                 cd = np.random.uniform(1, 5)
#                 up = [cu, bu, a]
#                 down = [cd, bd, -a]
#                 if k == 0:
#                     add.append(up)
#                 elif k == 1:
#                     if num % 2 == 0:
#                         add.append(down)
#                     else:
#                         add.append(up)
#                 elif k == 2:
#                     if num % 2 == 0:
#                         add.append(up)
#                     else:
#                         add.append(down)
#                 else:
#                     add.append(down)
#
#             subtype_fxn[k] = subtype_fxn[k] + add
#
#         datalist = []
#         stypelist = np.argmax(self.V_discrete, axis=1)
#         for s in stypelist:
#             fxn = subtype_fxn[s]
#             if add_feats == 0:
#                 dim0 = np.array([polyval(x, fxn[0]) for x in Xvals])
#                 dim1 = np.array([polyval(x, fxn[1]) for x in Xvals])
#                 both = np.concatenate([dim0[..., None], dim1[..., None]], axis=-1)
#                 datalist.append(both)
#             else:
#                 dims = [np.array([polyval(x, fxn[num]) for x in Xvals]) for num in range(len(fxn))]
#                 tot = np.concatenate([dim[..., None] for dim in dims], axis=-1)
#                 datalist.append(tot)
#
#         data = np.array(datalist)
#         data = 0.5 * data + 0.5 * self.rng.random(data.shape)
#         return data
#
#     def _set_trt(self):
#         ldata = self.x_orig
#         # treatment line
#         a = np.zeros((ldata.shape[0], ldata.shape[1], self.num_trt+4))
#         # time to apply treatment
#         t = np.random.randint(low=4, high=a.shape[1], size=a.shape[0])
#
#         for i in range(a.shape[0]):
#             if t[i] == a.shape[1] - 1:
#                 t[i] -= 3
#             #         l = choice(np.arange(2,5), 1, p=[0.6, 0.3, 0.1])
#             l = self.num_trt + 1  # fix the line of therapy as first line for now
#             # randomly select how many of the num_trt should be given
#             if self.num_trt == 1:
#                 trt_idxs = np.array([1])
#             else:
#                 num_select = np.random.randint(low=2, high=self.num_trt + 1)
#                 # then, pick which indices should be turned "on"
#                 trt_idxs = np.random.randint(low=1, high=self.num_trt + 1, size=num_select)
#             a[i, t[i], trt_idxs] = 1.
#             a[i, t[i], l] = 1.
#
#         return a
#     def _apply_trt(self):
#         alpha_1s = [10, 5, -5, -10]
#         alpha_2 = 0.6
#         alpha_3s = [0.6, 0.8, 0.9]
#         gamma = 2
#         b = [3, 4.5, 6]
#         s = [1, 1, -1, -1]
#         params = {}
#         params[1] = [alpha_1s, alpha_3s, alpha_2, gamma, b]
#
#         for trt_idx in range(2, self.num_trt + 1):
#             l = np.random.randint(low=5, high=15)
#             alpha_1s = [l, int(0.5 * l), -int(0.5 * l), -l]
#             alpha_2 = np.random.uniform(low=0., high=1.)
#             alpha_3 = np.random.uniform(low=0., high=.7)
#             alpha_3s = [alpha_3, alpha_3 + 0.2, alpha_3 + 0.3]
#             gamma = np.random.randint(low=2, high=5)
#             params[trt_idx] = [alpha_1s, alpha_3s, alpha_2, gamma, b]
#
#         ldata = self.x_orig.copy()
#         subtype =  np.argmax(self.V_discrete, axis=1)
#         for i in range(ldata.shape[0]):
#             int_t = 0
#             l = 0
#             trt_select = []
#             for t in range(ldata.shape[1]):
#                 if t == 0:
#                     continue
#                 for k in range(1, self.num_trt + 1):
#                     alpha_1s, alpha_3s, alpha_2, gamma, b = params[k]
#                     # if received treatment at previous time
#                     if self.trt[i, t - 1, k] == 1.:
#                         if t + gamma < ldata.shape[1]:
#                             # gamma_i: when the treatment effect end
#                             gamma_i = t + gamma
#                             real_t1 = np.arange(0, gamma)
#                             real_t2 = np.arange(gamma, gamma + (ldata.shape[1] - gamma_i))
#                         else:
#                             gamma_i = ldata.shape[1] - 1
#                             real_t1 = np.arange(0, gamma - 1)
#                             real_t2 = np.arange(gamma - 1, gamma - 1 + (ldata.shape[1] - gamma_i))
#                         l = np.where(self.trt[i, t - 1, self.num_trt + 1:] == 1.)[0]
#                         alpha_1 = alpha_1s[int(subtype[i])]
#                         alpha_3 = alpha_3s[int(l)]
#                         base_0 = -alpha_1 / (1 + np.exp(alpha_2 * gamma_i / 2.))
#                         alpha_0 = (alpha_1 + 2 * base_0 - b[int(l)]) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
#                         ldata[i, t:gamma_i, :] += (
#                                     base_0 + alpha_1 / (1 + np.exp(-alpha_2 * (real_t1[:, None] - gamma_i / 2.))))
#                         ldata[i, gamma_i:, :] += (
#                                     b[int(l)] + alpha_0 / (1 + np.exp(alpha_3 * (real_t2[:, None] - 3 * gamma_i / 2.))))
#                         trt_select.append(k)
#                         int_t = t
#             # import pdb; pdb.set_trace()
#             self.trt[i, int_t:, trt_select] = 1.
#             self.trt[i, int_t - 1:, 0] = 1.
#             time_val = (np.cumsum(self.trt[i, int_t - 1:, 0])) * 0.1
#             self.trt[i, int_t - 1:, 0] = time_val
#             self.trt[i, int_t:, self.num_trt + 1:] = np.repeat(self.trt[i, int_t - 1, self.num_trt + 1:][None, ...],
#                                                           self.trt[i, int_t:, 0].shape[0],
#                                                    axis=0)
#         return ldata
#
#
#     def _create_init_state_dist(self):
#
#         # transform the genetic info into initial distribution \pi
#
#         # create a linear layer
#         w_init = self.rng.normal(size=(self.d_gene, self.d_z))
#         b_init = self.rng.normal(size=self.d_z)
#
#         # compute \pi by add a linear layer to the mean value of the genetic info
#         pi = np.abs(self.mu_gene @ w_init + b_init)
#         pi = pi / np.expand_dims(np.sum(pi, axis=1), axis=-1)
#
#         return pi
#
#     def _scale_matrix(self, scale=10, shrink_factor=10):
#         """
#         create a scale matrix to make transition matrix have small prob
#         to transit from healthy to the most severe state
#
#         :param scale:
#         :param shrink_factor:
#         :return:
#         """
#         scale_matrix = scale * np.ones((self.d_z, self.d_z))
#         for k in range(len(scale_matrix)):
#             for i in range(len(scale_matrix)):
#                 try:
#                     # shrink off diagonal elements
#                     scale_matrix[i][i + k] /= shrink_factor ** (k)
#                     scale_matrix[i + k][i] /= shrink_factor ** (k)
#                 except:
#                     pass
#         return scale_matrix
#
#     def _create_trans(self, nlayer, scaled=True):
#         if self.G is None:
#             print("Please generate genetic info first, try to use method _sample_gene()")
#             raise ValueError
#         if self.trans_type == "nn":
#             # all the genetic clusters share the same transition function
#             nn = MLP(self.d_gene, self.d_z ** 2, 2 * self.d_z ** 2, nlayer)
#             P = np.abs(nn(torch.from_numpy(self.mu_gene).float()).detach().cpu().numpy())
#             P = P.reshape(-1, self.d_z, self.d_z)
#         elif self.trans_type == "individual_nn":
#             # different genetic cluster have individual transition function
#             n_ggroup = self.mu_gene.shape[0]
#             nn_lst = [MLP(self.d_gene, self.d_z ** 2, 2 * self.d_z ** 2, nlayer) for i in range(n_ggroup)]
#
#             P = [torch.abs(nn_lst[i](torch.from_numpy(self.mu_gene[i]).float())) for i in range(n_ggroup)]
#             P = torch.stack(P).detach().cpu().numpy().reshape(-1, self.d_z, self.d_z)
#
#         elif self.trans_type == "linear":
#             w_trans = np.abs(self.rng.normal(size=(self.d_gene, self.d_z ** 2)))
#             b_trans = np.abs(self.rng.normal(size=(self.d_z ** 2)))
#             P = np.abs(self.mu_gene @ w_trans + b_trans).reshape(-1, self.d_z, self.d_z)
#         else:
#             raise NotImplementedError
#
#         assert P.shape == (self.mu_gene.shape[0], self.d_z, self.d_z)
#
#         if scaled:
#             # whether to scale the transition matrix
#             scale_matrix = self._scale_matrix()
#             P = P * scale_matrix
#
#         P = P / np.expand_dims(np.sum(P, axis=-1), -1)
#         return P
#
#     def _sample_z_init(self, state_prob, zsample_type="cat"):
#         if self.nsample is None:
#             print("Please set the number of sample before using _sample_z")
#             raise ValueError
#
#         if zsample_type == "cat":
#             return self.rng.choice(self.d_z, self.nsample, p=state_prob)
#         else:
#             raise NotImplementedError
#
#     def _sample_z(self, trans_mat, prev_state, zsample_type="cat"):
#         if self.nsample is None:
#             print("Please set the number of sample before using _sample_z")
#             raise ValueError
#
#         if zsample_type == "cat":
#             return np.array([self.rng.choice(self.d_z, 1, p=trans_mat[i]) for i in prev_state]).squeeze(-1)
#         else:
#             raise NotImplementedError
#
#     def _gen_z_seq(self, init_stat_prob, trans_mat, zsample_type="cat"):
#
#         z_seq = []
#         for tt in range(self.max_seq_len):
#             # initial state, using pi
#             if tt == 0:
#                 cur_state = self._sample_z_init(init_stat_prob, zsample_type=zsample_type)
#             else:
#                 prev_state = z_seq[-1]
#                 cur_state = self._sample_z(trans_mat, prev_state, zsample_type=zsample_type)
#             z_seq.append(cur_state)
#         return np.array(z_seq)
#
#     def _sample_x(self, k, cur_zs):
#         """
#
#         :param k: k-th genetic group
#         :return:
#         """
#         x_seq = []
#         for _sample in cur_zs:
#             x_sample = []
#             for tt_z in _sample:
#                 mu_x_tmp = self.mu_x[k][tt_z]
#                 var_x_tmp = self.var_x[k][tt_z]
#
#                 x_tmp = self.rng.multivariate_normal(mu_x_tmp, np.diag(var_x_tmp))
#                 x_sample.append(x_tmp)
#             x_seq.append(np.array(x_sample))
#         return np.array(x_seq)
#
#     def sample(self, nsample, nlayer_trans=2, scaled_trans_matrix=True):
#
#         """
#
#         :param nsample:
#         :param nlayer_trans:
#         :param scaled_trans_matrix:
#         :return:
#         """
#         self.nsample = nsample
#         # sample the prior parameters
#         self.G = self._sample_gene()
#         self.pi = self._create_init_state_dist()
#         self.P = self._create_trans(nlayer_trans, scaled_trans_matrix)
#
#         # generate z
#         self.z = []
#
#         for k in range(self.n_gene_group):
#             # sample the latent states for each genetic group
#             cur_zs = self._gen_z_seq(self.pi[k], self.P[k], zsample_type="cat").transpose()
#             self.z.append(cur_zs)
#
#         self.z = np.array(self.z)
#
#         # sample the observations
#         self.x = []
#         for k in range(self.n_gene_group):
#             self.x.append(self._sample_x(k, self.z[k]))
#
#         return self.x, self.G, self.z,
#
#     def check_z_distribution_match(self):
#
#         # Check the initial probability
#         init_prob_est = []
#         for k in range(self.n_gene_group):
#             # check init
#             z_init = Counter(self.z[k][:, 0])
#             init_prob_est.append([z_init[i] / np.sum(list(z_init.values())) for i in range(self.d_z)])
#
#         # Check the transition matrix
#         trans_mat_est = np.zeros_like(self.P)
#         for k in range(self.n_gene_group):
#             len_tmp = self.z[k].shape[1] - 1
#             for _interval in range(len_tmp):
#                 tmp = self.z[k][:, _interval:_interval + 2]
#                 for _sample in tmp:
#                     trans_mat_est[k][_sample[0]][_sample[1]] += 1
#         for _trans in trans_mat_est:
#             _trans /= np.expand_dims(np.sum(_trans, axis=1), axis=-1)
#
#         init_prob_est = np.array(init_prob_est)
#         trans_mat_est = np.array(trans_mat_est)
#
#         print("the error between est. and true initial latent state probability is {}".format(
#             np.linalg.norm(self.pi - init_prob_est)))
#         print("the error between est. and true transition matrix is {}".format(
#             np.linalg.norm(self.P - trans_mat_est)))
#
#         for k in range(self.n_gene_group):
#             print("for {}-th genetic group".format(k))
#             print("The true initial latent state probability is")
#             print(self.pi[k].round(2))
#             print("The estimated initial latent state probability is")
#             print(init_prob_est[k].round(2))
#
#             print("The true transition matrix is")
#             print(self.P[k].round(2))
#             print("The estimated transition matrix is")
#             print(trans_mat_est[k].round(2))
#             print("\n")
#
#     def check_x_distribution_match(self):
#
#         # Check the initial probability
#         x_mu_est = []
#         x_var_est = []
#         for k in range(self.n_gene_group):
#             x_mu_est_tmp = []
#             x_var_est_tmp = []
#
#             # check init
#             z_vals = self.z[k].reshape(-1)
#             x_vals = self.x[k].reshape(-1, self.x[k].shape[-1])
#
#             for i in range(self.d_z):
#                 x_sel = x_vals[z_vals == i]
#                 mu_est_i = np.mean(x_sel, axis=0)
#                 var_est_i = np.var(x_sel, axis=0)
#
#                 x_mu_est_tmp.append(mu_est_i)
#                 x_var_est_tmp.append(var_est_i)
#             x_mu_est.append(np.array(x_mu_est_tmp))
#             x_var_est.append(np.array(x_var_est_tmp))
#
#         x_mu_est = np.array(x_mu_est)
#         x_var_est = np.array(x_var_est)
#
#         print("the error between est. and true mean value of x is {}".format(
#             np.linalg.norm(self.mu_x - x_mu_est)))
#         print("the error between est. and true var value of x is {}".format(
#             np.linalg.norm(self.var_x - x_var_est)))
#
#         for k in range(self.n_gene_group):
#             print("for {}-th genetic group".format(k))
#             print("The true mean value of x is")
#             print(self.mu_x[k].round(2))
#             print("The estimated mean value of x is")
#             print(x_mu_est[k].round(2))
#
#             print("for {}-th genetic group".format(k))
#             print("The true var value of x is")
#             print(self.var_x[k].round(2))
#             print("The estimated var value of x is")
#             print(x_var_est[k].round(2))
#
#
# self = SynDataAttn(0, 3, 10, 4, 20, max_seq_len=10, trans_type="nn")
# nsample = 1000
# nlayer_trans = 2
# scaled_trans_matrix = True
# self.nsample = nsample
# # sample the prior parameters
# self.G = self._sample_gene()
# self.x_orig = self._get_longitudinal_data()
# self.trt = self._set_trt()
# self.x_after_trt = self._apply_trt()
#
# class SynDataAttn:
#     def __init__(self, seed, n_gene_group, d_gene, d_z, d_x, max_seq_len=10, trans_type="nn"):
#         """
#
#         :param seed: control the randomness
#         :param ncluster: number of genetic clusters
#         :param d_gene: dimension of genetic information
#         :param d_z: dimension of hidden state
#         :param trans_type: type of transition function, choose from ["nn", "linear", "individual_nn"]
#         """
#
#         self.seed = seed
#         self.rng = np.random.default_rng(self.seed)
#         self.n_gene_group = n_gene_group
#         self.d_gene = d_gene
#         self.d_z = d_z
#         self.d_x = d_x
#         self.trans_type = trans_type
#         self.max_seq_len = max_seq_len
#
#         self.mu_x = self.rng.normal(size=(self.n_gene_group, self.d_z, self.d_x))
#         self.var_x = np.abs(self.rng.normal(size=(self.n_gene_group, self.d_z, self.d_x)) / 10)
#
#         self.mu_gene = None  # ncluster x d_gene
#         self.cov_gene = None  # ncluster x d_gene x d_gene
#         self.pi = None
#         self.P = None
#         self.G = None
#         self.nsample = None
#
#     def _sample_gene(self, nsample):
#         """
#
#         :param nsample: number of sample
#         :return: gene_cluster, a ncluster x n (#patient) x g (dimension of gene) tensor
#         """
#         # set the mean and covariance of genetic information
#         self.mu_gene = self.rng.uniform(-10, 10, (self.n_gene_group, self.d_gene))  # mu_g
#         self.cov_gene = np.ones_like(self.mu_gene)  # sigma_g
#
#         gene_cluster = []
#
#         for c in range(self.n_gene_group):
#             # get the parameter for c-th genetic group
#             mu_gene_cluster = self.mu_gene[c]
#             cov_gene_cluster = self.cov_gene[c]
#
#             # genetic info is sampled from multivariate Gaussian distribution
#             gene_info = self.rng.multivariate_normal(mu_gene_cluster, np.diag(cov_gene_cluster), nsample)
#             gene_cluster.append(gene_info)
#
#         # gene_cluster is a ncluster x n x g tensor
#         return np.array(gene_cluster)
#
#     def _create_init_state_dist(self):
#
#         # transform the genetic info into initial distribution \pi
#
#         # create a linear layer
#         w_init = self.rng.normal(size=(self.d_gene, self.d_z))
#         b_init = self.rng.normal(size=self.d_z)
#
#         # compute \pi by add a linear layer to the mean value of the genetic info
#         pi = np.abs(self.mu_gene @ w_init + b_init)
#         pi = pi / np.expand_dims(np.sum(pi, axis=1), axis=-1)
#
#         return pi
#
#     def _scale_matrix(self, scale=10, shrink_factor=10):
#         """
#         create a scale matrix to make transition matrix have small prob
#         to transit from healthy to the most severe state
#
#         :param scale:
#         :param shrink_factor:
#         :return:
#         """
#         scale_matrix = scale * np.ones((self.d_z, self.d_z))
#         for k in range(len(scale_matrix)):
#             for i in range(len(scale_matrix)):
#                 try:
#                     # shrink off diagonal elements
#                     scale_matrix[i][i + k] /= shrink_factor ** (k)
#                     scale_matrix[i + k][i] /= shrink_factor ** (k)
#                 except:
#                     pass
#         return scale_matrix
#
#     def _create_trans(self, nlayer, scaled=True):
#         if self.G is None:
#             print("Please generate genetic info first, try to use method _sample_gene()")
#             raise ValueError
#         if self.trans_type == "nn":
#             # all the genetic clusters share the same transition function
#             nn = MLP(self.d_gene, self.d_z ** 2, 2 * self.d_z ** 2, nlayer)
#             P = np.abs(nn(torch.from_numpy(self.mu_gene).float()).detach().cpu().numpy())
#             P = P.reshape(-1, self.d_z, self.d_z)
#         elif self.trans_type == "individual_nn":
#             # different genetic cluster have individual transition function
#             n_ggroup = self.mu_gene.shape[0]
#             nn_lst = [MLP(self.d_gene, self.d_z ** 2, 2 * self.d_z ** 2, nlayer) for i in range(n_ggroup)]
#
#             P = [torch.abs(nn_lst[i](torch.from_numpy(self.mu_gene[i]).float())) for i in range(n_ggroup)]
#             P = torch.stack(P).detach().cpu().numpy().reshape(-1, self.d_z, self.d_z)
#
#         elif self.trans_type == "linear":
#             w_trans = np.abs(self.rng.normal(size=(self.d_gene, self.d_z ** 2)))
#             b_trans = np.abs(self.rng.normal(size=(self.d_z ** 2)))
#             P = np.abs(self.mu_gene @ w_trans + b_trans).reshape(-1, self.d_z, self.d_z)
#         else:
#             raise NotImplementedError
#
#         assert P.shape == (self.mu_gene.shape[0], self.d_z, self.d_z)
#
#         if scaled:
#             # whether to scale the transition matrix
#             scale_matrix = self._scale_matrix()
#             P = P * scale_matrix
#
#         P = P / np.expand_dims(np.sum(P, axis=-1), -1)
#         return P
#
#     def _sample_z_init(self, state_prob, zsample_type="cat"):
#         if self.nsample is None:
#             print("Please set the number of sample before using _sample_z")
#             raise ValueError
#
#         if zsample_type == "cat":
#             return self.rng.choice(self.d_z, self.nsample, p=state_prob)
#         else:
#             raise NotImplementedError
#
#     def _sample_z(self, trans_mat, prev_state, zsample_type="cat"):
#         if self.nsample is None:
#             print("Please set the number of sample before using _sample_z")
#             raise ValueError
#
#         if zsample_type == "cat":
#             return np.array([self.rng.choice(self.d_z, 1, p=trans_mat[i]) for i in prev_state]).squeeze(-1)
#         else:
#             raise NotImplementedError
#
#     def _gen_z_seq(self, init_stat_prob, trans_mat, zsample_type="cat"):
#
#         z_seq = []
#         for tt in range(self.max_seq_len):
#             # initial state, using pi
#             if tt == 0:
#                 cur_state = self._sample_z_init(init_stat_prob, zsample_type=zsample_type)
#             else:
#                 prev_state = z_seq[-1]
#                 cur_state = self._sample_z(trans_mat, prev_state, zsample_type=zsample_type)
#             z_seq.append(cur_state)
#         return np.array(z_seq)
#
#     def _sample_x(self, k, cur_zs):
#         """
#
#         :param k: k-th genetic group
#         :return:
#         """
#         x_seq = []
#         for _sample in cur_zs:
#             x_sample = []
#             for tt_z in _sample:
#                 mu_x_tmp = self.mu_x[k][tt_z]
#                 var_x_tmp = self.var_x[k][tt_z]
#
#                 x_tmp = self.rng.multivariate_normal(mu_x_tmp, np.diag(var_x_tmp))
#                 x_sample.append(x_tmp)
#             x_seq.append(np.array(x_sample))
#         return np.array(x_seq)
#
#     def sample(self, nsample, nlayer_trans=2, scaled_trans_matrix=True):
#
#         """
#
#         :param nsample:
#         :param nlayer_trans:
#         :param scaled_trans_matrix:
#         :return:
#         """
#         self.nsample = nsample
#         # sample the prior parameters
#         self.G = self._sample_gene(nsample)
#         self.pi = self._create_init_state_dist()
#         self.P = self._create_trans(nlayer_trans, scaled_trans_matrix)
#
#         # generate z
#         self.z = []
#
#         for k in range(self.n_gene_group):
#             # sample the latent states for each genetic group
#             cur_zs = self._gen_z_seq(self.pi[k], self.P[k], zsample_type="cat").transpose()
#             self.z.append(cur_zs)
#
#         self.z = np.array(self.z)
#
#         # sample the observations
#         self.x = []
#         for k in range(self.n_gene_group):
#             self.x.append(self._sample_x(k, self.z[k]))
#
#         return self.x, self.G, self.z,
#
#     def check_z_distribution_match(self):
#
#         # Check the initial probability
#         init_prob_est = []
#         for k in range(self.n_gene_group):
#             # check init
#             z_init = Counter(self.z[k][:, 0])
#             init_prob_est.append([z_init[i] / np.sum(list(z_init.values())) for i in range(self.d_z)])
#
#         # Check the transition matrix
#         trans_mat_est = np.zeros_like(self.P)
#         for k in range(self.n_gene_group):
#             len_tmp = self.z[k].shape[1] - 1
#             for _interval in range(len_tmp):
#                 tmp = self.z[k][:, _interval:_interval + 2]
#                 for _sample in tmp:
#                     trans_mat_est[k][_sample[0]][_sample[1]] += 1
#         for _trans in trans_mat_est:
#             _trans /= np.expand_dims(np.sum(_trans, axis=1), axis=-1)
#
#         init_prob_est = np.array(init_prob_est)
#         trans_mat_est = np.array(trans_mat_est)
#
#         print("the error between est. and true initial latent state probability is {}".format(
#             np.linalg.norm(self.pi - init_prob_est)))
#         print("the error between est. and true transition matrix is {}".format(
#             np.linalg.norm(self.P - trans_mat_est)))
#
#         for k in range(self.n_gene_group):
#             print("for {}-th genetic group".format(k))
#             print("The true initial latent state probability is")
#             print(self.pi[k].round(2))
#             print("The estimated initial latent state probability is")
#             print(init_prob_est[k].round(2))
#
#             print("The true transition matrix is")
#             print(self.P[k].round(2))
#             print("The estimated transition matrix is")
#             print(trans_mat_est[k].round(2))
#             print("\n")
#
#     def check_x_distribution_match(self):
#
#         # Check the initial probability
#         x_mu_est = []
#         x_var_est = []
#         for k in range(self.n_gene_group):
#             x_mu_est_tmp = []
#             x_var_est_tmp = []
#
#             # check init
#             z_vals = self.z[k].reshape(-1)
#             x_vals = self.x[k].reshape(-1, self.x[k].shape[-1])
#
#             for i in range(self.d_z):
#                 x_sel = x_vals[z_vals == i]
#                 mu_est_i = np.mean(x_sel, axis=0)
#                 var_est_i = np.var(x_sel, axis=0)
#
#                 x_mu_est_tmp.append(mu_est_i)
#                 x_var_est_tmp.append(var_est_i)
#             x_mu_est.append(np.array(x_mu_est_tmp))
#             x_var_est.append(np.array(x_var_est_tmp))
#
#         x_mu_est = np.array(x_mu_est)
#         x_var_est = np.array(x_var_est)
#
#         print("the error between est. and true mean value of x is {}".format(
#             np.linalg.norm(self.mu_x - x_mu_est)))
#         print("the error between est. and true var value of x is {}".format(
#             np.linalg.norm(self.var_x - x_var_est)))
#
#         for k in range(self.n_gene_group):
#             print("for {}-th genetic group".format(k))
#             print("The true mean value of x is")
#             print(self.mu_x[k].round(2))
#             print("The estimated mean value of x is")
#             print(x_mu_est[k].round(2))
#
#             print("for {}-th genetic group".format(k))
#             print("The true var value of x is")
#             print(self.var_x[k].round(2))
#             print("The estimated var value of x is")
#             print(x_var_est[k].round(2))