import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class SynData:
    def __init__(self, seed = 1, n_g_group = 3, d_g = 5, d_z = 4, d_u = 1, d_x = 5):
        """

        """

        """settings for reproducibility"""
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


        self.nsample = None
        self.Tmax = None



        """ ============ setting for generate genetic group ============"""
        self.n_g_group = n_g_group
        self.d_g = d_g

        # the genetic group info
        self.mu_s = None # (self.n_g_group, self.d_g)
        self.sigma_s = None # (self.n_g_group, self.d_g, self.d_g)
        self.s = None # (self.nsample, self.n_g_group, self.d_g)

        # the genetic group assignment
        self.v = None # (self.n_g_group, self.n_g_group)

        # the genetic group assignment
        self.g = None # (self.nsample, self.d_g)

        """ ============ setting for treatment ============"""
        self.d_u = d_u

        """ ============ setting for observations ============"""
        self.d_x = d_x

        """ ============ setting for generate initial state ============"""
        self.d_z = d_z
        self.mu_z0 = None # (self.n_g_group, self.d_z)
        self.sigma_z0 = None # (self.n_g_group, self.d_z, self.d_z)

        """ ============ setting for generate transitions ============"""
        self.cov_trans = np.eye(self.d_z)

        self.w_trans = self.rng.normal(size=(self.n_g_group, self.d_z + self.d_u, self.d_z))*0.01
        self.b_trans = self.rng.normal(size=(self.n_g_group, self.d_z))

        self.arcoef_trans = 0.0

        """ ============ setting for generate observational data ============"""
        # self.w_obs = self.rng.normal(size=(self.n_g_group, self.d_z, self.d_x))
        self.w_obs = self.rng.normal(size=(self.d_z, self.d_x))
        self.b_obs = self.rng.normal(size=(self.d_x))



    def sampleGaussian(self, mu, cov):
        assert type(cov) is float or type(cov) is np.ndarray, 'invalid type: ' + str(cov) + ' type: ' + str(
            type(cov))
        return mu + np.random.randn(*mu.shape) * np.sqrt(cov)
    def sample_s(self, mu_s_low = -1.0, mu_s_high = 1.0, sigma_s_low = 0.0, sigma_s_high = 1.0):
        """

        :return: self.s [nsample x n_ggroup x d_g]
        """

        s = np.empty((self.n_g_group, self.nsample, self.d_g))
        """ set the mean value and std for genetic info for different group"""
        self.mu_s = self.rng.uniform(low = mu_s_low, high = mu_s_high, size = (self.n_g_group, self.d_g))
        self.sigma_s = np.empty((self.n_g_group, self.d_g, self.d_g))

        for i in range(self.n_g_group):
            self.sigma_s[i] = np.eye(self.d_g) * self.rng.uniform(low = sigma_s_low, high = sigma_s_high)

            s[i] = self.rng.multivariate_normal(self.mu_s[i], self.sigma_s[i], size=self.nsample)
        return s.transpose((1,0,2))

    def sample_v(self, type = "soft"):
        """

        :param type: whether the type is one-hot or softmax style
        :return: V \in [self.n_g_group x self.n_g_group]: the genetic group assignment
        """

        V = self.rng.normal(size=(self.nsample, self.n_g_group))
        if type == "soft":
            V_continuous = np.exp(V) / np.sum(np.exp(V), axis=1, keepdims=True)
            return V_continuous
        elif type == "hard":
            V_discrete = (V == np.max(V, axis=1)[:, None]).astype(int)
            return V_discrete
        else:
            raise NotImplementedError

    def sample_init_state(self):

        # # transform the genetic info into initial distribution \pi

        # # create a linear layer
        # w_init = self.rng.normal(size=(self.d_g, self.d_z))/10
        # b_init = self.rng.normal(size=self.d_z)

        # # compute \pi by add a linear layer to the mean value of the genetic info
        # self.mu_z0 = self.mu_s @ w_init + b_init
        # self.sigma_z0 = np.empty((self.n_g_group, self.d_z, self.d_z))

        # z0 = np.empty((self.n_g_group, self.nsample, self.d_z))
        # for i in range(self.n_g_group):
        #     self.sigma_z0[i] = np.eye(self.d_z)
        #     # z0[i] = self.rng.multivariate_normal(self.mu_z0[i], self.sigma_z0[i], size=self.nsample)
        #     z0[i] = self.rng.multivariate_normal(np.zeros_like(self.mu_z0[i]), self.sigma_z0[i], size=self.nsample)

        # z0 = z0.transpose((1,0,2))
        # z0 = np.einsum("ij,ijk->ik", self.v, z0)
        z0 = self.rng.normal(size=(self.nsample, self.d_z))
        return z0

    def state_trans(self, z_prev, t, cf = False):

        z_next = []
        for i in range(self.nsample):
            # g_idx = np.argmax(self.v[i])
            
            z_new = self.arcoef_trans * z_prev[i] 
            + np.concatenate([z_prev[i], self.u[i][t]]) @ np.einsum("ijk, i -> jk", self.w_trans, self.v[i])  
            + np.einsum("ik, i -> k", self.b_trans, self.v[i])  
            z_next.append(z_new)

        z_next = np.array(z_next)
        return z_next

    def sample_treatment(self):
        u = np.zeros((self.nsample, self.Tmax, self.d_u))
        for k in range(self.d_u):
            t_trt = np.sort(np.random.randint(low=4, high=self.Tmax, size=(self.nsample, 2)), axis=1)
            t_trt_st = t_trt[:, 0 ]
            t_trt_end = t_trt[:, 1]

            for i in range(self.nsample):
                if t_trt_end[i] - 3 >= self.Tmax:
                    t_trt_end[i] -= 3
                u[i, t_trt_st[i]:t_trt_end[i], k] = 1.
                #     return a[...,None]
        return u

    def sample_obs(self):
        x = []
        for t in range(self.Tmax):
            z_cur = self.z_cat[:,t,:]
            # z_cur = self.z[:,t,:]
            # x_cur = []
            # for i in range(self.nsample):
            #     g_idx = np.argmax(self.v[i])
            #     x_cur.append(z_cur[g_idx] @ self.w_obs[g_idx] + self.b_obs)
 
            # mu_xt = np.array(x_cur)
            mu_xt = z_cur @ self.w_obs + self.b_obs
            xt = mu_xt + self.rng.random(mu_xt.shape) 
            x.append(xt)
        x=np.array(x).transpose(1,0,2)
        return x

    def sample(self, nsample = 1000, Tmax = 20, v_type = "hard", arcoef_trans = 0.0):
        """

        :return:
        """

        """ set up genetic information """
        self.arcoef_trans = arcoef_trans
        self.nsample = nsample
        self.Tmax = Tmax

        # sample s \in [nsample x n_ggroup x d_g]
        self.s = self.sample_s(mu_s_low = -5.0, mu_s_high = 5.0, sigma_s_low = 0.0, sigma_s_high = 1.0)
        # sample v \in [nsample x n_ggroup]
        self.v = self.sample_v(type=v_type)
        # check if self.v is one-hot vector
        if v_type == "hard":
            assert np.sum(self.v) == self.nsample
        self.g = np.einsum("ij,ijk->ik", self.v, self.s) 
        self.g = self.g + self.rng.random(self.g.shape)

        self.u = self.sample_treatment()

        """ sample z """
        all_z_cat = []
        all_z = []

        z_prev = self.sample_init_state()
        all_z.append(np.copy(z_prev))
        all_z_cat.append(softmax(np.copy(z_prev)))

        for t in range(self.Tmax-1):
            self.mu_trans = self.state_trans(z_prev, t+1)
            z_prev = self.sampleGaussian(self.mu_trans, 1.0)
            all_z.append(z_prev)
            all_z_cat.append(softmax(z_prev))


        self.z = np.array(all_z).transpose((1,0,2))
        self.z_cat = np.array(all_z_cat).transpose((1, 0, 2))

        """ sample x """
        self.x = self.sample_obs()

class SynDataIndGene:
    def __init__(self, seed = 1, n_g_group = 3, d_g = 5, d_z = 4, d_u = 1, d_x = 5):
        """

        """

        """settings for reproducibility"""
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


        self.nsample = None
        self.Tmax = None



        """ ============ setting for generate genetic group ============"""
        self.n_g_group = n_g_group
        self.d_g = d_g

        # the genetic group info
        self.mu_s = None # (self.n_g_group, self.d_g)
        self.sigma_s = None # (self.n_g_group, self.d_g, self.d_g)
        self.s = None # (self.nsample, self.n_g_group, self.d_g)

        # the genetic group assignment
        self.v = None # (self.n_g_group, self.n_g_group)

        # the genetic group assignment
        self.g = None # (self.nsample, self.d_g)

        """ ============ setting for treatment ============"""
        self.d_u = d_u

        """ ============ setting for observations ============"""
        self.d_x = d_x

        """ ============ setting for generate initial state ============"""
        self.d_z = d_z
        self.mu_z0 = None # (self.n_g_group, self.d_z)
        self.sigma_z0 = None # (self.n_g_group, self.d_z, self.d_z)

        """ ============ setting for generate transitions ============"""
        self.cov_trans = np.eye(self.d_z)

        self.w_trans = self.rng.normal(size=(self.n_g_group, self.d_z + self.d_u, self.d_z))*0.01
        self.b_trans = self.rng.normal(size=(self.n_g_group, self.d_z))

        self.arcoef_trans = 0.0

        """ ============ setting for generate observational data ============"""
        self.w_obs = self.rng.normal(size=(self.d_z, self.d_x))
        self.w_obs2 = self.rng.normal(size=(self.d_x, self.d_x))
        self.b_obs = self.rng.normal(size=(self.d_x))



    def sampleGaussian(self, mu, cov):
        assert type(cov) is float or type(cov) is np.ndarray, 'invalid type: ' + str(cov) + ' type: ' + str(
            type(cov))
        return mu + np.random.randn(*mu.shape) * np.sqrt(cov)
    def sample_s(self, mu_s_low = -5.0, mu_s_high = 5.0, sigma_s_low = 0.0, sigma_s_high = 1.0):
        """

        :return: self.s [nsample x n_ggroup x d_g]
        """

        s = np.empty((self.n_g_group, self.nsample, self.d_g))
        """ set the mean value and std for genetic info for different group"""
        self.mu_s = self.rng.uniform(low = mu_s_low, high = mu_s_high, size = (self.n_g_group, self.d_g))
        self.sigma_s = np.empty((self.n_g_group, self.d_g, self.d_g))

        for i in range(self.n_g_group):
            self.sigma_s[i] = np.eye(self.d_g) * self.rng.uniform(low = sigma_s_low, high = sigma_s_high)

            s[i] = self.rng.multivariate_normal(self.mu_s[i], self.sigma_s[i], size=self.nsample)
        return s.transpose((1,0,2))

    def sample_v(self, type = "soft"):
        """

        :param type: whether the type is one-hot or softmax style
        :return: V \in [self.n_g_group x self.n_g_group]: the genetic group assignment
        """

        V = self.rng.normal(size=(self.nsample, self.n_g_group))
        if type == "soft":
            V_continuous = np.exp(V) / np.sum(np.exp(V), axis=1, keepdims=True)
            return V_continuous
        elif type == "hard":
            V_discrete = (V == np.max(V, axis=1)[:, None]).astype(int)
            return V_discrete
        else:
            raise NotImplementedError

    def sample_init_state(self):

        z0 = self.rng.multivariate_normal(self.rng.random(self.d_z),  np.eye(self.d_z), size=self.nsample)

        return z0

    def state_trans(self, z_prev, t, cf = False):

        # z_next = []
        # for i in range(self.nsample):
        #     g_idx = np.argmax(self.v[i])
        #     if cf:
        #         z_new = self.arcoef_trans * z_prev[i] + np.concatenate([z_prev[i], np.zeros_like(self.u[i][t])])  @ self.w_trans[g_idx] + self.b_trans[g_idx]
        #     else:
        #         z_new = self.arcoef_trans * z_prev[i] + np.concatenate([z_prev[i], self.u[i][t]]) @ self.w_trans[g_idx]  + self.b_trans[g_idx]
        #     z_next.append(z_new)
        #     # z_new = np.array([self.arcoef_trans * z_prev[i] + np.concatenate([z_prev[i], self.u[i][t]]) @ self.w_trans[g_idx]  + self.b_trans[g_idx] for g_idx in range(self.n_g_group)])
        #     # z_next.append(self.v[i]@z_new)

        # z_next = np.array(z_next)
        # return z_next
        z_next2 = self.arcoef_trans * z_prev +  np.einsum("ij,hjk->hik", np.concatenate([z_prev, self.u[:,t,:]], axis = -1), self.w_trans)  + np.expand_dims(self.b_trans, 1).repeat(self.nsample, axis=1)
        z_next2 = z_next2.transpose((1,0,2))
        z_next2 = np.einsum("ijk,ij->ik", z_next2, self.v)
        z_next2 = z_next2 +  np.einsum("ijk,ij->ik", self.rng.random((self.nsample, self.n_g_group, self.d_z)), self.v) 
        return z_next2

    def sample_treatment(self):
        u = np.zeros((self.nsample, self.Tmax, self.d_u))
        for k in range(self.d_u):
            t_trt = np.sort(np.random.randint(low=4, high=self.Tmax, size=(self.nsample, 2)), axis=1)
            t_trt_st = t_trt[:, 0 ]
            t_trt_end = t_trt[:, 1]

            for i in range(self.nsample):
                if t_trt_end[i] - 3 >= self.Tmax:
                    t_trt_end[i] -= 3
                u[i, t_trt_st[i]:t_trt_end[i], k] = 1.
                #     return a[...,None]
        return u

    def sample_obs(self):
        x = []
        for t in range(self.Tmax):
            z_cur = self.z_cat[:,t,:]
            mu_xt = z_cur @ self.w_obs + self.b_obs
            xt = (mu_xt + np.einsum("ijk,ij->ik", self.rng.random((self.nsample, self.n_g_group, self.d_x)), self.v) * 2.0) @  self.w_obs2  + np.einsum("ijk,ij->ik", self.rng.random((self.nsample, self.n_g_group, self.d_x)), self.v) * 2.0
            # xt = (mu_xt + self.rng.random(mu_xt.shape) * 1.0) @  self.w_obs2 + self.rng.random(mu_xt.shape) * 1.0
            x.append(xt)
        x=np.array(x).transpose(1,0,2)
        return x

    def sample(self, nsample = 1000, Tmax = 20, v_type = "hard", arcoef_trans = 0.0):
        """

        :return:
        """

        """ set up genetic information """
        self.arcoef_trans = arcoef_trans
        self.nsample = nsample
        self.Tmax = Tmax

        # sample s \in [nsample x n_ggroup x d_g]
        self.s = self.sample_s(mu_s_low = -5.0, mu_s_high = 5.0, sigma_s_low = 0.0, sigma_s_high = 1.0)
        # sample v \in [nsample x n_ggroup]
        self.v = self.sample_v(type=v_type)
        # check if self.v is one-hot vector
        if v_type == "hard":
            assert np.sum(self.v) == self.nsample
        self.g = np.einsum("ij,ijk->ik", self.v, self.s)
        # self.g = self.g + self.rng.random(self.g.shape)

        self.u = self.sample_treatment()

        """ sample z """
        all_z_cat = []
        all_z = []

        z_prev = self.sample_init_state()
        all_z.append(np.copy(z_prev))
        all_z_cat.append(softmax(np.copy(z_prev)))

        for t in range(self.Tmax-1):
            self.mu_trans = self.state_trans(z_prev, t+1)
            z_prev = self.sampleGaussian(self.mu_trans, 1.0)
            all_z.append(z_prev)
            all_z_cat.append(softmax(z_prev))


        self.z = np.array(all_z).transpose((1,0,2))
        self.z_cat = np.array(all_z_cat).transpose((1, 0, 2))

        """ sample x """
        self.x = self.sample_obs()



class SynDataIndGene_ori:
    def __init__(self, seed = 1, n_g_group = 3, d_g = 5, d_z = 4, d_u = 1, d_x = 5):
        """

        """

        """settings for reproducibility"""
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


        self.nsample = None
        self.Tmax = None



        """ ============ setting for generate genetic group ============"""
        self.n_g_group = n_g_group
        self.d_g = d_g

        # the genetic group info
        self.mu_s = None # (self.n_g_group, self.d_g)
        self.sigma_s = None # (self.n_g_group, self.d_g, self.d_g)
        self.s = None # (self.nsample, self.n_g_group, self.d_g)

        # the genetic group assignment
        self.v = None # (self.n_g_group, self.n_g_group)

        # the genetic group assignment
        self.g = None # (self.nsample, self.d_g)

        """ ============ setting for treatment ============"""
        self.d_u = d_u

        """ ============ setting for observations ============"""
        self.d_x = d_x

        """ ============ setting for generate initial state ============"""
        self.d_z = d_z
        self.mu_z0 = None # (self.n_g_group, self.d_z)
        self.sigma_z0 = None # (self.n_g_group, self.d_z, self.d_z)

        """ ============ setting for generate transitions ============"""
        self.cov_trans = np.eye(self.d_z)

        self.w_trans = self.rng.normal(size=(self.n_g_group, self.d_z + self.d_u, self.d_z))*0.01
        self.b_trans = self.rng.normal(size=(self.n_g_group, self.d_z))

        self.arcoef_trans = 0.0

        """ ============ setting for generate observational data ============"""
        self.w_obs = self.rng.normal(size=(self.d_z, self.d_x))
        self.w_obs2 = self.rng.normal(size=(self.d_x, self.d_x))
        self.b_obs = self.rng.normal(size=(self.d_x))



    def sampleGaussian(self, mu, cov):
        assert type(cov) is float or type(cov) is np.ndarray, 'invalid type: ' + str(cov) + ' type: ' + str(
            type(cov))
        return mu + np.random.randn(*mu.shape) * np.sqrt(cov)
    def sample_s(self, mu_s_low = -5.0, mu_s_high = 5.0, sigma_s_low = 0.0, sigma_s_high = 1.0):
        """

        :return: self.s [nsample x n_ggroup x d_g]
        """

        s = np.empty((self.n_g_group, self.nsample, self.d_g))
        """ set the mean value and std for genetic info for different group"""
        self.mu_s = self.rng.uniform(low = mu_s_low, high = mu_s_high, size = (self.n_g_group, self.d_g))
        self.sigma_s = np.empty((self.n_g_group, self.d_g, self.d_g))

        for i in range(self.n_g_group):
            self.sigma_s[i] = np.eye(self.d_g) * self.rng.uniform(low = sigma_s_low, high = sigma_s_high)

            s[i] = self.rng.multivariate_normal(self.mu_s[i], self.sigma_s[i], size=self.nsample)
        return s.transpose((1,0,2))

    def sample_v(self, type = "soft"):
        """

        :param type: whether the type is one-hot or softmax style
        :return: V \in [self.n_g_group x self.n_g_group]: the genetic group assignment
        """

        V = self.rng.normal(size=(self.nsample, self.n_g_group))
        if type == "soft":
            V_continuous = np.exp(V) / np.sum(np.exp(V), axis=1, keepdims=True)
            return V_continuous
        elif type == "hard":
            V_discrete = (V == np.max(V, axis=1)[:, None]).astype(int)
            return V_discrete
        else:
            raise NotImplementedError

    def sample_init_state(self):

        z0 = self.rng.multivariate_normal(self.rng.random(self.d_z),  np.eye(self.d_z), size=self.nsample)

        return z0

    def state_trans(self, z_prev, t, cf = False):

        z_next = []
        for i in range(self.nsample):
            g_idx = np.argmax(self.v[i])
            if cf:
                z_new = self.arcoef_trans * z_prev[i] + np.concatenate([z_prev[i], np.zeros_like(self.u[i][t])])  @ self.w_trans[g_idx] + self.b_trans[g_idx]
            else:
                z_new = self.arcoef_trans * z_prev[i] + np.concatenate([z_prev[i], self.u[i][t]]) @ self.w_trans[g_idx]  + self.b_trans[g_idx]
            z_next.append(z_new)

        z_next = np.array(z_next)
        return z_next

    def sample_treatment(self):
        u = np.zeros((self.nsample, self.Tmax, self.d_u))
        for k in range(self.d_u):
            t_trt = np.sort(np.random.randint(low=4, high=self.Tmax, size=(self.nsample, 2)), axis=1)
            t_trt_st = t_trt[:, 0 ]
            t_trt_end = t_trt[:, 1]

            for i in range(self.nsample):
                if t_trt_end[i] - 3 >= self.Tmax:
                    t_trt_end[i] -= 3
                u[i, t_trt_st[i]:t_trt_end[i], k] = 1.
                #     return a[...,None]
        return u

    def sample_obs(self):
        x = []
        for t in range(self.Tmax):
            z_cur = self.z_cat[:,t,:]
            mu_xt = z_cur @ self.w_obs + self.b_obs
            xt = (mu_xt + self.rng.random(mu_xt.shape) * 1.0) @  self.w_obs2 + self.rng.random(mu_xt.shape) * 1.0
            x.append(xt)
        x=np.array(x).transpose(1,0,2)
        return x

    def sample(self, nsample = 1000, Tmax = 20, v_type = "hard", arcoef_trans = 0.0):
        """

        :return:
        """

        """ set up genetic information """
        self.arcoef_trans = arcoef_trans
        self.nsample = nsample
        self.Tmax = Tmax

        # sample s \in [nsample x n_ggroup x d_g]
        self.s = self.sample_s(mu_s_low = -5.0, mu_s_high = 5.0, sigma_s_low = 0.0, sigma_s_high = 1.0)
        # sample v \in [nsample x n_ggroup]
        self.v = self.sample_v(type=v_type)
        # check if self.v is one-hot vector
        if v_type == "hard":
            assert np.sum(self.v) == self.nsample
        self.g = np.einsum("ij,ijk->ik", self.v, self.s)

        self.u = self.sample_treatment()

        """ sample z """
        all_z_cat = []
        all_z = []

        z_prev = self.sample_init_state()
        all_z.append(np.copy(z_prev))
        all_z_cat.append(softmax(np.copy(z_prev)))

        for t in range(self.Tmax-1):
            self.mu_trans = self.state_trans(z_prev, t+1)
            z_prev = self.sampleGaussian(self.mu_trans, 1.0)
            all_z.append(z_prev)
            all_z_cat.append(softmax(z_prev))


        self.z = np.array(all_z).transpose((1,0,2))
        self.z_cat = np.array(all_z_cat).transpose((1, 0, 2))

        """ sample x """
        self.x = self.sample_obs()



# Tmax = 50
# self = SynData(seed = 10, d_z=3)
# self.sample(arcoef_trans=0.9, Tmax = Tmax)
#
# import matplotlib.pyplot as plt
# for s_number in range(5):
#     print(self.v[s_number])
#     plt.figure()
#     plt.plot(self.z[s_number,:,0])
#     plt.plot(self.z[s_number,:,1])
#     plt.plot(self.z[s_number,:,2])
#     # plt.plot(self.z[0,:,3])
#     plt.show()
#
#     plt.figure()
#     plt.plot(self.z_cat[s_number,:,0])
#     plt.plot(self.z_cat[s_number,:,1])
#     plt.plot(self.z_cat[s_number,:,2])
#     # plt.plot(self.z[0,:,3])
#     plt.show()
#
#     plt.figure()
#     plt.plot(self.x[s_number,:,0])
#     plt.plot(self.x[s_number,:,1])
#     plt.plot(self.x[s_number,:,2])
#
#     plt.plot(self.x[s_number,:,3])
#     plt.plot(self.x[s_number,:,4])
#     # plt.plot(self.z[0,:,3])
#     plt.show()
#
#
# import matplotlib.pyplot as plt
#
# for z_number in range(self.d_z):
#     g1 = []
#     g2 = []
#     g3 = []
#     for i in range(30):
#
#         if np.argmax(self.v[i]) ==0:
#             g1.append(self.z_cat[i,:,z_number])
#             # plt.plot(self.z[i,:,3], c = "r")
#         elif np.argmax(self.v[i]) ==1:
#             g2.append(self.z_cat[i, :, z_number])
#             # plt.plot(self.z[i,:,3], c = "g")
#         else:
#             g3.append(self.z_cat[i, :, z_number])
#             # plt.plot(self.z[i,:,3], c = "b")
#     g1 = np.array(g1)
#     g2 = np.array(g2)
#     g3 = np.array(g3)
#
#     plt.figure()
#     plt.plot(np.mean(g1,axis=0), c = "r")
#     plt.plot(np.mean(g2,axis=0), c = "g")
#     plt.plot(np.mean(g3,axis=0), c = "b")
#     plt.show()
#
# # plot box for genetic
# # plots for z and x
# # save 10 dataset w/ parameters
