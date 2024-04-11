import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import os
from seq2seq import Seq2Seq
import torch.nn as nn
import logging
from time import time
from utils import calc_MI


def get_transitions(state_array, num_states):
    trans_matrix = np.zeros((num_states, num_states))
    each_state = [np.sum((state_array == i) * 1) for i in range(num_states)]

    for k in range(num_states):
        # get index for state k
        where_states = np.where(state_array == k)[0]
        where_states_ = where_states[where_states < len(state_array) - 1]

        # get next states given current state == k
        after_states = np.array([state_array[where_states_[i] + 1] for i in range(len(where_states_))])

        # added to the trans_matrix
        trans_matrix[k, :] = np.array([(np.where(after_states == i))[0].shape[0] for i in range(num_states)])

    return trans_matrix, each_state


def padd_data(X, padd_length):
    X_padded = []

    for k in range(len(X)):

        if X[k].shape[0] < padd_length:

            if len(X[k].shape) > 1:
                X_padded.append(np.array(np.vstack((np.array(X[k]),
                                                    np.zeros((padd_length - X[k].shape[0], X[k].shape[1]))))))
            else:
                X_padded.append(np.array(np.vstack((np.array(X[k]).reshape((-1, 1)),
                                                    np.zeros((padd_length - X[k].shape[0], 1))))))

        else:

            if len(X[k].shape) > 1:
                X_padded.append(np.array(X[k]))
            else:
                X_padded.append(np.array(X[k]).reshape((-1, 1)))

    X_padded = np.array(X_padded)

    return X_padded


def state_to_array(state_index, number_of_states):
    state_array = np.zeros(number_of_states)
    state_array[state_index] = 1

    return state_array


class attentive_state_space_model(nn.Module):
    '''
    Class for the "Attentive state space model" implementation. Based on the paper:
    "Attentive state space model for disease progression" by Ahmed M. Alaa and Mihaela van der Schaar.

    ** Key arguments **

    :param maximum_seq_length: Maximum allowable length for any trajectory in the training data.
    :param input_dim: Dimensionality of the observations (emissions).
    :param num_states: Cardinality of the state space.
    :param inference_network: Configuration of the inference network. Default is: 'Seq2SeqAttention'.
    :param rnn_type: Type of RNN cells to use in the inference network. Default is 'LSTM'.
    :param unsupervised: Boolean for whether the model is supervised or unsupervised. Default is True.
                         Supervised is NOT IMPLEMENTED.
    :param generative: Boolean for whether to enable sampling from the model.
    :param irregular: Whether the trajectories are in continuous time. NOT IMPLEMENTED.
    :param multitask: Boolean for whether multi-task output layers are used in inference network. NOT IMPLEMENTED
    :param num_iterations: Number of iterations for the stochastic variational inference algorithm.
    :param num_epochs: Number of epochs for the stochastic variational inference algorithm.
    :param batch_size: Size of the batch subsampled from the training data.
    :param learning_rate: Learning rate for the ADAM optimizer. (TO DO: enable selection of the optimizer)
    :param num_rnn_hidden: Size of the RNN layers used in the inference network.
    :param num_rnn_layers: Number of RNN layers used in the inference network.
    :param dropout_keep_prob: Dropout probability. Default is None.
    :param num_out_hidden: Size of output layer in inference network.
    :param num_out_layers: Size of output layer in inference network.

    ** Key attributes **

    After fitting the model, the key model parameters are stored in the following attributes:

    :attr states_mean: Mean of each observation in each of the num_states states.
    :attr states_covars: Covariance matrices of observations.
    :attr transition_matrix: Baseline Markov transition matrix for the attentive state space.
    :attr intial probabilities: Initial distribution of states averaged accross all trajectories in training data.

    ** Key methods **

    Three key methods are implemented in the API:

    :method fit: Takes a list of observations and fits an attentive state space model in an unsupervised fashion.
    :method predict: Takes a new observation and returns three variables:
                     - Prediction of the next state at every time step.
                     - Expected observation at the next time tep.
                     - List of attention weights assigned to previous states at every time step.
    :method sample: This method samples synthetic trajectories from the model.

    '''

    def __init__(self,
                 maximum_seq_length,
                 input_dim,
                 random_seed=1,
                 num_states=3,
                 emit_fnc=True,
                 unsupervised=True,
                 irregular=False,
                 multitask=False,
                 num_iterations=50,
                 num_epochs=10,
                 batch_size=100,
                 learning_rate=5 * 1e-4,
                 num_rnn_hidden=100,
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=100,
                 num_out_layers=1,
                 verbosity=True,
                 **kwargs
                 ):
        super().__init__()

        # Set all model variables
        self.random_seed = random_seed
        self.maximum_seq_length = maximum_seq_length
        self.input_dim = input_dim
        self.num_states = num_states
        self.unsupervised = unsupervised
        self.irregular = irregular
        self.multitask = multitask
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_rnn_hidden = num_rnn_hidden
        self.num_rnn_layers = num_rnn_layers
        self.dropout_keep_prob = dropout_keep_prob
        self.num_out_hidden = num_out_hidden
        self.num_out_layers = num_out_layers
        self.verbosity = verbosity
        self.emit_fnc = emit_fnc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s2s = Seq2Seq(self.input_dim, hidden_dim=self.num_rnn_hidden, output_dim=self.num_rnn_hidden)
        # decoder layer to generate seq2seq attention
        self.dec_layer = torch.nn.Linear(self.num_rnn_hidden, 1).to(self.device)
        self.combiner_func = torch.nn.Linear(self.input_dim, self.num_out_hidden).to(self.device)
        self.weight_func = torch.nn.Linear(self.num_out_hidden, self.num_states).to(self.device)

        self.emitter = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.num_states, self.num_rnn_hidden),
            nn.ReLU(),
            nn.Linear(self.num_rnn_hidden, self.num_rnn_hidden),
            nn.ReLU(),
            nn.Linear(self.num_rnn_hidden, self.input_dim),
            nn.Sigmoid()
        )

    def length(self, X):
        a, b = torch.max(torch.abs(X), dim=-1)
        used = torch.sign(a)

        length = torch.sum(used)

        return length

    def forward(self, X):

        self.attentive_inference_network_inputs(X)

        # get each step's value
        self.enc_inp = self.rnn_input_
        # create a lagged version input
        self.dec_inp = torch.cat([torch.zeros_like(self.enc_inp[0]).unsqueeze(0), self.enc_inp[:-1]], dim=0)

        _seq_len = self.enc_inp.shape[1]
        _input_dim = self.enc_inp.shape[-1]

        self.dec_outputs, self.dec_memory = self.s2s(self.enc_inp, self.dec_inp)

        """================ get attention weight ================"""
        # rnn model
        self.seq2seq_attn = self.dec_layer(self.dec_outputs)
        # plus output layer
        self.attention = torch.softmax(self.seq2seq_attn, dim=1)

        #
        _a, _b = torch.max(torch.abs(self.rnn_input_), dim=-1)
        attn_mask = torch.sign(_a).unsqueeze(2)
        masked_attn = torch.multiply(attn_mask, self.attention)

        attn_norms = torch.sum(masked_attn, dim=1).repeat([1, self.maximum_seq_length]).unsqueeze(2)

        self.attention = masked_attn / attn_norms
        self.attention_ = self.attention.repeat([1, 1, self.input_dim])

        """================ get context from input ================"""
        self.context = torch.sum(torch.multiply(self.attention_, self.rnn_input_), dim=1)
        context_layer = self.combiner_func(self.context)

        # forward is the estimated Z
        forward = torch.softmax(self.weight_func(context_layer), dim=-1)
        forward = torch.reshape(forward, [-1, self.maximum_seq_length, self.num_states])

        self.predicted = forward
        return forward

    def attentive_inference_network_inputs(self, X):

        self.num_samples = X.shape[0]

        # repeat the data to the shape maximum_seq_length * num_samples x maximum_seq_length x input_dim
        conv_data = torch.reshape(torch.tile(X, [1, self.maximum_seq_length, 1]),
                                  [self.maximum_seq_length * self.num_samples,
                                   self.maximum_seq_length, self.input_dim])

        conv_mask_ = torch.ones([self.maximum_seq_length, self.maximum_seq_length]).to(self.device)

        # tf.matrix_band_part(conv_mask_, -1, 0) is Lower triangular part of a all ones matrix
        # conv_mask is a tensor num_samples x maximum_seq_length x input_dim
        # where conv_mask[:,:,0] is a vertical stacked Lower triangular matrix

        conv_mask = torch.tile(
            torch.unsqueeze(
                torch.tile(torch.triu(conv_mask_, diagonal=0).T, [self.num_samples, 1]),
                2),
            [1, 1, self.input_dim])

        masked_data = torch.multiply(conv_data, conv_mask)

        self.rnn_input_ = masked_data

    def ELBO(self, X, state_guess):
        # self.observation is input
        # nbatch x t x dim_obs
        # tf.reduce_max(tf.abs(self.observation), reduction_indices=2)
        # select the max element across the dim_obs dimension
        # decide the mask for original length

        # HAOYU: Added mask to compute the correct ELBOs
        padding_mask = torch.where(X == 0, 0, 1)[:,:,0].unsqueeze(-1).bool()
        flat_state_guess = torch.masked_select(state_guess,  padding_mask)

        # forward step - generate Z - inference network
        predicted = self.forward(X)
        flat_forward = torch.masked_select(predicted,  padding_mask)

        # likelihood_loss = self.kl_div(flat_state_guess, torch.ones_like(flat_state_guess), flat_forward, torch.ones_like(flat_forward))
        likelihood_loss = torch.sum(-1 * (flat_state_guess * torch.log(flat_forward)))

        # Average over actual sequence lengths. << Did I forget masking padded ELBOs? >>
        likelihood_loss /= self.length(X)

        return likelihood_loss

    def rec_error(self, batch_train_x, batch_states_z, state_mean, state_cov):
        # batch_train_x is input
        # nbatch x t x dim_obs``
        # state_mean: K (number of states) x D (number of features)
        # state_cov: K (number of states) x D (number of features) x D (number of features)
        # batch_states_z: N (number of samples) x t (time steps) x K (number of states)
        mnormal = torch.distributions.MultivariateNormal(torch.from_numpy(state_mean), torch.from_numpy(state_cov))

        # batch_train_x_flatten: nbatch*t x dim_obs
        batch_train_x_flatten = batch_train_x.reshape((-1, batch_train_x.shape[-1]))
        # rec_candidate: Nt x K (number of states) x D (number of features)
        x_candidate_rec = mnormal.sample([batch_train_x_flatten.shape[0]]).float().to(self.device)
        # batch_states_z_flatten: nbatch*t x 1 x K
        batch_states_z_flatten = batch_states_z.reshape((-1, 1, batch_states_z.shape[-1]))
        # Nt x 1 x K @ Nt x K (number of states) x D (number of features) -> Nt x 1 x D
        x_rec = torch.bmm(batch_states_z_flatten, x_candidate_rec).squeeze(1)

        # [[1,2,3], [0, 0, 0]], [[1,2,3],[2, 3, 4]]
        # select the non-padded part
        padding_mask = torch.where(batch_train_x_flatten == 0, 0, 1)
        rec_error = torch.norm((x_rec - batch_train_x_flatten) * padding_mask, p=2) ** 2

        # N_batch = torch.sum(padding_mask) / batch_train_x_flatten.shape[-1]
        return rec_error / (torch.sum(padding_mask) / batch_train_x_flatten.shape[-1])

    def initialize_hidden_states(self, X):

        self.init_states = GaussianMixture(n_components=self.num_states,
                                           covariance_type='full', random_state=self.random_seed)

        self.init_states.fit(np.concatenate(X).reshape((-1, self.input_dim)))

    def sample_posterior_states(self, q_posterior):

        sampled_list = [state_to_array(np.random.choice(self.num_states, 1, p=q_posterior[k, :])[0], self.num_states)
                        for k in range(q_posterior.shape[0])]
        return np.array(sampled_list)

    def get_observations(self, preds):

        pred_obs = []

        for v in range(preds.shape[0]):

            observations = np.zeros(self.input_dim)

            for k in range(self.num_states):
                observations += self.state_means[k] * preds[v, k]

            pred_obs.append(observations)

        return np.array(pred_obs)

    def sample(self, trajectory_length=5):

        initial_state = np.random.choice(self.num_states, 1,
                                         p=self.initial_probabilities)[0]

        State_trajectory = [initial_state]
        first_observation = np.random.multivariate_normal(self.state_means[initial_state],
                                                          self.state_covars[initial_state])

        Obervation_trajectory = [first_observation.reshape((1, -1))]

        for _ in range(trajectory_length):
            next_state_pred = self.predict(Obervation_trajectory)[0][0][0]
            next_state = np.random.choice(self.num_states, 1, p=next_state_pred)[0]

            State_trajectory.append(next_state)

            next_observation = np.random.multivariate_normal(self.state_means[next_state],
                                                             self.state_covars[next_state]).reshape((1, -1))

            Obervation_trajectory[0] = np.vstack((Obervation_trajectory[0], next_observation))

        return State_trajectory, Obervation_trajectory


class train_attn_model:
    '''
    Class for the "Attentive state space model" implementation. Based on the paper:
    "Attentive state space model for disease progression" by Ahmed M. Alaa and Mihaela van der Schaar.

    ** Key arguments **

    :param maximum_seq_length: Maximum allowable length for any trajectory in the training data.
    :param input_dim: Dimensionality of the observations (emissions).
    :param loss_type: the choice for the loss fucntion. ["elbo", "infer_only", "elbo_sampled_z"]
    :param update_z_sample_method:  decide how to sample z from a probability vector, 
                                    e.g. [0.1,0.2,0.7] -> [0, 0, 1]. choose from ["argmax", "categorical"]
    :param update_para_theta: Bool, whether to update "pi", "P" or "mu_Sigma"
    :param theta_update_set:    only work when update_para_theta is True
                                decide which parameter to update, the whole set will be ("pi", "P", "mu_Sigma"),
                                any subset of ("pi", "P", "mu_Sigma") is also a valid input
    :param random_seed: set random seed
    :param num_states: Cardinality of the state space.
    :param num_iterations: Number of iterations for the stochastic variational inference algorithm.
    :param num_epochs: Number of epochs for the stochastic variational inference algorithm.
    :param batch_size: Size of the batch subsampled from the training data.
    :param learning_rate: Learning rate for the ADAM optimizer. (TO DO: enable selection of the optimizer)
    :param num_rnn_hidden: Size of the RNN layers used in the inference network.
    :param num_rnn_layers: Number of RNN layers used in the inference network.
    :param dropout_keep_prob: Dropout probability. Default is None.
    :param num_out_hidden: Size of output layer in inference network.
    :param num_out_layers: Size of output layer in inference network.

    ** Key attributes **

    After fitting the model, the key model parameters are stored in the following attributes:

    :attr states_mean: Mean of each observation in each of the num_states states.
    :attr states_covars: Covariance matrices of observations.
    :attr transition_matrix: Baseline Markov transition matrix for the attentive state space.
    :attr intial_probabilities: Initial distribution of states averaged accross all trajectories in training data.

    ** Key methods **

    Three key methods are implemented in the API:

    :method fit: Takes a list of observations and fits an attentive state space model in an unsupervised fashion.
    :method predict: Takes a new observation and returns three variables:
                     - Prediction of the next state at every time step.
                     - Expected observation at the next time tep.
                     - List of attention weights assigned to previous states at every time step.
    :method sample: This method samples synthetic trajectories from the model.

    '''

    def __init__(self,
                 maximum_seq_length,
                 input_dim,
                 loss_type="elbo",
                 update_z_sample_method = "argmax",
                 update_para_theta=False,
                 theta_update_set = ("pi"),
                 random_seed=1,
                 num_states=3,
                 num_iterations=50,
                 num_epochs=10,
                 batch_size=100,
                 learning_rate=5 * 1e-4,
                 num_rnn_hidden=100,
                 num_rnn_layers=1,
                 dropout_keep_prob=None,
                 num_out_hidden=100,
                 num_out_layers=1,
                 verbosity=True,
                 lambda_inference = 1e-3,
                 lambda_generative=1e-3,
                 **kwargs
                 ):
        # Set all model variables
        # \Sigma
        self.state_covars = None
        # \mu
        self.state_means = None

        self.lambda_inference = lambda_inference
        self.lambda_generative = lambda_generative

        self.initial_probabilities = None
        self.transition_matrix = None

        # initial prior z - categorical encoding - in shape of N x t
        self.all_states = None
        # initial prior z - one-hot encoding - in shape of N x t x k
        # e.g. self.all_states = [1, 0, 2] -> self.state_trajectories_ = [[0,1,0], [1,0,0], [0,0,1]]
        self.state_trajectories_ = []
        self.true_trajectories_ = []

        self.maximum_seq_length = maximum_seq_length
        self.input_dim = input_dim
        self.num_states = num_states
        self.update_para_theta = update_para_theta
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_rnn_hidden = num_rnn_hidden
        self.num_rnn_layers = num_rnn_layers
        self.dropout_keep_prob = dropout_keep_prob
        self.num_out_hidden = num_out_hidden
        self.num_out_layers = num_out_layers
        self.verbosity = verbosity
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_z_sample_method = update_z_sample_method
        self.theta_update_set = theta_update_set

        self.loss_type = loss_type
        self.model = attentive_state_space_model(
            self.maximum_seq_length, self.input_dim,
            random_seed=self.random_seed,
            num_states=self.num_states, num_iterations=self.num_iterations,
            num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
            num_rnn_hidden=self.num_rnn_hidden, num_rnn_layers=self.num_rnn_layers,
            dropout_keep_prob=self.dropout_keep_prob, num_out_hidden=self.num_out_hidden,
            num_out_layers=self.num_out_layers
        )

        self.likelihood_x_lst = []
        self.inference_error_lst = []
        self.rec_loss_lst = []
        self.total_loss_lst = []
        self.z_gt_error_lst = []
        self.state_mean_lst = []
        self.state_var_lst = []

        self.z_init_error = 0

    def sample_trajectory(self, batch_size):
        """
        sample the hidden states based on initial probabilities and transition matrix
        z \sim p_theta(z)
        """

        # sample the initial state from the initial probabilities
        # z_0 \sim p(z_0) - initial probabilities \mu
        init_state = torch.from_numpy(np.array([
            state_to_array(i, self.num_states) for i in
            np.random.choice(self.num_states, batch_size, p=self.initial_probabilities)
        ])).unsqueeze(1)

        # the list for hidden states z at all time steps
        z_lst = [init_state]

        # sample the current state given the previous state
        # z_t \sim p(z_t| z_{t-1}) - transition matrix P
        pre_state = init_state
        for i in range(self.maximum_seq_length - 1):
            # select the current hidden state probability based on previous step
            transition_per_patient = pre_state @ self.transition_matrix
            cur_state = torch.from_numpy(
                np.array([
                    state_to_array(  # convert state indicator (int) to one-hot encoding
                        np.random.choice(  # randomly choose current state based on the transition_per_patient
                            self.num_states, p=transition_per_patient[i][0]
                        )
                        , self.num_states
                    ) for i in range(batch_size)
                ])
            ).unsqueeze(1)
            z_lst.append(cur_state)
            pre_state = cur_state

        z_infer = torch.cat(z_lst, dim=1).to(self.device)
        return z_infer

    def update_theta(self, batch_preds_z, batch_train_x, len_lst, update_z_sample_method = "argmax", theta_update_set = ("pi")):
        # batch_preds_z nbatch x T x H (#hidden states) in probability
        # binarize batch_preds_z for counting e.g., [0.4154354 , 0.08998488, 0.49457964] -> [0., 0., 1.]
        # approach 1 argmax
        if update_z_sample_method == "argmax":
            batch_preds_z_one_hot= np.array([
                np.array([
                    state_to_array(visit, self.num_states) for visit in patient_seq
                ]) for patient_seq in np.argmax(batch_preds_z, axis=-1)
            ])
            batch_preds_z_categorical = np.argmax(batch_preds_z, axis=-1)
        # approach 2 draw from distribution
        elif update_z_sample_method == "categorical":

            batch_preds_z_categorical = np.array([
                np.array([
                    np.random.choice(self.num_states, p=batch_preds_z[v][k]) for k in range(batch_preds_z.shape[1])
                ]) for v in range(batch_preds_z.shape[0])
            ])
            batch_preds_z_one_hot = np.array([
                np.array([
                    state_to_array(visit, self.num_states) for visit in patient_seq
                ]) for patient_seq in batch_preds_z_categorical
            ])
        else:
            raise NotImplementedError

        # only select the non-padded part
        batch_preds_z_categorical_lst = [
            batch_preds_z_categorical[i][:len_lst[i]] for i in range(len(batch_preds_z_categorical))
                                         ]

        if "pi" in theta_update_set:
            initial_point = batch_preds_z_one_hot[:, 0, :]
            self.initial_probabilities = np.sum(initial_point, axis=0) / np.sum(initial_point)
            # self.initial_probabilities = [0.5, 0.3, 0.2]

        if "P" in theta_update_set:
            transits = np.zeros((self.num_states, self.num_states))

            each_state = np.zeros(self.num_states)

            # only count the transition from upper to lower
            for _k in range(self.batch_size):
                new_trans, new_each_state = get_transitions(batch_preds_z_categorical_lst[_k], self.num_states)
                transits += new_trans
                each_state += new_each_state
            # if self.verbosity:
            #     print("each_state count")
            #     print(each_state)
            #     print("transits count")
            #     print(transits)

            for _ in range(self.num_states):
                transits[_, :] = transits[_, :] / np.sum(transits[_, :])
            self.transition_matrix = transits

        """================ add lower tri mask to trans """


        """=================== update centroid (obs model) using new assignment ==================="""
        if "mu_Sigma" in theta_update_set:
            new_state_means = np.zeros_like(self.state_means)
            new_state_covars = np.zeros_like(self.state_covars)

            # aggregate the observation data X according to the new hidden label z
            _state_data = [[] for i in range(self.num_states)]

            for patient_id in range(len(batch_preds_z_categorical_lst)):
                patient_z_seq = batch_preds_z_categorical_lst[patient_id]
                for visit_idx in range(len(patient_z_seq)):
                    _state = patient_z_seq[visit_idx]
                    _state_data[_state].append(batch_train_x[patient_id][visit_idx])

            _state_data = [np.array(i) for i in _state_data]
            for i in range(self.num_states):
                if len(_state_data) == 0:
                    pass
                else:
                    new_state_means[i] = np.mean(_state_data[i], axis = 0)
                    new_state_covars[i] = np.cov(_state_data[i].T)

            self.state_means = new_state_means
            # self.state_covars = new_state_covars

    def fit(self, X, true_states = None):
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/log'):
            os.makedirs(cwd + '/log')
        logging.basicConfig(filename=cwd + '/log/attn_torch.log', level=logging.DEBUG)

        tik = time()
        self.state_trajectories_ = []
        self.true_trajectories_ = []
        # self._Losses = []

        """=============== step 1: Using Mixture Gaussian to predict the initial hidden states ==============="""
        # generate \mu \Sigma
        self.model.initialize_hidden_states(X)
        # generate z_0 (self.all_states)
        state_inferences_init = [np.argmax(self.model.init_states.predict_proba(X[k]), axis=1) for k in range(len(X))]
        self.all_states = state_inferences_init

        """=============== step 2: create (delayed) trajectories for each patient - one-hot encoding ==============="""
        # line 470 is the non-delayed version - haoyu
        # lines 468-469 for delayed
        # for each patient
        for v in range(len(state_inferences_init)):
            # for each time step k
            state_list = [state_to_array(state_inferences_init[v][k], self.num_states) for k in
                          range(len(state_inferences_init[v]))]
            # delayed_traject = np.vstack((np.array(state_list)[1:, :], np.array(state_list)[-1, :]))
            #
            # self.state_trajectories_.append(delayed_traject)
            self.state_trajectories_.append(np.array(state_list))
            if true_states is not None:
                true_state_list = [state_to_array(true_states[v][k], self.num_states) for k in
                                   range(len(true_states[v]))]
                self.true_trajectories_.append(np.array(true_state_list))
        if true_states is not None:
            z_clustering_pred = np.concatenate(self.all_states)
            z_gt_categorical = np.argmax(np.concatenate(self.true_trajectories_), axis=-1)
            self.z_init_error = calc_MI(z_gt_categorical, z_clustering_pred)





        """=============== step 3: stochastic variational inference ==============="""
        self.stochastic_variational_inference(X)

        tok = time()
        logging.info('{:.2f} min'.format((tok - tik) / 60))
        return self.likelihood_x_lst, self.inference_error_lst, self.rec_loss_lst, self.total_loss_lst, self.z_gt_error_lst, self.state_mean_lst, self.state_var_lst, self.z_init_error

    def stochastic_variational_inference(self, X):


        # pad data
        X_, state_update, state_gt = padd_data(X, self.maximum_seq_length), \
                                     padd_data(self.state_trajectories_, self.maximum_seq_length), \
                                     padd_data(self.true_trajectories_, self.maximum_seq_length)

        """====== step 1: create initial probability and transition matrix by counting numbers ========"""
        # Generate initial probabilities \pi, according to the initial guess of z
        initial_states = np.array([self.all_states[k][0] for k in range(len(self.all_states))])
        init_probs = [np.where(initial_states == k)[0].shape[0] / len(initial_states) for k in range(self.num_states)]
        self.initial_probabilities = np.array(init_probs)

        # Generate transition matrix P, according to the initial guess of z
        transits = np.zeros((self.num_states, self.num_states))
        each_state = np.zeros(self.num_states)

        # count the number of each state and the transition state
        for _ in range(len(self.all_states)):
            new_trans, new_each_state = get_transitions(self.all_states[_], self.num_states)

            transits += new_trans
            each_state += new_each_state

        for _ in range(self.num_states):
            if np.sum(transits[_, :]) == 0:
                transits[_, :] = 0
            else:
                transits[_, :] = transits[_, :] / np.sum(transits[_, :])

        self.transition_matrix = np.array(transits)

        # \mu \Sigma from mixture Gaussian
        self.state_means = self.model.init_states.means_
        self.state_covars = self.model.init_states.covariances_

        """====== step 2: optimization ========"""
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):

            for _ in range(self.num_iterations):
                # print("state_means")
                # print(self.state_means.round(1))

                # get batched data
                batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train_x = torch.from_numpy(X_[batch_samples, :, :]).float().to(self.device)
                batch_states_z = torch.from_numpy(state_update[batch_samples, :, :]).float().to(self.device)


                # ============== infer true hidden state (prior) z using \pi and P - Haoyu
                states_z_infer = self.sample_trajectory(batch_train_x.shape[0])

                # Gradient descent
                opt.zero_grad()

                if self.loss_type == "elbo":
                    loss = self.model.ELBO(batch_train_x, batch_states_z)
                    # \hat{z} model: inference model
                    batch_preds_z = self.model(batch_train_x)
                    rec_loss = self.model.rec_error(batch_train_x, batch_preds_z, self.state_means, self.state_covars)
                    L = self.lambda_inference * loss + self.lambda_generative * rec_loss
                    L.backward()
                elif self.loss_type == "infer_only":
                    loss = self.model.ELBO(batch_train_x, batch_states_z)
                    # \hat{z}
                    batch_preds_z = self.model(batch_train_x)
                    rec_loss = self.model.rec_error(batch_train_x, batch_preds_z, self.state_means, self.state_covars)
                    L = loss
                    L.backward()
                elif self.loss_type == "elbo_sampled_z":
                    loss = self.model.ELBO(batch_train_x, states_z_infer)
                    # \hat{z}
                    batch_preds_z = self.model(batch_train_x)
                    rec_loss = self.model.rec_error(batch_train_x, batch_preds_z, self.state_means, self.state_covars)
                    L = self.lambda_inference * loss + self.lambda_generative * rec_loss
                    L.backward()
                else:
                    raise NotImplementedError

                opt.step()

                """===================== update \pi, P, \mu, \Sigma - Exp setting 5, 6 - Haoyu ================"""
                # get the length list for the input sequence
                len_lst = torch.sum(torch.where(batch_train_x[:, :, 0].detach().cpu() == 0, 0, 1), dim=1).numpy()
                if self.update_para_theta:
                    self.update_theta(batch_preds_z.detach().cpu().numpy(), batch_train_x.detach().cpu().numpy(), len_lst,
                                      update_z_sample_method = self.update_z_sample_method,
                                      theta_update_set = self.theta_update_set)


                # Loss of inference model q_\phi when using ground-truth z
                batch_preds_z = self.model(batch_train_x).detach().cpu().numpy()
                batch_preds_z_categorical = np.argmax(batch_preds_z, axis=-1)


                loss_inf = self.model.ELBO(batch_train_x, batch_states_z).detach().cpu().numpy()

                # loss_inf_gt = self.model.ELBO(batch_train_x, batch_states_z_gt).detach().cpu().numpy()
                # loss of generative model
                rec_loss = rec_loss.detach().cpu().numpy()

                # log p(x|z)
                log_likelihood_ = np.array(
                    [self.get_likelihood(
                        batch_train_x[k, :, :].detach().cpu().numpy(), batch_preds_z[k, :, :]
                    )
                        for k in range(batch_train_x.shape[0])]
                )

                log_likelihood_ = np.sum(log_likelihood_) / self.batch_size

                # when there is no ground truth of z, set loss_inf_gt to 0
                if len(state_gt) != 0:
                    batch_states_z_gt = [state_gt[batch_samples, :, :][i, :len_lst[i], :] for i in range(len(batch_samples))]
                    batch_gt_z_categorical = np.argmax(np.concatenate(batch_states_z_gt), axis=-1)

                    batch_preds_z_cat = np.concatenate([batch_preds_z_categorical[i,:len_lst[i]] for i in range(len(batch_samples))])

                    loss_inf_gt = calc_MI(batch_gt_z_categorical, batch_preds_z_cat, bins=10)
                else:
                    loss_inf_gt = 0

                if self.verbosity:
                    print(
                        'Epoch {} \t----- \tBatch {} \t----- \tLog-Likelihood of X {:.2f} \trec loss {:.2f} '
                        '\t-Likelihood of Z {:.2f}  \tMI between Z_hat and Z_gt {:.2f}'.format(
                            epoch, _, log_likelihood_, rec_loss, loss.detach().cpu().numpy(), loss_inf_gt))
                    # print(self.transition_matrix.round(2))
                    # print(self.initial_probabilities)
                logging.info(
                    'Epoch {} \t----- \tBatch {} \t----- \tLog-Likelihood of X {:.2f} \trec loss {:.2f} \t-Likelihood '
                    'of Z {:.2f}  \tMI between Z_hat and Z_gt {:.2f}'.format(
                        epoch, _, log_likelihood_, rec_loss, loss.detach().cpu().numpy(), loss_inf_gt))
                self.likelihood_x_lst.append(log_likelihood_)
                self.inference_error_lst.append(loss_inf)
                self.rec_loss_lst.append(rec_loss)
                self.total_loss_lst.append(L.detach().cpu().numpy())
                self.z_gt_error_lst.append(loss_inf_gt)
                self.state_mean_lst.append(self.state_means )
                self.state_var_lst.append(self.state_covars)
        # Save model
        # cwd = os.getcwd()
        # if not os.path.exists(cwd + '/saved'):
        #     os.makedirs(cwd + '/saved')
        # torch.save(self.model.state_dict(), cwd + "/saved/attnss.pt")
        # return self.likelihood_x_lst, self.inference_error_lst, self.rec_loss_lst, self.total_loss_lst, self.z_gt_error_lst, self.state_mean_lst, self.state_var_lst

    def get_likelihood(self, X, pred):

        # XX = X1.reshape((-1, self.input_dim))
        XX = X.reshape((-1, self.input_dim))
        pred = pred[np.where(np.sum(XX, axis=1) == 0, 0, 1).astype(bool)]
        XX = XX[np.where(np.sum(XX, axis=1) == 0, 0, 1).astype(bool)]

        # lks_ = np.array([multivariate_normal.logpdf(XX, self.state_means[k], self.state_covars[k]).reshape(
        #     (-1, 1)) * pred[:, k].reshape((-1, 1)) for k in range(self.num_states)])

        lks_ = np.array([multivariate_normal.logpdf(XX, self.state_means[k], self.state_covars[k]).reshape(
            (-1, 1)) * pred[:, k].reshape((-1, 1)) for k in range(self.num_states)])

        likelihoods_ = lks_

        # return np.mean(likelihoods_[np.isfinite(likelihoods_)])
        return np.mean(likelihoods_)
        # return np.mean(likelihoods_[np.where(np.sum(XX, axis=1) == 0 ,0 ,1)])

    def sample_posterior_states(self, q_posterior):

        sampled_list = [state_to_array(np.random.choice(self.num_states, 1, p=q_posterior[k, :])[0], self.num_states)
                        for k in range(q_posterior.shape[0])]
        self.state_trajectories = np.array(sampled_list)

    def predict(self, X):

        preds_lengths = [len(X[k]) for k in range(len(X))]

        X_pred = padd_data(X, padd_length=self.maximum_seq_length)
        X_pred = torch.from_numpy(X_pred).float().to(self.device)
        prediction_ = self.model.forward(X_pred).reshape([-1, self.maximum_seq_length, self.num_states])

        prediction_ = prediction_.detach().cpu().numpy()
        preds_ = []
        obs_ = []

        for k in range(len(X)):
            preds_.append(prediction_[k, 0: preds_lengths[k]])
            obs_.append(self.get_observations(preds_[-1]))

        attn_ = self.model.attention.detach().cpu().numpy()
        attn_per_patient = [
            attn_[u * self.maximum_seq_length: u * self.maximum_seq_length + self.maximum_seq_length, :, :] for
            u in range(len(X))]
        attn_lists_per_patient = [[attn_per_patient[u][k, 0: k + 1, :] for k in range(self.maximum_seq_length)]
                                  for u in range(len(X))]

        all_preds_ = (preds_, obs_, attn_lists_per_patient)

        return all_preds_

    def get_observations(self, preds):

        pred_obs = []

        for v in range(preds.shape[0]):

            observations = np.zeros(self.input_dim)

            for k in range(self.num_states):
                observations += self.state_means[k] * preds[v, k]

            pred_obs.append(observations)

        return np.array(pred_obs)

    def sample(self, trajectory_length=5):

        initial_state = np.random.choice(self.num_states, 1,
                                         p=self.initial_probabilities)[0]

        State_trajectory = [initial_state]
        first_observation = np.random.multivariate_normal(self.state_means[initial_state],
                                                          self.state_covars[initial_state])

        Obervation_trajectory = [first_observation.reshape((1, -1))]

        for _ in range(trajectory_length):
            next_state_pred = self.predict(Obervation_trajectory)[0][0][0]
            next_state = np.random.choice(self.num_states, 1, p=next_state_pred)[0]

            State_trajectory.append(next_state)

            next_observation = np.random.multivariate_normal(self.state_means[next_state],
                                                             self.state_covars[next_state]).reshape((1, -1))

            Obervation_trajectory[0] = np.vstack((Obervation_trajectory[0], next_observation))

        return State_trajectory, Obervation_trajectory
