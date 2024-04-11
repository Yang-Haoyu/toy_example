import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import os
from torch_attn.seq2seq import Seq2Seq
import torch.nn as nn
import logging
from time import time
from baseline_models.utils import masked_gaussian_nll_3d
from baseline_models.base import Model
from sklearn.utils import gen_batches
from scipy.stats import chi2_contingency
def get_transitions(state_array, num_states):

    trans_matrix = np.zeros((num_states, num_states))
    each_state   = [np.sum((state_array==k)*1) for k in range(num_states)]

    for k in range(num_states):

        where_states       = np.where(state_array==k)[0]
        where_states_      = where_states[where_states < len(state_array) - 1]

        after_states       = [state_array[where_states_[k] + 1] for k in range(len(where_states_))]
        trans_matrix[k, :] = np.array([(np.where(np.array(after_states)==k))[0].shape[0] for k in range(num_states)])

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
                 random_seed = 1,
                 num_states=3,
                 unsupervised=True,
                 irregular=False,
                 multitask=False,
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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_rnn_hidden = num_rnn_hidden
        self.num_rnn_layers = num_rnn_layers
        self.dropout_keep_prob = dropout_keep_prob
        self.num_out_hidden = num_out_hidden
        self.num_out_layers = num_out_layers
        self.verbosity = verbosity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s2s = Seq2Seq(self.input_dim, hidden_dim=self.num_rnn_hidden, output_dim=self.num_rnn_hidden)
        # decoder layer to generate seq2seq attention
        self.dec_layer = torch.nn.Linear(self.num_rnn_hidden, 1).to(self.device)
        self.combiner_func = torch.nn.Linear(self.input_dim, self.num_out_hidden).to(self.device)
        self.weight_func =  torch.nn.Linear(self.num_out_hidden, self.num_states).to(self.device)

    def length(self, X):
        a, b = torch.max(torch.abs(X),dim= -1)
        used = torch.sign(a)

        length = torch.sum(used)

        return length

    def forward(self, X):

        self.attentive_inference_network_inputs(X)

        # get each step's value
        self.enc_inp = self.rnn_input_
        # create a lagged version input
        self.dec_inp  = torch.cat([torch.zeros_like(self.enc_inp[0]).unsqueeze(0), self.enc_inp[:-1]], dim=0)

        _seq_len = self.enc_inp.shape[1]
        _input_dim = self.enc_inp.shape[-1]

        self.dec_outputs, self.dec_memory = self.s2s(self.enc_inp, self.dec_inp)

        """================ get attention weight ================"""
        # rnn model
        self.seq2seq_attn = self.dec_layer(self.dec_outputs)
        # plus output layer
        self.attention = torch.softmax(self.seq2seq_attn, dim=1)

        #
        _a, _b = torch.max(torch.abs(self.rnn_input_), dim= -1)
        attn_mask = torch.sign(_a).unsqueeze(2)
        masked_attn = torch.multiply(attn_mask, self.attention)

        attn_norms = torch.sum(masked_attn, dim = 1).repeat([1, self.maximum_seq_length]).unsqueeze(2)

        self.attention = masked_attn / attn_norms
        self.attention_ = self.attention.repeat([1, 1, self.input_dim])

        """================ get context from input ================"""
        self.context = torch.sum(torch.multiply(self.attention_, self.rnn_input_), dim=1)
        context_layer = self.combiner_func(self.context)

        # forward is the estimated Z
        forward = torch.softmax(self.weight_func(context_layer), dim = -1)
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

        flat_state_guess = torch.reshape(state_guess, [-1, self.num_states])

        # forward step - generate Z - inference network
        predicted = self.forward(X)
        flat_forward = torch.reshape(predicted, [-1, self.num_states])

        # likelihood_loss = self.kl_div(flat_state_guess, torch.ones_like(flat_state_guess), flat_forward, torch.ones_like(flat_forward))
        likelihood_loss = torch.sum(-1 * (flat_state_guess * torch.log(flat_forward)))


        # Average over actual sequence lengths. << Did I forget masking padded ELBOs? >>
        likelihood_loss /= self.length(X)

        return likelihood_loss


    def rec_error(self, batch_train, batch_states, state_mean, state_cov):
        # batch_train is input
        # nbatch x t x dim_obs
        # state_mean: K (number of states) x D (number of features)
        # state_cov: K (number of states) x D (number of features) x D (number of features)
        # batch_states: N (number of samples) x t (time steps) x K (number of states)
        mnormal = torch.distributions.MultivariateNormal(torch.from_numpy(state_mean), torch.from_numpy(state_cov))
        # batch_train_flatten: nbatch*t x dim_obs
        batch_train_flatten = batch_train.reshape((-1,batch_train.shape[-1]))
        # rec_candidate: Nt x K (number of states) x D (number of features)
        rec_candidate = mnormal.sample([batch_train_flatten.shape[0]]).float().to(self.device)
        # batch_states_flatten: nbatch*t x 1 x K
        batch_states_flatten = batch_states.reshape((-1, 1, batch_states.shape[-1]))
        # Nt x 1 x K @ Nt x K (number of states) x D (number of features) -> Nt x 1 x D
        x_rec = torch.bmm(batch_states_flatten, rec_candidate).squeeze(1)

        # [[1,2,3].[0, 0, 0]], [[1,2,3].[2, 3, 4]]
        rec_error = torch.norm((x_rec - batch_train_flatten)*torch.where(batch_train_flatten == 0,0,1), p = 2) ** 2
        return rec_error/(torch.sum(torch.where(batch_train_flatten == 0,0,1))/batch_train_flatten.shape[-1])
    def initialize_hidden_states(self, X):

        self.init_states = GaussianMixture(n_components=self.num_states,
                                           covariance_type='full', random_state = self.random_seed)

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
                 loss_type= "elbo",
                 random_seed = 1,
                 num_states=3,
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

        # Set all model variables

        self.maximum_seq_length = maximum_seq_length
        self.input_dim = input_dim
        self.num_states = num_states
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

        self.loss_type = loss_type
        self.model = attentive_state_space_model(
            self.maximum_seq_length, self.input_dim,
            random_seed = self.random_seed,
            num_states = self.num_states,
            num_epochs = self.num_epochs, batch_size = self.batch_size, learning_rate = self.learning_rate,
            num_rnn_hidden = self.num_rnn_hidden, num_rnn_layers = self.num_rnn_layers,
            dropout_keep_prob = self.dropout_keep_prob, num_out_hidden = self.num_out_hidden,
            num_out_layers = self.num_out_layers
        )



    def fit(self, X, x_val):
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/log'):
            os.makedirs(cwd + '/log')
        logging.basicConfig(filename=cwd + '/log/attn_torch.log',  level=logging.DEBUG)


        tik = time()
        self.state_trajectories_ = []
        self._Losses = []

        # =============== step 1: Using Mixture Gaussian to predict the initial hidden states ===============
        self.model.initialize_hidden_states(X)

        state_inferences_init = [np.argmax(self.model.init_states.predict_proba(X[k]), axis=1) for k in range(len(X))]
        self.all_states = state_inferences_init

        # =============== step 2: create delayed trajectories for each patient ===============
        # for each patient
        for v in range(len(state_inferences_init)):
            # for each time step k
            state_list = [state_to_array(state_inferences_init[v][k], self.num_states) for k in
                          range(len(state_inferences_init[v]))]
            # delayed_traject = np.vstack((np.array(state_list)[1:, :], np.array(state_list)[-1, :]))
            #
            # self.state_trajectories_.append(delayed_traject)
            self.state_trajectories_.append(np.array(state_list))

        # =============== step 3: stochastic variational inference ===============
        likelihood_x_lst, inference_error, rec_loss = self.stochastic_variational_inference(X, x_val)

        tok = time()
        logging.info('{:.2f} min'.format((tok-tik)/60))
        return likelihood_x_lst, inference_error, rec_loss

    def stochastic_variational_inference(self, X, x_val):


        """====== step 1: create initial probability and transition matrix by counting numbers ========"""
        # pad data
        X_, state_update = padd_data(X, self.maximum_seq_length), padd_data(self.state_trajectories_,
                                                                            self.maximum_seq_length)
        x_val = padd_data(x_val, self.maximum_seq_length)
        x_val = torch.from_numpy(x_val)
        # Baseline transition matrix
        initial_states = np.array([self.all_states[k][0] for k in range(len(self.all_states))])
        init_probs = [np.where(initial_states == k)[0].shape[0] / len(initial_states) for k in range(self.num_states)]

        transits = np.zeros((self.num_states, self.num_states))
        each_state = np.zeros(self.num_states)

        # count the number of each state and the transition state
        for _ in range(len(self.all_states)):
            new_trans, new_each_state = get_transitions(self.all_states[_], self.num_states)

            transits += new_trans
            each_state += new_each_state

        for _ in range(self.num_states):
            transits[_, :] = transits[_, :] / each_state[_]
            transits[_, :] = transits[_, :] / np.sum(transits[_, :])

        self.initial_probabilities = np.array(init_probs)
        self.transition_matrix = np.array(transits)

        # -----------------------------------------------------------
        # Observational distribution
        # -----------------------------------------------------------

        self.state_means = self.model.init_states.means_
        self.state_covars = self.model.init_states.covariances_

        prev_loss = np.inf
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            likelihood_x_lst = []
            inference_error_lst = []
            rec_loss_lst = []
            batch_idx_gen = gen_batches(len(X), self.batch_size)

            for _, batch_samples in enumerate(batch_idx_gen):

                # get batched data
                # batch_samples = np.random.choice(list(range(X_.shape[0])), size=self.batch_size, replace=False)
                batch_train = torch.from_numpy(X_[batch_samples, :, :]).float().to(self.device)
                batch_states = torch.from_numpy(state_update[batch_samples, :, :]).float().to(self.device)

                # Gradient descent
                opt.zero_grad()
                loss = self.model.ELBO(batch_train, batch_states)
                batch_preds = self.model(batch_train)
                # rec_loss = self.model.rec_error(batch_train, batch_preds, self.state_means, self.state_covars)
                # if self.loss_type == "elbo":
                #     L = loss + rec_loss
                #     L.backward()
                # elif self.loss_type == "infer":
                #     loss.backward()
                # else:
                #     raise NotImplementedError
                loss.backward()

                opt.step()

                batch_preds = self.model(batch_train)
                # sample and update posterior states
                # posterior = self.model.sample_posterior_states(batch_preds.reshape((-1, self.num_states)))
                # rec_loss = rec_loss.detach().cpu().numpy()
                # rec_loss = 0
                log_likelihood_ = np.array(
                    [self.get_likelihood(
                        batch_train[k, :, :].detach().cpu().numpy(), batch_preds[k, :, :].detach().cpu().numpy()
                    )
                        for k in range(batch_train.shape[0])]
                )

                log_likelihood_ = np.sum(log_likelihood_) / self.batch_size

                self._Losses.append(log_likelihood_)
                # nll_ = np.array(
                #     [self.get_nll(
                #         batch_train[k, :, :].detach().cpu().numpy(), batch_preds[k, :, :].detach().cpu().numpy()
                #     )
                #         for k in range(batch_train.shape[0])]
                # )
                # nll = np.sum(nll_)

                if self.verbosity:
                    print(
                        'Epoch {} \t----- \tBatch {} \t----- \t val loss {:.2f} \t train ELBO {:.2f}'.format(
                            epoch, _, prev_loss, loss.detach().cpu().numpy()))
                logging.info(
                        'Epoch {} \t----- \tBatch {} \t----- \t val loss {:.2f} \t train ELBO {:.2f}'.format(
                            epoch, _, prev_loss, loss.detach().cpu().numpy()))
                likelihood_x_lst.append(log_likelihood_)
                inference_error_lst.append(loss.detach().cpu().numpy())
                rec_loss_lst.append(loss.detach().cpu().numpy())
                # break

            state_val_inferences_init = [np.argmax(self.model.init_states.predict_proba(x_val[k]), axis=1) for k in range(len(x_val))]

            # =============== step 2: create delayed trajectories for each patient ===============
            # for each patient
            state_trajectories_value=  []
            for v in range(len(state_val_inferences_init)):
                # for each time step k
                state_list = [state_to_array(state_val_inferences_init[v][k], self.num_states) for k in
                            range(len(state_val_inferences_init[v]))]
                # delayed_traject = np.vstack((np.array(state_list)[1:, :], np.array(state_list)[-1, :]))
                #
                # self.state_trajectories_.append(delayed_traject)
                state_trajectories_value.append(np.array(state_list))
            
            
            loss_val = self.model.ELBO(x_val, torch.FloatTensor(state_trajectories_value).to(self.device))
            loss_val = loss_val 
            loss_val = loss_val.item()

            if loss_val > prev_loss:
                break
            prev_loss = loss_val

        # Save model
        cwd = os.getcwd()
        if not os.path.exists(cwd + '/saved'):
            os.makedirs(cwd + '/saved')
        torch.save(self.model.state_dict(), cwd +  "/saved/attnss.pt")
        return likelihood_x_lst, inference_error_lst, rec_loss_lst

    # k_lst = np.argmax(pred, axis= 1)
    # _means = torch.from_numpy(np.array([self.state_means[k] for k in k_lst]))
    # _covs = torch.from_numpy(np.array([self.state_covars[k, k] for k in k_lst]))
    # masked_nll = masked_gaussian_nll_3d(torch.from_numpy(XX), _means, np.abs(_covs), torch.ones_like(_means))
    # masked_nll         = masked_nll.sum(-1).sum(-1)
    def get_likelihood(self, X, pred):

        # XX = X1.reshape((-1, self.input_dim))
        XX = X.reshape((-1, self.input_dim))
        pred = pred[np.where(np.sum(XX, axis=1) == 0, 0, 1).astype(bool)]
        XX = XX[np.where(np.sum(XX, axis=1) == 0 ,0 ,1).astype(bool)]

        # lks_ = np.array([multivariate_normal.logpdf(XX, self.state_means[k], self.state_covars[k]).reshape(
        #     (-1, 1)) * pred[:, k].reshape((-1, 1)) for k in range(self.num_states)])

        lks_ = np.array([multivariate_normal.logpdf(XX, self.state_means[k], self.state_covars[k]).reshape(
            (-1, 1)) * pred[:, k].reshape((-1, 1)) for k in range(self.num_states)])

        likelihoods_ = lks_

        # return np.mean(likelihoods_[np.isfinite(likelihoods_)])


        return np.mean(likelihoods_)
        # return np.mean(likelihoods_[np.where(np.sum(XX, axis=1) == 0 ,0 ,1)])
    def get_nll(self, X, pred ):
        XX = X.reshape((-1, self.input_dim))
        pred = pred[np.where(np.sum(XX, axis=1) == 0, 0, 1).astype(bool)]
        XX = XX[np.where(np.sum(XX, axis=1) == 0 ,0 ,1).astype(bool)]

        k_lst = np.argmax(pred, axis= 1)
        _means = torch.from_numpy(np.array([self.state_means[k] for k in k_lst]))
        _covs = torch.from_numpy(np.array([self.state_covars[k, k] for k in k_lst]))
        masked_nll = masked_gaussian_nll_3d(torch.from_numpy(XX), _means, np.abs(_covs), torch.ones_like(_means))
        masked_nll         = masked_nll.sum(-1).sum(-1)
        return masked_nll
    def sample_posterior_states(self, q_posterior):

        sampled_list = [state_to_array(np.random.choice(self.num_states, 1, p=q_posterior[k, :])[0], self.num_states)
                        for k in range(q_posterior.shape[0])]
        self.state_trajectories = np.array(sampled_list)

    def predict(self, X, true_states):

        preds_lengths = [len(X[k]) for k in range(len(X))]
        # batch_preds = self.model(X_pred)

        X_pred = padd_data(X, padd_length=self.maximum_seq_length)
        X_pred = torch.from_numpy(X_pred).float().to(self.device)
        true_states_pad = padd_data(true_states, padd_length=self.maximum_seq_length)

        batch_idx_gen = gen_batches(len(X_pred), self.batch_size)
        _pred_lst = []
        for _, batch_samples in enumerate(batch_idx_gen):
            _prediction = self.model.forward(X_pred[batch_samples]).detach().cpu().numpy().reshape([-1, self.maximum_seq_length, self.num_states])
            _pred_lst.append(_prediction)
        _prediction = np.concatenate(_pred_lst)
        assert len(_prediction) == len(X)

        _nll = self.get_nll(X_pred.numpy(), _prediction.reshape((-1, self.num_states)) )

        nstates = len(set(np.concatenate(true_states)))
        result = np.zeros((nstates, nstates))

        for i in range(len(X)):
            for j in range(1,len(X[i])):
                result[int(true_states[i][j]), int(np.argmax(_prediction[i][j]))] += 1
        chi2 = chi2_contingency(result+1)[0]

        result = np.zeros((nstates, nstates))

        for i in range(len(X)):
            for j in range(1,len(X[i])):
                result[int(np.argmax(_prediction[i][j])), np.random.randint(_prediction.shape[-1])] += 1


        return _nll, chi2, result, chi2_contingency(result+1)[0]

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



