import torch
import torch.nn as nn
from torch.autograd import Variable



class Encoder(nn.Module):
    def __init__(self, input_dim = 1, num_layers = 1, hidden_dim=64):
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn1 = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True
        )

    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device))

        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        # return hidden_n.reshape((self.n_features, self.embedding_dim))
        return x, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 64, num_layers = 1, output_dim=1):
        super(Decoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.num_layers = num_layers
        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, input_hidden, input_cell):

        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, input_dim, hidden_dim = 64,output_dim = 64, output_length=28):
        super(Seq2Seq, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_dim = input_dim, num_layers = 1, hidden_dim = hidden_dim).to(self.device)
        self.output_length = output_length
        self.decoder = Decoder(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = 1, output_dim=output_dim).to(self.device)

    def forward(self, enc_inp, dec_inp):
        encoder_output, hidden, cell = self.encoder(enc_inp)

        # Prepare place holder for decoder output
        dec_outputs = []
        dec_hiddens = []

        # itearate over LSTM - according to the required output days
        for tt in range(dec_inp.shape[1]):
            prev_x, prev_hidden, prev_cell = self.decoder(dec_inp[:,tt,:].unsqueeze(1), hidden, cell)
            hidden, cell = prev_hidden, prev_cell

            dec_outputs.append(prev_x)
            dec_hiddens.append(prev_hidden)

        dec_outputs = torch.cat(dec_outputs, dim = 1)
        dec_hiddens = torch.transpose(torch.stack(dec_hiddens).squeeze(1),0,1)

        return dec_outputs, dec_hiddens
