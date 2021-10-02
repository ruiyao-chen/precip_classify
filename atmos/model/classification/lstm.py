import torch
from torch import nn
import torch.nn.functional as F
from atmos.model.basemodel import BaseModel


class LSTMModel(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_classes, drop_prob, n_highway_layers=2):
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input
        self.n_highway_layers = n_highway_layers
        self.lstm1 = nn.LSTM(n_input, n_hidden, n_layers, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(self.n_highway_layers):
            # x = F.relu(x)
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        out = F.softmax(out)

        return out

    def init_hidden(self, batch_size):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


class Bidir_LSTMModel(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_classes, drop_prob, n_highway_layers=2):
        super(Bidir_LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input
        self.n_highway_layers = n_highway_layers
        self.lstm1 = nn.LSTM(n_input, int(n_hidden / 2), n_layers, bidirectional=True, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, int(n_hidden / 2), n_layers, bidirectional=True, dropout=self.drop_prob)

        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(self.n_highway_layers):
            x, hidden2 = self.lstm2(x, hidden)
            x = F.relu(x)
        x = self.dropout(x)
        out = x[-1]
        # out = out.contiguous().view(-1, 2*self.n_hidden)
        out = self.fc(out)
        out = F.softmax(out)
        return out

    def init_hidden(self, batch_size):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if torch.cuda.is_available():
            hidden = (weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_().cuda(),
                      weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_().cuda())
        else:
            hidden = (weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_(),
                      weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_())

        return hidden


class Res_LSTMModel(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_classes, drop_prob, n_residual_layers=3, n_highway_layers=2):
        super(Res_LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input
        self.n_residual_layers = n_residual_layers
        self.n_highway_layers = n_highway_layers
        self.lstm1 = nn.LSTM(n_input, n_hidden, n_layers, bidirectional=False, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, bidirectional=False, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def addResidualLayers(self, x, hidden):
        for i in range(self.n_residual_layers):
            mid = F.relu(x)
            x, hidden2 = self.lstm2(mid, hidden)
            x = F.relu(x)
            x += mid

        return x

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(self.n_highway_layers):
            x = self.addResidualLayers(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = self.fc(out)
        out = F.softmax(out)

        return out

    def init_hidden(self, batch_size):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


# Source and target input embedding
class Embeddings(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_col):
        super(Embeddings, self).__init__()
        self.cov1D = nn.ConvTranspose1d(input_dim, n_col, output_dim)
        self.batchNorm = nn.BatchNorm1d(n_col)

    def forward(self, x):
        """
        :param x: The input 3D dimension (batch_size, windows_size, nums_channel)
        :return: The output with 3D (batch_size, windows_size, d_model)
        """
        x = self.cov1D(x.unsqueeze(dim=-1))

        return self.batchNorm(x)


class Res_Bidir_LSTMModel(BaseModel):
    def __init__(self, ctx, feature_size, label_size, **kwargs):
        super(Res_Bidir_LSTMModel, self).__init__(ctx, feature_size, label_size)
        step_size = kwargs.get("step_size", 3)
        self.n_layers = self.config.layers
        self.n_hidden = self.config.layer_size
        self.n_classes = label_size[0]
        self.drop_prob = self.config.dropout
        # self.n_input = self.n_hidden // 4
        self.n_input = feature_size

        self.n_residual_layers = self.config.n_residual_layers
        self.n_highway_layers = self.config.n_highway_layers

        self.embedding = Embeddings(feature_size, self.n_input, feature_size)

        self.lstm1 = nn.LSTM(self.n_input, int(self.n_hidden / 2), self.n_layers,
                             bidirectional=True, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(self.n_hidden, int(self.n_hidden / 2), self.n_layers,
                             bidirectional=True, dropout=self.drop_prob)

        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)

    def addResidualLayers(self, x, hidden):

        for i in range(self.n_residual_layers):
            mid = F.relu(x)
            x, hidden2 = self.lstm2(mid, hidden)
            x = F.relu(x)
            x += mid
        return x

    def forward(self, x, **kwargs):
        hidden = kwargs.get("hidden", None)

        if hidden is not None:
            hidden = (hidden[0][:, :x.shape[0], :].contiguous(),
                      hidden[1][:, :x.shape[0], :].contiguous())

        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(self.n_highway_layers):
            x = self.addResidualLayers(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = self.fc(out)
        out = F.softmax(out)

        return out

    def init_hidden(self, batch_size):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if torch.cuda.is_available():
            hidden = (weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_().cuda(),
                      weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_().cuda())
        else:
            hidden = (weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_(),
                      weight.new(2 * self.n_layers, batch_size, int(self.n_hidden / 2)).zero_())

        return hidden


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
