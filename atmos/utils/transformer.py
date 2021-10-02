import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [ln(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for ln, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # [MultiHeadedAttention] The query torch.Size([10, 8, 34, 16]),
        #                        key torch.Size([10, 8, 34, 16]),
        #                        value torch.Size([10, 8, 34, 16]),
        #                        output torch.Size([10, 34, 128])

        return self.linears[-1](x)


"""
Reference Resource about Simple layers
https://nn.readthedocs.io/en/rtd/simple/index.html
"""


def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    (See citation for details: https://arxiv.org/abs/1607.06450).
    """

    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        # print(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    (See citation for details: https://arxiv.org/abs/1512.03385).
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        # return x + self.dropout(sublayer(self.norm(x)))
        reg = self.norm((x + self.dropout(sublayer(x))))
        return reg


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, n_col, d_model, output_channel):
        super(Generator, self).__init__()
        # y = A*x + B,
        # where x.Size = d_model is input dimension,
        # and y.size = vocab is output dimension
        self.input_channel = n_col * d_model
        self.proj = nn.Linear(self.input_channel, output_channel)

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        # return F.relu(self.proj(x))
        return self.proj(x.reshape((-1, self.input_channel)))


class PositionwiseFeedForward(nn.Module):
    """
    In addition to attention sub-layers, each of the layers in our encoder and decoder contains
    a fully connected feed-forward network, which is applied to each position separately and
    identically. This consists of two linear transformations with a ReLU activation in between.

    While the linear transformations are the same across different positions, they use different
    parameters from layer to layer. Another way of describing this is as two convolutions
    with kernel
    size 1. The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer
    has dimensionality $d_{ff}=2048$.

    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.feed_forward = nn.Sequential(self.w_1, self.dropout, self.relu, self.w_2,
                                          self.dropout)

        # Solution 2: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        # self.w_2(self.dropout(F.relu(self.w_1(x))))
        # self.feed_forward_simply = nn.Sequential(self.w_1, self.relu, self.dropout, self.w_2)

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


# Position of input source/target word embedding
class PositionalEncoding(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       d_model (int): embedding size  d_model
    """

    def __init__(self, d_model, dropout, max_windows_size=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_windows_size, d_model)  # dim: (max_len, d_model)

        # In a sentense, it consists of several words which is indexed from 0.
        # Here max_len means the max number of words can hold by a input sentense.
        # We create refer table [[pe]] with 3D dimension (1, max_len, d_model),
        position = torch.arange(0, max_windows_size).unsqueeze(1)  # dim: (max_len, 1)
        div_term = torch.exp((torch.arange(0, d_model, 2) *  # tensor([ 0,  2,  4, ..., d_model])
                              (-math.log(10000.0) / d_model)).float())  #
        # In index of numpy or tensor, start:end:step  0:d_model:2 = 0:-1:2
        # position.float() * div_term --> dim: (max_len, d_model/2)
        # Replace values in the even position of cols: [0, 2, ..]
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        # Replace values in the odd positions of cols: [1, 3, ..]
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # dim: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 3D deminsion input with a batch of input (batch_size, seq_len, d_model).
                  We can see the input like: there are [[batch_size]] size of sentence,
                  In each sentence, there are [[seq_len]] size of words
                  each word is embedded as 1D [[d_model]] dimension feature vector.
        :return:
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


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


class TransformerEmbeddings(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_col, dropout, max_windows_size=1000):
        super(TransformerEmbeddings, self).__init__()
        self.n_col = n_col
        self.embedding = Embeddings(input_dim, output_dim, n_col)
        self.positional_embedding = PositionalEncoding(output_dim, dropout, max_windows_size)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_embedding_with_pos = self.positional_embedding(x_embedding)
        return x_embedding_with_pos


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, N, d_model):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x: embedded_sequence, (batch_size, seq_len, embed_size)
        :param mask:
        :return: encoded_sequence, (batch_size, seq_len, embed_size)
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: encoded/embedded_sequence, (batch_size, seq_len, d_model)
        :param mask:
        :return: encoded_sequence, (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # x = self.dropout(x)  # Optional
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_col, opt):
        super(TransformerEncoder, self).__init__()
        self.opt = opt
        self.n_col = n_col

        self.src_embed = TransformerEmbeddings(input_dim, opt.d_model, self.n_col, opt.dropout)

        attn = MultiHeadedAttention(opt.head, opt.d_model)
        ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.dropout)
        self.encoder = Encoder(EncoderLayer(opt.d_model, attn, ff, opt.dropout),
                               opt.layers_count,
                               opt.d_model)
        self.generator = Generator(n_col, opt.d_model, output_dim)

    def forward(self, src, src_mask=None):
        """
        :param src: a batch of input sentense with 2D dimension (batch_size, seq_len)
        :param src_mask:
        :return: a batch of input sentense with encoding embedding,
                 3D dimentsion (batch_size, seq_len, d_model)
        """

        # after src_embed, return src_embed:(batch_size, d_model), torch.Size([500, 231])
        src_embed = self.src_embed(src)
        output = self.encoder(src_embed, src_mask)
        output = self.generator(output)
        return output


"""
Attention Layer
"""


class SelfAttention(nn.Module):
    def __init__(self, input_dim, d_model, n_col, h, dropout=0.1):
        """Take in model size and number of heads."""
        super(SelfAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.n_col = n_col  # nums columns of input
        self.h = h  # nums of head
        self.d_model = d_model  # dimension size of input and output model
        self.d_k = d_model // h
        self.conv1D = clones(nn.ConvTranspose1d(input_dim, n_col, d_model), 3)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, _input, mask=None):
        # _input: [batch_size, d_model], torch.Size([500, 743])
        # batch_size, 500
        nbatches = _input.size(0)

        # query: [batch_size, new_d_model], torch.Size([500, 768])
        query = _input.unsqueeze(-1)
        key = _input.unsqueeze(-1)
        value = _input.unsqueeze(-1)

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do individual linear projection on query, key, value
        # in batch from d_model => h x d_k,
        # query = linear_fn(query)
        # key = linear_fn(key)
        # value = linear_fn(value)
        # shape: torch.Size([batch_size, h, n_col, d_k]), torch.Size([500, 2, 24, 16])
        query, key, value = [ln(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for ln, x in zip(self.conv1D, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: [batch_size, h, n_col, d_k], torch.Size([500, 2, 24, 16])
        # self.attention: [batch_size, h, n_col, n_col], torch.Size([500, 2, 24, 24])
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: [batch_size, new_d_model], torch.Size([500, 768])
        output = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # output: [batch_size, d_model], e.g. torch.Size([500, 743])
        # output = self.proj(x)
        return output


class FeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        input_dim (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        output_dim (int): the size of input for the first-layer of the FFN.
    """

    def __init__(self, input_dim, d_ff, output_dim, dropout=0.1, residual=False):
        super(FeedForward, self).__init__()
        self.layer_norm = LayerNorm(input_dim, epsilon=1e-6)
        self.w_1 = nn.Linear(input_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, output_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

        self.residual = True if (input_dim == output_dim) and residual else False

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))

        if self.residual:
            return output + x
        else:
            return output
