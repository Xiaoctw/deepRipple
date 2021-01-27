import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math


def random_surfing(adj: np.ndarray, epochs: int, alpha: float) -> np.ndarray:
    """
    :param adj: 邻接矩阵，numpy数组,没有经过处理
    :param epochs: 最大迭代次数
    :param alpha: random surf 过程继续的概率
    :return: numpy数组
    """
    N = adj.shape[0]
    # 在此进行归一化
    P0, P = np.eye(N), np.eye(N)
    mat = np.zeros((N, N))
    for _ in range(epochs):
        P = alpha * P.dot(adj) + (1 - alpha) * P0
        mat = mat + P
    return mat


def PPMI_matrix(mat: np.ndarray) -> np.ndarray:
    """
    :param mat: 上一步构建完成的corjuzhen
    """
    m, n = mat.shape
    assert m == n
    D = np.sum(mat)
    col_sums = np.sum(mat, axis=0)
    row_sums = np.sum(mat, axis=1).reshape(-1, 1)
    dot_mat = row_sums * col_sums
    PPMI = np.log(D * mat / dot_mat)
    PPMI = np.maximum(PPMI, 0)
    PPMI[np.isinf(PPMI)] = 0
    #  PPMI = PPMI / PPMI.sum(1).reshape(-1, 1)
    return PPMI


class Layer(nn.Module):
    """
    堆叠的自编码器中的单一一层
    """

    def __init__(self, input_dim, output_dim, zero_ratio, GPU, activation='leaky_relu'):
        super(Layer, self).__init__()
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            # nn.LeakyReLU(negative_slope=0.2)
        )
        if activation == 'relu':
            self.decoder.add_module(activation, nn.ReLU())
        elif activation == 'leaky_relu':
            self.decoder.add_module(activation, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, zero=True):
        if not self.GPU:
            if zero:
                x = x.cpu().clone()
                rand_mat = torch.rand(x.shape)
                zero_mat = torch.zeros(x.shape)
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        else:
            if zero:
                x = x.clone()
                # 直接在cuda中创建数据，速度会快很多
                rand_mat = torch.rand(x.shape, device='cuda')
                zero_mat = torch.zeros(x.shape, device='cuda')
                # 随机初始化为0
                x = torch.where(rand_mat > self.zero_ratio, x, zero_mat)
        x = self.encoder(x)
        #   x = F.leaky_relu(x, negative_slope=0.2)
        x = self.decoder(x)
        return x

    def emb(self, x):
        """
        获得嵌入向量
        :param x:
        :return:
        """
        return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Encoder, self).__init__()
        self.hidden_dims = hidden_dims
        setattr(self, 'l0', nn.Linear(input_dim, hidden_dims[0], ))
        setattr(self, 'b0', nn.BatchNorm1d(hidden_dims[0]))
        setattr(self, 'drop0', nn.Dropout(p=0.5))
        # setattr(self, 'relu0', nn.ReLU(), )
        setattr(self, 'relu0', nn.LeakyReLU(negative_slope=0.2), )
        for i in range(1, len(hidden_dims)):
            setattr(self, 'l{}'.format(i), nn.Linear(hidden_dims[i - 1], hidden_dims[i], ))
            setattr(self, 'b{}'.format(i), nn.BatchNorm1d(hidden_dims[i]))
            setattr(self, 'drop{}'.format(i), nn.Dropout(0.5))
            # setattr(self, 'relu{}'.format(i), nn.ReLU(), )
            setattr(self, 'relu{}'.format(i), nn.LeakyReLU(negative_slope=0.2), )
        setattr(self, 'l{}'.format(len(hidden_dims)), nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for i in range(len(self.hidden_dims)):
            x = getattr(self, 'l{}'.format(i))(x)
            #      x = getattr(self, 'b{}'.format(i))(x)
            x = getattr(self, 'drop{}'.format(i))(x)
            x = getattr(self, 'relu{}'.format(i))(x)
        return getattr(self, 'l{}'.format(len(self.hidden_dims)))(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        setattr(self, 'l0', nn.Linear(output_dim, hidden_dims[-1]))
        setattr(self, 'b0', nn.BatchNorm1d(hidden_dims[-1]))
        setattr(self, 'drop0', nn.Dropout(p=0.5))
        setattr(self, 'relu0', nn.LeakyReLU(negative_slope=0.2))
        # setattr(self, 'relu0', nn.ReLU())
        for i in range(len(hidden_dims) - 1, 0, -1):
            setattr(self, 'l{}'.format(len(hidden_dims) - i), nn.Linear(hidden_dims[i], hidden_dims[i - 1], ))
            setattr(self, 'b{}'.format(len(hidden_dims) - i), nn.BatchNorm1d(hidden_dims[i - 1]))
            setattr(self, 'drop{}'.format(len(hidden_dims) - i), nn.Dropout(0.5))
            setattr(self, 'relu{}'.format(len(hidden_dims) - i), nn.LeakyReLU(negative_slope=0.2), )
            # setattr(self, 'relu{}'.format(len(hidden_dims) - i), nn.ReLU(), )
        setattr(self, 'l{}'.format(len(hidden_dims)), nn.Linear(hidden_dims[0], input_dim))
        setattr(self, 'relu{}'.format(len(hidden_dims)), nn.ReLU())

    def forward(self, x):
        for i in range(len(self.hidden_dims)):
            x = getattr(self, 'l{}'.format(i))(x)
            x = getattr(self, 'b{}'.format(i))(x)
            x = getattr(self, 'drop{}'.format(i))(x)
            x = getattr(self, 'relu{}'.format(i))(x)
        x = getattr(self, 'l{}'.format(len(self.hidden_dims)))(x)
        x = getattr(self, 'relu{}'.format(len(self.hidden_dims)))(x)
        return x


class DeepRipple(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, GPU=False):
        super(DeepRipple, self).__init__()
        assert len(hidden_dims) >= 1
        self.num_layers = len(hidden_dims) + 1
        # self.zero_ratio = zero_ratio
        self.GPU = GPU
        self.encoder1 = Encoder(input_dim, hidden_dims, output_dim)
        self.encoder2 = Encoder(input_dim, hidden_dims, output_dim)
        self.decoder1 = Decoder(input_dim, hidden_dims, output_dim)
        self.decoder2 = Decoder(input_dim, hidden_dims, output_dim)
        self.W1 = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(output_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
        self.b2 = nn.Parameter(torch.FloatTensor(output_dim))
        self.Q = nn.Parameter(torch.FloatTensor(output_dim, 1))
        stdv = 1. / math.sqrt(output_dim)
        nn.init.xavier_uniform_(self.W1, gain=1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.uniform_(self.b1, -stdv, stdv)
        nn.init.uniform_(self.b2, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)

    def forward(self, x1, x2):
        emb = self.emb(x1, x2)
        # 通过解码器
        x1 = self.decoder1(emb)
        x2 = self.decoder2(emb)
        return x1, x2

    def emb(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        w1 = F.leaky_relu((torch.mm(x1, self.W1) + self.b1).mm(self.Q), negative_slope=0.2)
        w2 = F.leaky_relu((torch.mm(x2, self.W2) + self.b2).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w1, w2), dim=1), dim=1)
        # print(x1.shape)
        # print(w.shape)
        emb = w[:, 0].reshape(-1, 1) * x1 + w[:, 1].reshape(-1, 1) * x2
        return emb


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight, gain=1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)


class RippleGCN(nn.Module):
    def __init__(self, num_feat, num_hidden, n_class):
        super(RippleGCN, self).__init__()
        self.gc1 = GraphConvolution(num_feat, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, n_class)
        self.gc3 = GraphConvolution(num_feat, num_hidden)
        self.gc4 = GraphConvolution(num_hidden, n_class)
        self.W1 = nn.Parameter(torch.FloatTensor(n_class, n_class))
        self.b1 = nn.Parameter(torch.FloatTensor(n_class))
        self.W2 = nn.Parameter(torch.FloatTensor(n_class, n_class))
        self.b2 = nn.Parameter(torch.FloatTensor(n_class))
        self.Q = nn.Parameter(torch.FloatTensor(n_class, 1))
        stdv = 1. / math.sqrt(n_class)
        nn.init.xavier_uniform_(self.W1, gain=1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.uniform_(self.b1, -stdv, stdv)
        nn.init.uniform_(self.b2, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)

    def forward(self, x, adj1, adj2):
        emb = self.emb(x, adj1, adj2)
        return torch.log_softmax(emb, dim=1)

    def emb(self, x, adj1, adj2):
        x1 = F.leaky_relu(self.gc1(x, adj1), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x1 = F.dropout(x1, self.dropout, training=self.training)  # x要dropout
        x1 = self.gc2(x1, adj1)
        x2 = F.leaky_relu(self.gc1(x, adj2), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x2 = F.dropout(x2, self.dropout, training=self.training)  # x要dropout
        x2 = self.gc2(x2, adj2)
        w1 = F.leaky_relu((torch.mm(x1, self.W1) + self.b1).mm(self.Q), negative_slope=0.2)
        w2 = F.leaky_relu((torch.mm(x2, self.W2) + self.b2).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w1, w2), dim=1), dim=1)
        emb = w[:, 0] * x1 + w[:, 1] * x2
        return emb


class StackAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, zero_ratio, GPU=False):
        super(StackAutoEncoder, self).__init__()
        assert len(hidden_dims) >= 1
        self.num_layers = len(hidden_dims) + 1
        self.zero_ratio = zero_ratio
        self.GPU = GPU
        setattr(self, 'autoEncoder0', Layer(input_dim, hidden_dims[0], zero_ratio=zero_ratio, GPU=GPU,
                                                       activation='relu'))
        for i in range(1, len(hidden_dims)):
            setattr(self, 'autoEncoder{}'.format(i),
                    Layer(hidden_dims[i - 1], hidden_dims[i], zero_ratio=zero_ratio, GPU=GPU,
                                     activation='relu'))
        setattr(self, 'autoEncoder{}'.format(self.num_layers - 1),
                Layer(hidden_dims[-1], output_dim, zero_ratio=zero_ratio, GPU=GPU,
                                 activation='relu'))
        self.init_weights()

    def emb(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).emb(x)
        return x

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'autoEncoder{}'.format(i)).encoder(x)
        for i in range(self.num_layers - 1, -1, -1):
            x = getattr(self, 'autoEncoder{}'.format(i)).decoder(x)
        return x

    def init_weights(self):
        # 初始化参数十分重要，可以显著降低loss值
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                # mean=0,std = gain * sqrt(2/fan_in + fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1)
            #   nn.init.xavier_uniform_(m.bias,gain=1)
            #    nn.init.uniform_()
            if isinstance(m, nn.BatchNorm1d):
                # nn.init.constant(m.weight, 1)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant(m.bias, 0)
