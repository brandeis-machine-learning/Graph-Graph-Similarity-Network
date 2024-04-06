from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from layers import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init



def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=True,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y




class GCN(nn.Module):
    def __init__(self, input_shape, nClass, dropout=0.0):
        super(GCN, self).__init__()
        self.input_shape = input_shape
        self.nClass = nClass
        self.hidden_gcn = 128
        self.embedding_gcn = 256
        self.dropout = dropout
        if self.dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

        self.conv_first = GraphConvolution(self.input_shape, self.hidden_gcn)
        self.conv_second = GraphConvolution(self.hidden_gcn, self.embedding_gcn)
        self.pred_model = nn.Linear(self.embedding_gcn, self.nClass)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj, **kwargs):
        adj = normalize_adj_torch(adj)
        g1 = self.conv_first(x, adj)
        g1 = self.act(g1)
        g1 = self.apply_bn(g1)
        if self.dropout > 0.001:
            g1 = self.dropout_layer(g1)
        g1 = self.conv_second(g1, adj)
        g1 = self.act(g1)
        g1 = self.apply_bn(g1)
        if self.dropout > 0.001:
            g1 = self.dropout_layer(g1)
        pred = self.pred_model(g1)
        return nn.Softmax(dim=1)(pred)



class G2G_model(nn.Module):
    def __init__(self, input_shape, nClass, max_nodes=None, dropout=0.0):
        super(G2G_model, self).__init__()
        self.input_shape = input_shape
        self.nClass = nClass
        self.max_nodes = max_nodes
        self.hidden_gcn = 64
        self.embedding_gcn = 128
        self.hidden_ae = 64
        self.hidden_dg = 32
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_first = GraphConv(input_dim=self.input_shape, output_dim=self.hidden_gcn)
        self.conv_second = GraphConv(input_dim=self.hidden_gcn, output_dim=self.embedding_gcn)
        self.pred_model = nn.Linear(self.embedding_gcn, self.nClass)
        
        self.encoder_model1 = nn.Linear(self.embedding_gcn, self.hidden_ae)
        self.encoder_model2 = nn.Linear(self.hidden_ae, self.hidden_dg)
        self.decoder_model1 = nn.Linear(self.hidden_dg, self.hidden_ae)
        self.decoder_model2 = nn.Linear(self.hidden_ae, self.embedding_gcn)

        self.dist_model1 = nn.Linear(self.embedding_gcn, self.embedding_gcn*2)
        self.dist_model2 = nn.Linear(self.embedding_gcn*2, self.embedding_gcn)

        self.layers_model = [self.conv_first, self.conv_second, self.pred_model, self.encoder_model1, self.encoder_model2, self.decoder_model1, self.decoder_model2, self.dist_model1, self.dist_model2]

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)


    def lockModel(self):
        for layers in self.layers_model:
            for i in layers.parameters():
                i.requires_grad = False
        for i in self.encoder_model1.parameters():
            i.requires_grad = True
        for i in self.encoder_model2.parameters():
            i.requires_grad = True

    def unlockModel(self):
        for layers in self.layers_model:
            for i in layers.parameters():
                i.requires_grad = True
        for i in self.encoder_model1.parameters():
            i.requires_grad = False
        for i in self.encoder_model2.parameters():
            i.requires_grad = False

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)


    def forward(self, x, adj, isAdv=False, **kwargs):
        x = torch.as_tensor(x, dtype=torch.float32).cuda()

        if isAdv == False:
            adj = torch.as_tensor(adj, dtype=torch.float32).cuda()

            g = self.conv_first(x, adj)
            g = nn.ReLU()(g)
            g = self.apply_bn(g)
            if self.dropout > 0.001:
                g = self.dropout_layer(g)
            g = self.conv_second(g, adj)
            g = nn.ReLU()(g)
            g = self.apply_bn(g)

            z = self.encoder_model1(g)
            if self.dropout > 0.001:
                z = self.dropout_layer(z)
            z = nn.ReLU()(z)
            z = self.apply_bn(z)
            z = self.encoder_model2(z)

            decode = self.decoder_model1(z)
            if self.dropout > 0.001:
                decode = self.dropout_layer(decode)
            decode = nn.ReLU()(decode)
            decode = self.apply_bn(decode)
            decode = self.decoder_model2(decode)

            vec, _ = torch.max(decode, dim=1)
            pred = self.pred_model(vec)

            dist = self.dist_model1(vec)
            dist = nn.ReLU()(dist)
            if self.dropout > 0.001:
                dist = self.dropout_layer(dist)
            dist = self.apply_bn(dist)
            dist = self.dist_model2(dist)
            dist = torch.matmul(dist, dist.T)
            return g, z, decode, vec, pred, dist
        else:
            z = self.encoder_model1(x)
            if self.dropout > 0.001:
                z = self.dropout_layer(z)
            z = nn.ReLU()(z)
            z = self.apply_bn(z)
            z = self.encoder_model2(z)
            return z

    def loss(self, pred, y):
        return F.cross_entropy(pred, y.cuda(), reduction='mean')



# Discriminator
class Discriminator(nn.Module):  
    def __init__(self, z_dim, dropout=0.0):
        super(Discriminator, self).__init__()
        
        self.hidden_dg = z_dim
        self.hidden_ae = z_dim * 2
        self.dropout = dropout

        self.adversarial_model1 = nn.Linear(self.hidden_dg, self.hidden_ae)
        self.adversarial_model2 = nn.Linear(self.hidden_ae, 1)

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
        # x = F.dropout(self.adversarial_model1(x), p=self.dropout, training=self.training)

        x, _ = torch.max(x, dim=1)
        x = self.adversarial_model1(x)
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        x = nn.ReLU()(x)
        x = self.adversarial_model2(x)
        x = nn.Sigmoid()(x)
        return x#F.sigmoid(self.adversarial_model2(x))