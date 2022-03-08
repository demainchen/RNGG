import numpy

from config import Config

config = Config()
device = config.device

import math
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

class ConvsLayer(nn.Module):
    def __init__(self):
        super(ConvsLayer, self).__init__()
        self.kernels = config.kernels
        self.hidden_channels = 32
        self.in_channel = 1
        self.features_L = config.features_L

        self.W_size = 54

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(self.in_channel, self.hidden_channels,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], 1)))
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(self.kernels[0], 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(self.in_channel, self.hidden_channels,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], 1)))
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(self.kernels[1], 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(self.in_channel, self.hidden_channels,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], 1)))
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(self.kernels[2], 1), stride=1))

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)

        features1 = features1.reshape(features1.size()[0], features1.size()[1],
                                      features1.size()[2] * features1.size()[3])
        features2 = features2.reshape(features2.size()[0], features2.size()[1],
                                      features2.size()[2] * features2.size()[3])
        features3 = features3.reshape(features3.size()[0], features3.size()[1],
                                      features3.size()[2] * features3.size()[3])

        features = torch.cat((features1, features2, features3), 2)

        return features


class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=3, drop_rate=0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.num_neighbor = num_neighbor
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask=None, attention_weight=None, use_top=True):
        # q: bsz, protein_len, hid=heads*hidd'
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)  # q: bsz, heads, protein_len, hid'
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1,
                                                       -2))  # bsz, heads, protein_len, protein_len + bsz, 1, protein_len, protein_len
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = config.drop_layer(outputs)
        outputs = self.ln(outputs)
        return outputs


class NodeModel(nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.bert_lin = nn.Linear(1024, 54)
        self.liner = nn.Linear(86, 32)

        self.attention_list = nn.ModuleList()
        for _ in range(2):
            self.attention_list.append(Self_Attention(hidden_size=54))
        self.conv = ConvsLayer()

        self.out_f_tran1 = nn.Linear(972, 256)
        self.out_f_tran2 = nn.Linear(32 * 256, 256)

    def forward(self, beft_f, pssm_f, hmm_f, dssp_f):
        # f = torch.cat((pssm_f, hmm_f, dssp_f), dim=1).to(device)
        beft_f = self.bert_lin(beft_f)
        f = beft_f
        window_size = config.window_size

        new_f = numpy.zeros((len(f), window_size, f.size()[1]))

        for i in range(len(f)):
            if i - window_size <= 0:
                local = torch.cat((f[i].unsqueeze(0).to(device), torch.zeros(window_size - 1, f.size()[1]).to(device)),
                                  dim=0)
                # print('<0'+str(local.shape))
            elif i + window_size >= len(f):
                local = torch.cat((f[i].unsqueeze(0).to(device), torch.zeros(window_size - 1, f.size()[1]).to(device)),
                                  dim=0)
                # print('<len'+str(local.shape))
            else:
                local = f[i - window_size // 2 - 1:i + window_size // 2].to(device)
            assert local.size()[0] == window_size
            local = local.cpu().detach().numpy()
            new_f[i] = local
        new_f = torch.from_numpy(new_f).to(torch.float32).to(device)
        sf = new_f

        for attention_layer in self.attention_list:
            sf = attention_layer(sf, sf, sf)

        new_f = torch.cat((new_f, sf), dim=2)

        new_f_temp = torch.cat(
            (new_f,
             torch.zeros(config.features_L - new_f.size()[0], new_f.size()[1], new_f.size()[2]).to(device))).unsqueeze(
            0).to(device)
        new_f_temp = new_f_temp.transpose(0, 1).to(device)

        new_f_temp = self.conv(new_f_temp).to(device)
        new_f_temp = new_f_temp[:new_f.size()[0]].to(device)

        new_f_temp = self.out_f_tran1(new_f_temp).to(device)

        new_f_temp = new_f_temp.reshape(new_f_temp.size()[0], new_f_temp.size()[1] * new_f_temp.size()[2])

        new_f_temp = config.leaky_relu(self.out_f_tran2(new_f_temp))

        return new_f_temp


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = min(1, math.log(lamda / l + 1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:  # speed up convergence of the training process
            output = output + input
        return output


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant, path):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.f_str = nn.Linear(nhidden, nhidden)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.weight = pickle.load(open(path, 'rb'))

        self.init_weight()

    def init_weight(self):
        gcn = self.convs

        for i, g in enumerate(gcn):
            w = 'deep_gcn.convs.{}.weight'.format(i)
            g.weight = torch.nn.Parameter(self.weight[w], requires_grad=False)
        fc = self.fcs
        for i, g in enumerate(fc):
            w = 'deep_gcn.fcs.{}.weight'.format(i)
            b = 'deep_gcn.fcs.{}.bias'.format(i)
            g.weight = torch.nn.Parameter(self.weight[w], requires_grad=False)
            g.bias = torch.nn.Parameter(self.weight[b], requires_grad=False)

        # for g in gcn:
        #     print('gcn_' + str(g.weight.requires_grad))
        # for g in fc:
        #     print('fc_' + str(g.weight.requires_grad))

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training).to(config.device)

        layer_inner = self.act_fn(self.fcs[0](x).to(config.device)).to(config.device)
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.f_str(layer_inner)
        return layer_inner


class GN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GN, self).__init__()
        self.deep_gcn = deepGCN(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,
                                dropout=dropout, lamda=lamda, alpha=alpha, variant=variant, path=config.weight_path)
        self.node_f = NodeModel()
        self.attention = Self_Attention(hidden_size=32)
        self.liner_model = nn.ModuleList()

        for _ in range(1):
            self.liner_model.append(nn.Linear(512, 2)),
            self.liner_model.append(config.leaky_relu)

    def forward(self, x, adj, pssm, hmm, dssp):  # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.to(torch.float32).to(config.device)
        node_feature = torch.cat((pssm, hmm, dssp), dim=1).to(torch.float32).to(config.device)
        adj = adj.to(torch.float32).to(config.device)
        stucture = self.deep_gcn(node_feature, adj).to(config.device)
        # node = self.node_f(x, pssm, hmm, dssp).to(config.device)
        # rs = torch.cat((stucture, node), dim=1)
        # for liner in self.liner_model:
        #     rs = liner(rs)
        return stucture


config = Config()
from data import Test_Data
from torch.utils.data import Dataset, DataLoader

test_data = Test_Data(data_path=config.test_data_path)
eval_data_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

import os

m_list = os.listdir('../result/model/')

MAP_CUTOFF = 14
HIDDEN_DIM = 256
LAYER = 8
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True  # From GCNII

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]

INPUT_DIM = 54


# model=GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)


def eval_rs(pred, label, mode, epoch):
    fpr, tpr, _ = roc_curve(label, pred)
    auroc = metrics.roc_auc_score(label, pred)

    auprc = average_precision_score(label, pred)
    for i in range(0, len(pred)):
        if (pred[i] > config.Threashold):
            pred[i] = 1
        else:
            pred[i] = 0
    acc1 = accuracy_score(label, pred, sample_weight=None)
    # spec1 = spec1 + (cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    recall = recall_score(label, pred, sample_weight=None)
    prec1 = precision_score(label, pred, sample_weight=None)
    f1 = f1_score(label, pred)
    mcc = matthews_corrcoef(label, pred)
    rs = 'epoch={},mode={},acc={},precision={},recall={},F1={},MCC={},AUROC={},AUPRC={}'.format(epoch, mode, acc1,
                                                                                                prec1, recall, f1,
                                                                                                mcc, auroc, auprc)
    print(rs + '\n')


def eval(eval_data_loader, model, epoch):
    pred = []
    label = []
    for i, (dssp, hmm, pssm, seq_emb, structure_emb, labels) in enumerate(eval_data_loader):
        # Every data instance is an input + label pair
        seq_emb = seq_emb.squeeze().to(config.device)
        seq_emb = seq_emb.squeeze().to(config.device)
        structure_emb = structure_emb.squeeze().to(config.device)
        labels = labels.squeeze().unsqueeze(dim=-1).to(config.device)
        dssp = dssp.squeeze().to(config.device)
        hmm = hmm.squeeze().to(config.device)
        pssm = pssm.squeeze().to(config.device)

        # node_features = torch.cat((pssm, hmm, dssp), dim=1).to(torch.float)

        structure_emb = structure_emb.to(torch.float)

        # def forward(self, x, adj, pssm, hmm, dssp):  #
        y_pred = model(seq_emb, structure_emb, pssm, hmm, dssp)
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(y_pred)
        y_pred = y_pred.cpu().detach().numpy()
        pred += [p[1] for p in y_pred]
        label += [float(l) for l in labels]
    eval_rs(pred, label, 'test', epoch)


for m in m_list:
    m = '../result/model/' + m
    print(m)

    model = GN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model.load_state_dict(
            torch.load(m)[
                'state_dict'])
    else:
        model.load_state_dict(
            torch.load(m, map_location='cpu')[
                'state_dict'])

    eval(eval_data_loader, model, (m.split('.')[0]))
