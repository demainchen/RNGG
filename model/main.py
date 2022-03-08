import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from layers import deepGCN
from data import Train_Data, Test_Data
from sklearn.metrics import roc_curve
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")
from config import Config

config = Config()
from datetime import datetime
import sklearn.metrics as metrics

from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score


def train(train_data_loader, eval_data_loader, model, epochs, criterion, lr, hyper=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in range(epochs + 1):
        pred = []
        label = []
        l = 0
        for i, (dssp, hmm, pssm, seq_emb, structure_emb, labels) in enumerate(train_data_loader):
            # Every data instance is an input + label pair
            seq_emb = seq_emb.squeeze().to(torch.float32).to(config.device)
            structure_emb = structure_emb.squeeze().to(torch.float32).to(config.device)
            labels = labels.squeeze().unsqueeze(dim=-1).to(torch.float32).to(config.device)
            dssp = dssp.squeeze().to(torch.float32).to(config.device)
            hmm = hmm.squeeze().to(torch.float32).to(config.device)
            pssm = pssm.squeeze().to(torch.float32).to(config.device)
            node_features = torch.cat(( pssm, hmm, dssp), dim=1).to(torch.float).to(config.device)
            # node_features=seq_emb
            adj = structure_emb

            y_pred = model(node_features, adj,seq_emb)
            loss = criterion(y_pred, labels.squeeze(1).to(torch.int64))

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)

            y_pred = y_pred.cpu().detach().numpy()
            pred += [p[1] for p in y_pred]
            label += [float(l) for l in labels]

            optimizer.zero_grad()
            l += loss.item()
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        if epoch % 5 == 0:
            print('train_loss={}'.format(l / len(train_data_loader)))
        eval_rs(pred, label, 'train', epoch, model)
        l = eval(eval_data_loader, model, epoch, hyper, criterion)
        scheduler.step(l)


def eval_rs(pred, label, mode, epoch, model, hyper=None):
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
    rs = 'mode={},epoch={},acc={},precision={},recall={},F1={},MCC={},AUROC={},AUPRC={},time={}'.format(mode, epoch,
                                                                                                        acc1,
                                                                                                        prec1, recall,
                                                                                                        f1,
                                                                                                        mcc, auroc,
                                                                                                        auprc,
                                                                                                        datetime.now().strftime(
                                                                                                            "%Y-%m-%d %H:%M:%S"))

    if f1 > config.best_f1 and mode == 'test':
        torch.save({'state_dict': model.state_dict()},
                   config.save_path + config.save_model_path + str(f1) + str(hyper) + str(epoch) + '.pkl')
        with open(config.save_path + config.best_txt, 'a') as file:
            file.write(rs + '\n')
            if hyper is not None:
                file.write(str(hyper) + '\n')
        print("最好的f1={},此时recall={},precision={},hyper={},time={}".format(f1, recall, prec1, str(hyper),
                                                                          datetime.now().strftime(
                                                                              "%Y-%m-%d %H:%M:%S")))
    if epoch % 5 == 0:
        print(rs)
        with open(config.save_path + config.save_txt, 'a') as file:
            file.write(rs + '\n')
    return auroc, auprc, acc1, recall, prec1, f1, mcc


def eval(eval_data_loader, model, epoch, hyper, criterion):
    pred = []
    label = []
    l = 0
    for i, (dssp, hmm, pssm, seq_emb, structure_emb, labels) in enumerate(eval_data_loader):
        # Every data instance is an input + label pair
        seq_emb = seq_emb.squeeze().to(torch.float32).to(config.device)
        structure_emb = structure_emb.squeeze().to(torch.float32).to(config.device)
        labels = labels.squeeze().unsqueeze(dim=-1).to(torch.float32).to(config.device)
        dssp = dssp.squeeze().to(torch.float32).to(config.device)
        hmm = hmm.squeeze().to(torch.float32).to(config.device)
        pssm = pssm.squeeze().to(torch.float32).to(config.device)
        node_features = torch.cat(( pssm, hmm, dssp), dim=1).to(torch.float).to(config.device)

        adj = structure_emb

        y_pred = model(node_features, adj,seq_emb)
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(y_pred)
        l += criterion(y_pred, labels.squeeze(1).to(torch.int64))
        y_pred = y_pred.cpu().detach().numpy()
        pred += [p[1] for p in y_pred]
        label += [float(l) for l in labels]
    if epoch % 5 == 0:
        print('eval_loss={}'.format(l / len(eval_data_loader)))
    eval_rs(pred, label, 'test', epoch, model, hyper=hyper)
    return l / len(eval_data_loader)


def main():
    train_data = Train_Data(data_path=config.train_data_path)
    test_data = Test_Data(data_path=config.test_data_path)

    train_data_loader = DataLoader(dataset=train_data, batch_size=config.batch_size)
    eval_data_loader = DataLoader(dataset=test_data, batch_size=config.batch_size)

    HIDDEN_DIM = config.HIDDEN_DIM
    DROPOUT = config.DROPOUT
    ALPHA = config.ALPHA
    LAMBDA = config.LAMBDA
    VARIANT = config.VARIANT  # From GCNII

    NUM_CLASSES = config.NUM_CLASSES  # [not bind, bind]
    INPUT_DIM = config.INPUT_DIM
    Heads = config.heads
    Layer = config.LAYER

    # Seed
    SEED = config.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    Layer=[6,10,14,18,22]
    Heads=[1,2,3,4,5]
    LAMBDA=[0.5,1,1.5,2.0]
    Alpha=[0.1,0.3,0.5,0.7,0.9]
    Learning_rate=[0.005,0.001,0.0005]
    Dropout=[0.1,0.2,0.3]

    for layer in Layer:
        for heads in Heads:
            for l in LAMBDA:
                for alpha in Alpha:
                    for lr in Learning_rate:
                        for dr in Dropout:
                            model = deepGCN(layer, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, dr, l, alpha, VARIANT,
                                            heads=heads).to(
                                config.device)

                            train(train_data_loader, eval_data_loader, model=model, epochs=config.epochs,
                                  criterion=config.loss_fun, lr=lr, hyper={'lr':lr, 'Layer': layer,'head':heads,'LAMBDA':l,'ALPHA':alpha,'drop_out':dr})

if __name__ == '__main__':
    main()
