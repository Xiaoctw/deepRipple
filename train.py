import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import time
import torch.optim as optim


class DataSet(Data.Dataset):

    def __init__(self, mat1, mat2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.num_node = self.mat1.shape[0]

    def __getitem__(self, index):
        return self.mat1[index], self.mat2[index]

    def __len__(self):
        return self.num_node


class OneDataSet(Data.Dataset):

    def __init__(self, mat1):
        self.mat1 = mat1
        self.num_node = self.mat1.shape[0]

    def __getitem__(self, index):
        return self.mat1[index]

    def __len__(self):
        return self.num_node


def train(model: nn.Module, mat1, mat2, b, epochs, lr, batch_size, print_every, GPU=False):
    dataSet = DataSet(mat1, mat2)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(model.parameters(), lr=lr, )  # weight_decay=5e-4)
    criterion = nn.MSELoss()
    for epoch in range(-1, epochs):
        epoch_loss = 0
        for batch_x1, batch_x2 in dataLoader:
            if GPU:
                batch_x1, batch_x2 = batch_x1.cuda(), batch_x2.cuda()
                B = torch.zeros(batch_x1.shape, device='cuda').fill_(b)
                ones_mat = torch.ones(batch_x2.shape, device='cuda')
                B1 = torch.where(batch_x1 > 0, B, ones_mat)
                B2 = torch.where(batch_x2 > 0, B, ones_mat)
            else:
                B = torch.zeros(batch_x1.shape).fill_(b)
                ones_mat = torch.ones(batch_x2.shape)
                B1 = torch.where(batch_x1 > 0, B, ones_mat)
                B2 = torch.where(batch_x2 > 0, B, ones_mat)
            output1, output2 = model(batch_x1, batch_x2)
            batch_loss1 = (B1.mul(torch.pow(batch_x1 - output1, 2))).sum() + (
                B1.mul(torch.abs(batch_x1 - output1))).sum()
            batch_loss2 = (B2.mul(torch.pow(batch_x2 - output2, 2))).sum() + (
                B2.mul(torch.abs(batch_x2 - output2))).sum()
            optimizer.zero_grad()
            loss = batch_loss2 + batch_loss1
            loss.backward()
            # epoch_loss += (batch_loss2.item() + batch_loss1.item()) * batch_x2.shape[0] * batch_x2.shape[1]
            epoch_loss += loss.item()
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            print('Adjust model parameters, epoch:{}, loss: {:.4f}'.format(epoch + 1, epoch_loss / mat1.shape[0]))
    return model


def train_autoEncoder(model: nn.Module, M, b, epochs1,epochs2, lr, batch_size, print_every, GPU=False):
    for param in model.parameters():
        param.requires_grad = False  # 首先冻结所有的层
    num_layers = model.num_layers
    for i in range(num_layers):
        train_layer(model, i, M, lr, epochs1, print_every=print_every,GPU=GPU)
    for param in model.parameters():
        param.requires_grad = True  # 恢复计算各个层的参数梯度
    # 对模型整体进行微调
    dataSet = OneDataSet(M)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(model.parameters(), lr=lr, )  # weight_decay=5e-4)
    criterion = nn.MSELoss()
    for epoch in range(epochs2):
        loss = 0
        for batch_x in dataLoader:
            output = model(batch_x)
            B = torch.zeros(batch_x.shape).fill_(b)
            ones_mat = torch.ones(batch_x.shape)
            B = torch.where(batch_x > 0, B, ones_mat)
            batch_loss = (B.mul(torch.pow(batch_x - output, 2))).sum() + (
                B.mul(torch.abs(batch_x - output))).sum()
            optimizer.zero_grad()
            batch_loss.backward()
            loss += batch_loss.item()
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            print('Adjust model parameters, epoch:{}, loss: {:.4f}'.format(epoch + 1, loss / M.shape[0]))
    return model


def train_layer(model, i, X, lr=3e-5, epochs=200, print_every=20, batch_size=128,GPU=False):
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = True
    for j in range(i):
        # 通过前面各层
        X = getattr(model, 'autoEncoder{}'.format(j)).emb(X)
    dataSet = OneDataSet(X)
    dataLoader = Data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, )
    optimizer = optim.Adam(getattr(model, 'autoEncoder{}'.format(i)).parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    pre_time = time.time()
    for epoch in range(epochs):
        loss = 0
        for batch_x in dataLoader:
            output_x = getattr(model, 'autoEncoder{}'.format(i))(batch_x)
            batch_loss = criterion(batch_x, output_x)
            optimizer.zero_grad()
            batch_loss.backward()  # 反向传播计算参数的梯度
            loss += batch_loss.item() * batch_x.shape[0] * batch_x.shape[1]
            optimizer.step()
        if (epoch + 1) % print_every == 0:
            tem_time = time.time()
            print('Train layer {}, epoch:{}, loss:{:.4f}, time:{:.4f}'.format(i, epoch + 1, loss / X.shape[0],
                                                                              (tem_time - pre_time) / print_every))
            pre_time = tem_time
    for param in getattr(model, 'autoEncoder{}'.format(i)).parameters():
        param.requires_grad = False
