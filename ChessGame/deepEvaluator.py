import argparse
import chess

import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

import pickle

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import Linear, Sequential, ReLU, Conv2d, MaxPool2d, BatchNorm2d, Module, CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from evaluator import Evaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class CustomNet(Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.cnnModel = Sequential(
            # First layer
            Conv2d(12, 120, kernel_size=3, stride=1, padding=0).to(device),
            ReLU(inplace=True).to(device),
            MaxPool2d(kernel_size=2, stride=1).to(device),

            # Second layer
            Conv2d(120, 240, kernel_size=3, stride=1, padding=0).to(device),
            ReLU(inplace=True).to(device),
            MaxPool2d(kernel_size=2, stride=1).to(device),
        )

        self.fcModel = Sequential(
            Linear(960, 160).to(device),
            Linear(160, 16).to(device),
            Linear(16, 1).to(device)
        )

    def forward(self, x):
        xconv = self.cnnModel(x)
        xflat = xconv.flatten()
        # xflat = xconv.view(xconv.size(0), -1)
        res = self.fcModel(xflat)

        return res


class DeepEvaluator(Evaluator):
    def __init__(self):
        self.model = CustomNet()
        self.optimizer = Adam(self.model.parameters(), lr=0.07)
        # self.criterion = CrossEntropyLoss()
        self.criterion = MSELoss()

        # defining the number of epochs
        self.n_epochs = 25
        # empty list to store training losses
        # self.train_losses = []
        # empty list to store validation losses
        # self.val_losses = []

    @staticmethod
    def boardToTensor(board):
        tensor = torch.zeros(12, 8, 8)

        for wp in board.pieces(chess.PAWN, chess.WHITE):
            row = wp // 8
            col = wp % 8
            tensor[0][row][col] = 1
        for wn in board.pieces(chess.KNIGHT, chess.WHITE):
            row = wn // 8
            col = wn % 8
            tensor[1][row][col] = 1
        for wb in board.pieces(chess.BISHOP, chess.WHITE):
            row = wb // 8
            col = wb % 8
            tensor[2][row][col] = 1
        for wr in board.pieces(chess.ROOK, chess.WHITE):
            row = wr // 8
            col = wr % 8
            tensor[3][row][col] = 1
        for wq in board.pieces(chess.QUEEN, chess.WHITE):
            row = wq // 8
            col = wq % 8
            tensor[4][row][col] = 1
        for wk in board.pieces(chess.KING, chess.WHITE):
            row = wk // 8
            col = wk % 8
            tensor[5][row][col] = 1
        for bp in board.pieces(chess.PAWN, chess.BLACK):
            row = bp // 8
            col = bp % 8
            tensor[6][row][col] = 1
        for bn in board.pieces(chess.KNIGHT, chess.BLACK):
            row = bn // 8
            col = bn % 8
            tensor[7][row][col] = 1
        for bb in board.pieces(chess.BISHOP, chess.BLACK):
            row = bb // 8
            col = bb % 8
            tensor[8][row][col] = 1
        for br in board.pieces(chess.ROOK, chess.BLACK):
            row = br // 8
            col = br % 8
            tensor[9][row][col] = 1
        for bq in board.pieces(chess.QUEEN, chess.BLACK):
            row = bq // 8
            col = bq % 8
            tensor[10][row][col] = 1
        for bk in board.pieces(chess.KING, chess.BLACK):
            row = bk // 8
            col = bk % 8
            tensor[11][row][col] = 1

        return tensor

    def evaluate(self, board: chess.Board):
        tensor = DeepEvaluator.boardToTensor(board)
        # TODO
        # model = ?
        # model.load_state_dict(torch.load(MODELPATH))
        # return model.eval(tensor)

        pass

    def loadDataset(self):
        with open("chessInput", "rb") as file:
            trainInput = pickle.load(file)

        with open("chessOutput", "rb") as file:
            trainOutput = pickle.load(file)

        # train_X, val_X, train_y, val_y = train_test_split(
        #     trainInput, trainOutput, test_size=0.1)

        train_X = torch.stack(trainInput)
        train_y = torch.FloatTensor(trainOutput)

        train_X.to(device)
        train_y.to(device)

        train_data = TensorDataset(train_X, train_y)

        return train_data

    def train(self, epoch, train_X, train_y):
        self.model.train()

        # getting the training set
        X_train = Variable(train_X)
        y_train = Variable(train_y)

        # prediction for training and validation set
        output_train = self.model(X_train)

        # computing the training and validation loss
        loss_train = self.criterion(output_train, y_train)

        # computing the updated weights of all the model parameters
        loss_train.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_train.item()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Arguments for the training of the deep evaluator")

    # Fetch arguments
    args = parser.parse_args()

    evaluator = DeepEvaluator()

    train_data = evaluator.loadDataset()

    train_loader = DataLoader(
        dataset=train_data, batch_size=1024, shuffle=True)

    train_losses = []

    for epoch in range(evaluator.n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch.to(device)
            y_batch.to(device)

            loss = evaluator.train(epoch, X_batch, y_batch)
            train_losses.append(loss)

        if epoch % 2 == 0:
            print('Epoch : ', epoch+1, '\t', 'loss :', train_losses[-1])
