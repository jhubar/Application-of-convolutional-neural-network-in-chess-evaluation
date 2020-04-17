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
from torch.optim import Adam

from evaluator import Evaluator

class CustomNet(Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.cnnModel = Sequential(
            # First layer
            Conv2d(12, 18, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),

            # Second layer
            Conv2d(18, 24, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
        )

        self.fcModel = Sequential(
            Linear(96, 16),
            Linear(16, 1)
        )

    def forward(self, x):
        xconv = self.cnnModel(x)
        xflat = xconv.view(xconv.size(0), -1)
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
        self.train_losses = []
        # empty list to store validation losses
        self.val_losses = []

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

        train_X, val_X, train_y, val_y = train_test_split(
            trainInput, trainOutput, test_size=0.1)

        train_X = torch.stack(train_X)
        train_y = torch.FloatTensor(train_y)

        val_X = torch.stack(val_X)
        val_y = torch.FloatTensor(val_y)

        return train_X, train_y, val_X, val_y

    def train(self, epoch, train_X, train_y, val_X, val_y):
        self.model.train()

        tr_loss = 0

        # getting the training set
        X_train = Variable(train_X)
        y_train = Variable(train_y)

        # getting the validation set
        X_val = Variable(val_X)
        y_val = Variable(val_y)

        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        # prediction for training and validation set
        output_train = self.model(X_train)
        output_val = self.model(X_val)

        # computing the training and validation loss
        loss_train = self.criterion(output_train, y_train)
        loss_val = self.criterion(output_val, y_val)
        self.train_losses.append(loss_train)
        self.val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()
        tr_loss = loss_train.item()
        if epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', epoch+1, '\t', 'loss :', loss_val)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Arguments for the training of the deep evaluator")

    # Fetch arguments
    args = parser.parse_args()

    evaluator = DeepEvaluator()

    train_X, train_y, val_X, val_y = evaluator.loadDataset()

    for epoch in range(evaluator.n_epochs):
        evaluator.train(epoch, train_X, train_y, val_X, val_y)
