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
from torch.nn import Linear, Sequential, ReLU, Conv2d, MaxPool2d, BatchNorm2d, Module, CrossEntropyLoss, MSELoss, ELU, Softmax, Dropout, AvgPool2d
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from evaluator import Evaluator

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

dropout = 0.3
learning_rate = 0.01
nb_epochs = 100
batch  = 128

com = "Relu_4layers_Pooling" # additional commentary or smth
stringName  = com + "dropout_" + str(dropout) + "_lr_" + str(learning_rate) + "_epochs_" +  str(nb_epochs) + "_batch_" + str(batch) + ".png"

print(" with dropout = " + str(dropout) + " and learning_rate = " + str(learning_rate) + " for " + str(nb_epochs) + " epochs " + com )

def init_weights(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class CustomNet(Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.cnnModel = Sequential(                             # side =  (previous side - kernel size + stride + 2*padding)/stride
            # First layer
            Dropout(p=dropout),
            Conv2d(12, 30, kernel_size=2, stride=1, padding=2), #  8-2 + 1 + 4 = 11
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=1 ),                # 11 -2 + 1= 10
            # ELU(),
            # Second layer
            Dropout(p=dropout),
            Conv2d(30, 60, kernel_size=2, stride=1, padding=1), # 10 - 2 + 1 + 2 = 11
            ReLU(inplace=True),
            AvgPool2d(kernel_size=3, stride=2 ),               # (11 -3 +2 )/2= 5





        )
        self.cnnModel.apply(init_weights)

        self.fcModel = Sequential(
            Dropout(p=dropout),
            Linear(5*5*60, 100),
            Dropout(p=dropout),
            Linear(100, 1),
            Softmax(1),
        )
        self.fcModel.apply(init_weights)

    def forward(self, x):
        xconv = self.cnnModel(x)
	    # xflat = xconv.flatten()
        xflat = xconv.view(xconv.size(0), -1)
        res = self.fcModel(xflat)

        return res


class DeepEvaluator(Evaluator):
    def __init__(self):
        self.model = CustomNet().to(device)
        # self.optimizer = Adam(self.model.parameters(), lr=0.07)
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        # self.criterion = CrossEntropyLoss()
        self.criterion = MSELoss()

        # defining the number of epochs
        self.n_epochs = nb_epochs
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
        with open("Data/chessInput", "rb") as file:
            trainInput = pickle.load(file)

        with open("Data/chessOutput", "rb") as file:
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

        # # getting the training set
        # X_train = Variable(train_X)
        # y_train = Variable(train_y)
        X_train = train_X
        y_train = train_y.view(-1, 1)

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
        dataset=train_data, batch_size=batch, shuffle=True)

    train_losses = []

    for epoch in range(evaluator.n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = evaluator.train(epoch, X_batch, y_batch)
            train_losses.append(loss)

        if epoch % 1 == 0:
            print('Epoch : ', epoch+1, '\t', 'loss :', train_losses[-1])

    plt.plot(train_losses)
    plt.savefig("Graph/"+stringName)
