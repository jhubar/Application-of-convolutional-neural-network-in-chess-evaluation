import torch
import chess

PATH = "./model"

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

def save(model):
    torch.save(model.state_dict(), PATH)

def load(model):
    torch.load_state_dict(torch.load(PATH))

def train(dataset):
    # model = train(dataset)
    pass

def compute(board):
    tensor = boardToTensor(board)

    print(tensor)

    # model = ...
    # load
    # eval
    # return
    pass
