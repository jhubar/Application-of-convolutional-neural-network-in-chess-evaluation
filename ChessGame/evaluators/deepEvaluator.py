import argparse
import chess
import torch

from evaluators.evaluator import Evaluator

MODELPATH = ""

class DeepEvaluator(Evaluator):
    @staticmethod
    def boardToTensor(board):
        tensor = torch.zeros(8, 8, 12)

        for wp in board.pieces(chess.PAWN, chess.WHITE):
            row = wp // 8
            col = wp % 8
            tensor[row][col][0] = 1
        for wn in board.pieces(chess.KNIGHT, chess.WHITE):
            row = wn // 8
            col = wn % 8
            tensor[row][col][1] = 1
        for wb in board.pieces(chess.BISHOP, chess.WHITE):
            row = wb // 8
            col = wb % 8
            tensor[row][col][2] = 1
        for wr in board.pieces(chess.ROOK, chess.WHITE):
            row = wr // 8
            col = wr % 8
            tensor[row][col][3] = 1
        for wq in board.pieces(chess.QUEEN, chess.WHITE):
            row = wq // 8
            col = wq % 8
            tensor[row][col][4] = 1
        for wk in board.pieces(chess.KING, chess.WHITE):
            row = wk // 8
            col = wk % 8
            tensor[row][col][5] = 1
        for bp in board.pieces(chess.PAWN, chess.BLACK):
            row = bp // 8
            col = bp % 8
            tensor[row][col][6] = 1
        for bn in board.pieces(chess.KNIGHT, chess.BLACK):
            row = bn // 8
            col = bn % 8
            tensor[row][col][7] = 1
        for bb in board.pieces(chess.BISHOP, chess.BLACK):
            row = bb // 8
            col = bb % 8
            tensor[row][col][8] = 1
        for br in board.pieces(chess.ROOK, chess.BLACK):
            row = br // 8
            col = br % 8
            tensor[row][col][9] = 1
        for bq in board.pieces(chess.QUEEN, chess.BLACK):
            row = bq // 8
            col = bq % 8
            tensor[row][col][10] = 1
        for bk in board.pieces(chess.KING, chess.BLACK):
            row = bk // 8
            col = bk % 8
            tensor[row][col][11] = 1

        return tensor

    def evaluate(self, board: chess.Board):
        tensor = DeepEvaluator.boardToTensor(board)
        # TODO
        # model = ?
        # model.load_state_dict(torch.load(MODELPATH))
        # return model.eval(tensor)

        pass

    @staticmethod
    def train(dataset):
        # TODO
        pass


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Arguments for the training of the deep evaluator")

    # Fetch arguments
    args = parser.parse_args()

    # TODO
    DeepEvaluator.train(args)
