import torch
import pickle

import chess
import chess.pgn

import argparse

from stockfishEvaluator import StockfishEvaluator
from deepEvaluator import DeepEvaluator


def loadData():
    """
    Loads the data from a pgn file
    """
    filePath = "2010_896221.pgn"

    with open(filePath) as pgn:
        nbGames = 0

        pgn.seek(0)

        games = []
        nbStates = 0

        game = chess.pgn.read_game(pgn)

        while game is not None:
            games.append(game)
            nbGames += 1

            nbStates += int(game.headers['PlyCount'])

            game = chess.pgn.read_game(pgn)

    print("{} games found, {} states\n".format(nbGames, nbStates))

    X = []
    y = []

    stockfish = StockfishEvaluator()

    for game in games:
        node = game.end()

        while node.parent is not None:
            position = node.board()
            output = stockfish.evaluate(position)
            if position.turn is chess.BLACK:
                output = -output
                position = position.mirror()
            tensor = DeepEvaluator.boardToTensor(position)

            X.append(tensor)
            y.append(output)

            node = node.parent

    stockfish.quit()

    return X, y

def save(obj, filePath):
    with open(filePath, "wb") as file:
        pickle.dump(obj, file)

def load(filePath):
    with open(filePath, "rb") as file:
        return pickle.load(file)



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Arguments of the Chess Game")

    # Fetch arguments
    args = parser.parse_args()

    X, y = loadData()

    save(X, "DS4200K3500-input") # TODO
    save(y, "DS4200K3500-output")

    print("Completed. {} states have been generated\n".format(len(X)))
