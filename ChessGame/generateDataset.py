import torch
import pickle

import chess
import chess.pgn

import argparse

from stockfishEvaluator import StockfishEvaluator
from deepEvaluator import DeepEvaluator


def loadData(isWindows: bool):
    """
    Loads the data from a pgn file
    """
    filePath = "./Data/ficsgamesdb_201901_CvC_nomovetimes_120511.pgn"

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

    stockfish = StockfishEvaluator(isWindows)

    for game in games:
        node = game.end()

        while node.parent is not None:
            position = node.board()
            tensor = DeepEvaluator.boardToTensor(position)
            output = stockfish.evaluate(position)

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

    # Windows mode
    parser.add_argument("-w",
                        "--windows",
                        action="store_true",
                        help="Flags for windows user")

    # Fetch arguments
    args = parser.parse_args()

    # Extract depth
    isWindows = args.windows

    X, y = loadData(isWindows)

    save(X, "chessInput")
    save(y, "chessOutput")

    print("Completed. {} states have been generated\n".format(len(X)))
