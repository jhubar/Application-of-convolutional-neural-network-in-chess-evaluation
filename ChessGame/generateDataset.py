import torch
import pickle

import chess
import chess.pgn

import argparse
from tqdm import tqdm

from evaluators import StockfishEvaluator
from evaluators.deepEvaluator import DeepEvaluator


def loadData(isWindows: bool):
    """
    Loads the data from a pgn file
    """
    filePath = "ficsgamesdb_201901_CvC_nomovetimes_120511.pgn"

    with open(filePath) as pgn:
        nbGames = len(pgn.readlines()) // 22

        print("{} games found\n".format(nbGames))

        pgn.seek(0)

        games = []
        nbStates = 0

        game = chess.pgn.read_game(pgn)

        for i in tqdm(range(nbGames // 100), desc="Parsing games", unit="game"):
            games.append(game)

            nbStates += int(game.headers['PlyCount'])

            game = chess.pgn.read_game(pgn)

    print("Completed. {} games have been parsed\n".format(len(games)))

    X = []
    y = []

    stockfish = StockfishEvaluator(isWindows)

    for game in tqdm(games, desc="Generating states", unit="game"):
        node = game.end()

        while node.parent is not None:
            position = node.board()
            tensor = DeepEvaluator.boardToTensor(position)
            output = stockfish.evaluate(position)

            X.append(tensor)
            y.append(output)

            node = node.parent

    stockfish.quit()

    print("Completed. {} states have been generated\n".format(len(X)))

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

    print("############################################################")
    print("#################### GENERATING DATASET ####################")
    print("############################################################\n")

    X, y = loadData(isWindows)

    save(X, "chessInput")
    save(y, "chessOutput")
