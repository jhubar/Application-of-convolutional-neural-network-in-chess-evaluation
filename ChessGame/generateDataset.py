import torch
import pickle

import chess
import chess.pgn

from tqdm import tqdm

from AIChess import evaluate
from DeepChess import boardToTensor


def loadData():
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

    for game in tqdm(games, desc="Generating states", unit="game"):
        node = game.end()

        while node.parent is not None:
            position = node.board()
            tensor = boardToTensor(position)
            output = evaluate(position)

            X.append(tensor)
            y.append(output)

            node = node.parent

    print("Completed. {} states have been generated\n".format(len(X)))

    return X, y

def save(obj, filePath):
    with open(filePath, "wb") as file:
        pickle.dump(obj, file)

def load(filePath):
    with open(filePath, "rb") as file:
        return pickle.load(file)



if __name__ == "__main__":

    print("############################################################")
    print("#################### GENERATING DATASET ####################")
    print("############################################################\n")

    X, y = loadData()

    save(X, "chessInput")
    save(y, "chessOutput")

    print(len(load("chessOutput")))
