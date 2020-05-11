import chess
import chess.engine

from evaluator import Evaluator
from simpleEvaluator import SimpleEvaluator
from deepEvaluatorQuentin import DeepEvaluator

from minimax import searchNextMove

STOCKFISH_PATH = "stockfish"
STOCKFISH_PATH_WINDOWS = "C:\\Users\\diveb\\Downloads\\stockfish-11-win\\stockfish-11-win\\Windows\\stockfish_20011801_x64.exe"

class Game:
    def __init__(self, depth: int, isWindows: bool):
        self.depth = depth

        self.board = chess.Board()

        if isWindows:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WINDOWS)
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        self.answer = True

    def move(self, move: chess.Move):
        self.board.push(move)

    def simpleAIMove(self):
        return searchNextMove(self.board, self.depth, SimpleEvaluator())

    def simpleAIScore(self):
        return SimpleEvaluator().evaluate(self.board)

    def deepAIMove(self):
        return searchNextMove(self.board, self.depth, DeepEvaluator())

    def engineMove(self):
        return self.engine.play(self.board, chess.engine.Limit(depth=self.depth)).move

    def engineScore(self):
        return self.engine.analyse(self.board, chess.engine.Limit(depth=self.depth))["score"]

    def isGameOver(self):
        return self.board.is_game_over()

    def quit(self):
        self.engine.quit()

    def run(self):
        while not self.isGameOver():
            self.move(self.simpleAIMove())
            self.move(self.engineMove())
        print(self.board.result())
