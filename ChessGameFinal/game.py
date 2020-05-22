import chess
import chess.engine

from evaluator import Evaluator
from simpleEvaluator import SimpleEvaluator
from deepEvaluator import DeepEvaluator
from stockfishEvaluator import StockfishEvaluator

from minimax import searchNextMove

STOCKFISH_PATH = "stockfish"

class Game:
    def __init__(self, depth: int):
        self.depth = depth

        self.board = chess.Board()

        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        self.stockfish = StockfishEvaluator()

        self.answer = True

    def move(self, move: chess.Move):
        self.board.push(move)

    def simpleAIMove(self):
        return searchNextMove(self.board, self.depth, SimpleEvaluator())

    def simpleAIScore(self):
        return SimpleEvaluator().evaluate(self.board)

    def deepAIMove(self):
        return searchNextMove(self.board, self.depth, DeepEvaluator(True))

    def engineMove(self):
        return searchNextMove(self.board, self.depth, self.stockfish)

    def engineScore(self):
        return self.engine.analyse(self.board, chess.engine.Limit(depth=self.depth))["score"].white().score(mate_score=2048)

    def isGameOver(self):
        return self.board.is_game_over()

    def quit(self):
        self.engine.quit()
        self.stockfish.quit()

    def run(self):
        while not self.isGameOver():
            self.move(self.deepAIMove())
            self.move(self.engineMove())
        print(self.board.result())
