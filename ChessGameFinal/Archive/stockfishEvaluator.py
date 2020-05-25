import chess
import chess.engine

from evaluator import Evaluator

STOCKFISH_PATH = "stockfish"

class StockfishEvaluator(Evaluator):
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    def evaluate(self, board: chess.Board):
        """
        Evaluate a given position with the stockfish engine

        Arguments:
        ----------
        board : A chess.Board object representing the position to evaluate

        Return:
        -------
        A score as an integer from -9999 to 9999
        """

        return self.engine.analyse(board, chess.engine.Limit(depth=2))["score"].white().score(mate_score=3500)

    def quit(self):
        self.engine.quit()
