import chess
import chess.engine

from evaluators.evaluator import Evaluator

STOCKFISH_PATH = "stockfish"
STOCKFISH_PATH_WINDOWS = "C:\\Users\\diveb\\Downloads\\stockfish-11-win\\stockfish-11-win\\Windows\\stockfish_20011801_x64.exe"

class StockfishEvaluator(Evaluator):
    def __init__(self, isWindows: bool):
        if isWindows:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH_WINDOWS)
        else:
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

        return self.engine.analyse(board, chess.engine.Limit(depth=2))["score"].white().score(mate_score=100000)

    def quit(self):
        self.engine.quit()
