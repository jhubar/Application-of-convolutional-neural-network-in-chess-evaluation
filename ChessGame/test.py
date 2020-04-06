import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/usr/local/lib/python3.7/site-packages/stockfish")

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)

engine.quit()
