import chess

from evaluators import Evaluator

def alphabetaMinimax(board: chess.Board, alpha: int, beta: int, depth: int, evaluator: Evaluator):
    """
    Compute the minimax algorithm given a position, values of alpha and beta and a depth

    Arguments:
    ----------
    - board : A Board object that represents the position
    - alpha : The value of the alpha parameter
    - beta : The value of the beta parameter
    - depth : The depth in which to search the best move
    - evaluator : The evaluator to use for the evaluation

    Return:
    -------
    The score associated to the given board (when a move has been done)
    """
    bestScore = -9999

    # Terminal State
    if board.is_game_over():
        return evaluator.evaluate(board)

    # Depth reached
    if depth == 0:
        return quiescentSearch(board, alpha, beta, evaluator)

    for move in board.legal_moves:
        board.push(move)
        score = -alphabetaMinimax(board, -beta, -alpha, depth - 1, evaluator)
        board.pop()

        if score > bestScore:
            bestScore = score

        if score > alpha:
            alpha = score

        if score >= beta:
            return score
    return bestScore

def quiescentSearch(board: chess.Board, alpha: int, beta: int, evaluator: Evaluator):
    """
    Performs a quiescent search to avoid the Horizon Effect.
    Avoid the AI to make bad move because it didn't see a capture in what follows.

    Arguments:
    ----------
    - board : A Board object that represents the position
    - alpha : The value of the alpha parameter
    - beta : The value of the beta parameter
    - evaluator : The evaluator to use for the evaluation

    Return:
    -------
    The evaluation of the board for the best move to do
    """
    score = evaluator.evaluate(board)
    if score >= beta:
        return beta
    if score > alpha:
        alpha = score

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescentSearch(board, -beta, -alpha, evaluator)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

def searchNextMove(board: chess.Board, depth: int, evaluator: Evaluator):
    """
    Searches the given board for thebest move to play

    Arguments:
    ----------
    - board : A Board object to search in
    - depth : The maximum depth at which the AI need to search
    - evaluator : The evaluator to use for the evaluation

    Return:
    -------
    A Move object that represents the best move to play
    """
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabetaMinimax(board, -beta, -alpha, depth-1, evaluator)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if(boardValue > alpha):
            alpha = boardValue
        board.pop()
    return bestMove
