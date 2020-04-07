import sys
import argparse

import asyncio
import chess
import chess.svg
import chess.engine

from stockfish import Stockfish

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

from AIChess import searchNextMove
from AIChess import evaluate
from AIChess import deepEvaluation





class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, depth):
        """
        Initialize the chessboard.
        """
        super().__init__()
        # Window
        self.setWindowTitle("Chess GUI")
        self.setGeometry(200, 200, 600, 600)
        self.answer = True

        # Widget, canvas for the chessboard
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 600, 600)

        # Board Dimensions
        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())
        self.margin = 0.05 * self.boardSize
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0

        # Board
        self.board = chess.Board()

        # Square selected for a move
        self.selectedSquare = None

        # Set of legal squares to move to
        self.legalSquares = None

        # Last move played by the IA
        self.lastMove = None

        # Depth of the search for the IA
        self.depth = depth

        # Displays the board
        self.updateBoard()

    def evaluateMove(self):
        if self.lastMoveScore < self.currentScore:
            return "a bad move."
        elif self.lastMoveScore == self.currentScore:
            return "a not bad move."
        else:
            return "a good move."


    def whoIsWinning(self):
        if self.currentBlackScore < self.currentWhiteScore:
            return "The whites are winning."
        elif -self.currentBlackScore == self.currentWhiteScore:
            return "Black and white are equal ."
        else:
            return "The black are winning."

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """

        # If the event is inside the chessboard
        if event.x() <= self.boardSize and event.y() <= self.boardSize and not self.board.is_game_over():
            # If the event is a left click
            if event.buttons() == Qt.LeftButton:
                # If the event is not on the side notations
                if self.margin < event.x() < self.boardSize - self.margin and self.margin < event.y() < self.boardSize - self.margin:
                    # Column clicked
                    column = int((event.x() - self.margin) / self.squareSize)

                    # Row clicked
                    row = 7 - int((event.y() - self.margin) / self.squareSize)

                    # Square selected
                    square = chess.square(column, row)

                    # If there is a previously selected square and
                    # if the selected square belongs to the set of legal squares
                    if self.answer:
                        print("Le meilleur move est: ",searchNextMove(self.board,self.depth))
                        print("bla bla bla", self.stockfish.get_best_move())
                        self.answer = False

                    if self.legalSquares is not None and square in self.legalSquares:
                        self.answer = True
                        # Creates move
                        move = chess.Move(self.selectedSquare, square)
                        # save last move score
                        self.lastMoveScore = evaluate(self.board)
                        # print("LastMoveScore : ",self.lastMoveScore)
                        # Make move
                        self.board.push(move)
                        # Save current score
                        self.currentScore = evaluate(self.board)

                        # deepEvaluation(self.board)

                        self.currentWhiteScore = self.currentScore
                        # print("CurrentScore : ",self.currentScore)
                        print("White plays",self.evaluateMove(),"The current white score is: ", "%.2f" % round((self.currentWhiteScore/9999)*20,2))

                        # Unset temporary variables
                        self.selectedSquare = None
                        self.legalSquares = None


                        # Check game end
                        if self.board.is_game_over():
                            print("White wins")
                            self.updateBoard()
                            return

                        # AI TURN
                        # AI selection of the best move
                        # stockfish

                        # result = engine.play(board, chess.engine.Limit(time=0.1))

                        print("bla bla bla", self.stockfish.get_best_move())


                        # Make move
                        self.lastBlackScore = evaluate(self.board)
                        print("LastMoveScore : ",-self.lastMoveScore)
                        aiMove = searchNextMove(self.board, self.depth)
                        self.board.push(aiMove)
                        self.currentScore = evaluate(self.board)
                        self.currentBlackScore = -self.currentScore
                        # print("currentscore : ",-self.currentScore)
                        print("Black plays", self.evaluateMove() ,"The current black score is: ", "%.2f" % round((self.currentBlackScore/9999)*20,2))
                        print(self.whoIsWinning())
                        # Register last move
                        self.lastMove = aiMove

                        # Check game end
                        if self.board.is_game_over():
                            print("Black wins")
                            engine.quit()
                            self.updateBoard()
                            return
                    # If first selection of a square or click outside legal moves
                    else:
                        # Register selected square
                        self.selectedSquare = square

                        # Compute all legal moves from the selected square
                        self.legalSquares = chess.SquareSet()
                        legalMoves = self.board.legal_moves
                        for move in legalMoves:
                            if move.from_square == self.selectedSquare:
                                self.legalSquares.add(move.to_square)

                # Update the board
                self.updateBoard()
        else:
            QWidget.mousePressEvent(self, event)

    def updateBoard(self):
        """
        Draw a chessboard
        """
        # If a piece has been selected
        if self.selectedSquare is not None:
            self.boardSvg = chess.svg.board(board=self.board,
                                            size=self.boardSize,
                                            lastmove=self.lastMove,
                                            squares=self.legalSquares).encode("UTF-8")
        else:
            self.boardSvg = chess.svg.board(board=self.board,
                                            lastmove=self.lastMove,
                                            size=self.boardSize).encode("UTF-8")
        self.widgetSvg.load(self.boardSvg)


if __name__ == "__main__":



    # Create argument parser
    parser = argparse.ArgumentParser(description="Arguments of the Chess Game")

    # Depth
    parser.add_argument("-d",
                        "--depth",
                        type=int,
                        choices=range(1, 10),
                        action="store",
                        default=2,
                        help="Depth of the Negamax search")

    # Fetch arguments
    args = parser.parse_args()

    # Extract depth
    depth = args.depth

    # Create Qt application
    chessGame = QApplication(sys.argv)

    # Create chess game window
    window = MainWindow(depth)
    stockfish = Stockfish()
    # Display window
    window.show()

    # Run and exit
    sys.exit(chessGame.exec_())
