import sys
import argparse

import chess
import chess.svg

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

from AIChess import searchNextMove


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
                    if self.legalSquares is not None and square in self.legalSquares:
                        # Creates move
                        move = chess.Move(self.selectedSquare, square)

                        # Make move
                        self.board.push(move)

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
                        aiMove = searchNextMove(self.board, self.depth)

                        # Make move
                        self.board.push(aiMove)

                        # Register last move
                        self.lastMove = aiMove

                        # Check game end
                        if self.board.is_game_over():
                            print("Black wins")
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

    # Display window
    window.show()

    # Run and exit
    sys.exit(chessGame.exec_())
