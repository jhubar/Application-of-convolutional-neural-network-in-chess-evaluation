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

        self.setWindowTitle("Chess GUI")
        self.setGeometry(200, 200, 600, 600)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 600, 600)

        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())

        self.margin = 0.05 * self.boardSize
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0

        self.board = chess.Board()
        self.pieceToMove = [None, None]
        self.selectedPiece = [None, None]
        self.legalSquares = None
        self.lastMove = None

        self.depth = depth

        self.updateBoard()

    def isLegalMove(self, targetSquare):
        if self.legalSquares is None:
            return False
        for square in self.legalSquares:
            if targetSquare == square:
                return True
        return False

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """
        if event.x() <= self.boardSize and event.y() <= self.boardSize and not self.board.is_game_over():
            if event.buttons() == Qt.LeftButton:
                if self.margin < event.x() < self.boardSize - self.margin and self.margin < event.y() < self.boardSize - self.margin:
                    col = int((event.x() - self.margin) / self.squareSize)
                    row = 7 - int((event.y() - self.margin) / self.squareSize)
                    square = chess.square(col, row)

                    if self.isLegalMove(square):
                        move = chess.Move(self.selectedPiece[1], square)
                        self.board.push(move)

                        self.selectedPiece = [None, None]
                        self.legalSquares = None

                        # AI TURN
                        aiMove = searchNextMove(self.board, self.depth)
                        self.board.push(aiMove)
                        self.lastMove = aiMove

                        if self.board.is_game_over():
                            print("END")
                            self.updateBoard()
                            return
                    else:
                        piece = self.board.piece_at(square)
                        self.selectedPiece = [piece, square]
                        self.legalSquares = chess.SquareSet()
                        legalMoves = self.board.legal_moves
                        for move in legalMoves:
                            if move.from_square == self.selectedPiece[1]:
                                self.legalSquares.add(move.to_square)
                self.updateBoard()
        else:
            QWidget.mousePressEvent(self, event)

    def updateBoard(self):
        """
        Draw a chessboard
        """
        if self.selectedPiece[1] is not None:
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
    parser = argparse.ArgumentParser(description="Arguments of the Chess Game")

    # Depth
    parser.add_argument("-d",
                        "--depth",
                        type=int,
                        action="store",
                        default=2,
                        help="Depth of the Negamax search")

    args = parser.parse_args()

    depth = args.depth

    chessTitan = QApplication(sys.argv)
    window = MainWindow(depth)
    window.show()
    sys.exit(chessTitan.exec_())
