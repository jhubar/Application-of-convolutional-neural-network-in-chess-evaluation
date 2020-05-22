import sys
import argparse

import chess
import chess.svg
import chess.engine

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

from game import Game

class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, game):
        """
        Initialize the chessboard window.
        """
        super().__init__()
        # Window
        self.setWindowTitle("Chess UI")
        self.setGeometry(200, 200, 600, 600)

        # Widget, canvas for the chessboard
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 600, 600)

        # Board Dimensions
        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())
        self.margin = 0.05 * self.boardSize
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0

        # Square selected for a move
        self.selectedSquare = None

        # Set of legal squares to move to
        self.legalSquares = None

        # Last move played by the IA
        self.lastMove = None

        # Game object
        self.game = game

        # Displays the board
        self.updateBoard()

    def closeEvent(self, event):
        self.game.quit()

    # def evaluateMove(self):
    #     if self.lastMoveScore < self.currentScore:
    #         return "a bad move."
    #     elif self.lastMoveScore == self.currentScore:
    #         return "a not bad move."
    #     else:
    #         return "a good move."

    # def whoIsWinning(self):
    #     if self.currentBlackScore < self.currentWhiteScore:
    #         return "The whites are winning."
    #     elif -self.currentBlackScore == self.currentWhiteScore:
    #         return "Black and white are equal ."
    #     else:
    #         return "The black are winning."

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """

        # If the event is inside the chessboard
        if event.x() <= self.boardSize and event.y() <= self.boardSize and not self.game.isGameOver():
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

                    # if self.game.answer:
                    #     print("MinMax move proposition: ", self.game.simpleAIMove())

                    #     print("stockfish move proposition", self.game.engineMove())
                    #     print("stockfish score evaluation", self.game.engineScore())

                    #     self.game.answer = False

                    # If there is a previously selected square and
                    # if the selected square belongs to the set of legal squares
                    if self.legalSquares is not None and square in self.legalSquares:
                        # self.game.answer = True
                        # Creates move
                        move = chess.Move(self.selectedSquare, square)
                        # save last move score
                        # self.lastMoveScore = self.game.simpleAIScore()

                        # Make move
                        self.game.move(move)
                        # Save current score
                        # self.currentScore = self.game.simpleAIScore()

                        # self.currentWhiteScore = self.currentScore

                        # print("White plays",self.evaluateMove(),"The current white score is: ", "%.2f" % round((self.currentWhiteScore/9999)*20,2))

                        # Unset temporary variables
                        self.selectedSquare = None
                        self.legalSquares = None

                        # Check game end
                        if self.game.isGameOver():
                            print("White wins")
                            self.updateBoard()
                            return

                        # AI TURN
                        # Make move
                        # self.lastBlackScore = self.game.simpleAIScore()

                        # print("LastMoveScore : ", -self.lastMoveScore)

                        # aiMove = self.game.simpleAIMove()
                        aiMove = self.game.deepAIMove()
                        # aiMove = self.game.engineMove()

                        self.game.move(aiMove)

                        # self.currentScore = self.game.simpleAIScore()

                        # self.currentBlackScore = -self.currentScore

                        # print("Black plays", self.evaluateMove() ,"The current black score is: ", "%.2f" % round((self.currentBlackScore/9999)*20,2))

                        # print(self.whoIsWinning())

                        # Register last move
                        self.lastMove = aiMove

                        # Check game end
                        if self.game.isGameOver():
                            print("Black wins")
                            self.updateBoard()
                            return

                    # If first selection of a square or click outside legal moves
                    else:
                        # Register selected square
                        self.selectedSquare = square

                        # Compute all legal moves from the selected square
                        self.legalSquares = chess.SquareSet()
                        legalMoves = self.game.board.legal_moves
                        for move in legalMoves:
                            if move.from_square == self.selectedSquare:
                                self.legalSquares.add(move.to_square)

                # Update the board
                self.updateBoard()
        else:
            QWidget.mousePressEvent(self, event)

    def updateBoard(self):
        """
        Draw the chessboard
        """
        # Print score
        print("Current score from white perspective = {}".format(self.game.engineScore()))

        # If a piece has been selected
        if self.selectedSquare is not None:
            self.boardSvg = chess.svg.board(board=self.game.board,
                                            size=self.boardSize,
                                            lastmove=self.lastMove,
                                            squares=self.legalSquares).encode("UTF-8")
        else:
            self.boardSvg = chess.svg.board(board=self.game.board,
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
                        help="Depth of the Negamax search. Default = 2")

    # Silent mode
    parser.add_argument("-s",
                        "--silent",
                        action="store_true",
                        help="Flag for the hidden window mode")

    # Windows mode
    parser.add_argument("-w",
                        "--windows",
                        action="store_true",
                        help="Flags for windows user")

    # Fetch arguments
    args = parser.parse_args()

    # Extract depth
    depth = args.depth
    isSilent = args.silent
    isWindows = args.windows

    # Create Qt application
    chessGame = QApplication(sys.argv)

    game = Game(depth, isWindows)

    if isSilent:
        game.run()
        game.quit()
    else:
        # Create chess game window
        window = MainWindow(game)

        # Display window
        window.show()

        # Run and exit
        sys.exit(chessGame.exec_())
