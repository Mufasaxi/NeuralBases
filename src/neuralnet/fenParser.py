import numpy as np
import chess
import chess.syzygy
def board_to_matrix(board: chess.Board):

    matrix = np.zeros((12, 8, 8))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_colour = 0 if piece.color else 6
        matrix[piece_type + piece_colour, row, col] = 1
    return matrix

def create_nn_input(fens: list[str]):
    X = []
    y = []
    # TABLEBASE_PATH = "/../endgame_tablebase"
    TABLEBASE_PATH = "D:/musta/Github Projects/NeuralBases/endgame_tablebase"


    # Initialise the tablebase reader
    tablebase = chess.syzygy.open_tablebase(TABLEBASE_PATH)
    for fen in fens:
        board = chess.Board(fen)
        X.append(board_to_matrix(board))
        y.append(tablebase.probe_wdl(board))
    return np.array(X, dtype=np.float32), np.array(y)
