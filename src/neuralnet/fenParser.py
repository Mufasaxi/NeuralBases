import numpy as np
import chess
import chess.syzygy
def board_to_matrix(board: chess.Board):
    """
    Takes a chess.Board instance and returns a (12, 8, 8) matrix representing it.

    The first 6 8x8 matrices are the white pawn, knight, bishop, rook, queen, and king, the second set of 6 are the black pieces in the same order.

    Each 8x8 matrix is a bitboard for the respective piece
    """
    matrix = np.zeros((12, 8, 8))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_colour = 0 if piece.color else 6
        matrix[piece_type + piece_colour, row, col] = 1
    return matrix

def create_nn_input(fens: list[str]):
    """
    Takes a list of FENs as input, and for each element, creates an element in the array X representing the matrix representation of the FEN board, and another element in the array y representing the WDL of the board.

    Returns X, y as a tuple of np.arrays
    """
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
