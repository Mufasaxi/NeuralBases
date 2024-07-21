import chess
import chess.engine
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", filemode='w', format='%(asctime)s, %(msecs)d, %(message)s', level=logging.DEBUG)

# source: chessprogrammingwiki
# realtive piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# square values indicating effetive of a piece on that square
PIECE_SQUARE_TABLES = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ],
    chess.KING: [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    ]
}

class MinMaxEngine:
    """
    MinMaxEngine is the brains alongside the prober. It uses MinMax with Alpha-Beta Pruning to find the best move within a given depth
    """
    def __init__(self, prober) -> None:
        self.prober = prober

    def evaluate_position(self, board: chess.Board) -> int:
        """
        returns an evaluation for a given position. The higher the better it is for the side to move. Positive values favour white and negative favour black.

        Parameters:
            board: Representation of the position with chess.Board

        Returns: 
            evaluation: 9999 if mate for white and -9999 for black.
        """

        if board.is_checkmate():
            return -9999 if board.turn else 9999
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0
        else:
            eval = 0
            for piece_type in chess.PIECE_TYPES:
                # Evaluate material values
                eval += sum(PIECE_VALUES.get(piece_type, 0) for piece in board.pieces(piece_type, chess.WHITE))
                eval -= sum(PIECE_VALUES.get(piece_type, 0) for piece in board.pieces(piece_type, chess.BLACK))

                # Evaluate piece-square tables
                for square in board.pieces(piece_type, chess.WHITE):
                    sq_index = chess.square_mirror(square)  # mirror the square for piece-square table indexing
                    eval += PIECE_SQUARE_TABLES.get(piece_type, [0]*64)[sq_index]
                for square in board.pieces(piece_type, chess.BLACK):
                    sq_index = chess.square_mirror(square)  # mirror the square for piece-square table indexing
                    eval -= PIECE_SQUARE_TABLES.get(piece_type, [0]*64)[sq_index]
                
            return eval

    def minmax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximising_player: bool) -> tuple[int, chess.Move]:
        if depth == 0 or board.is_game_over():
            eval = self.evaluate_position(board)
            logger.debug(f"Evaluating position at depth {depth}: {eval}")
            return eval, None
        
        # Check for potential checkmate situation
        if self.is_checkmate_likely(board):
            depth = 1  # Do full depth search for this level only

        best_move = None

        if is_maximising_player:
            max_eval = -float("inf")
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minmax(board, depth - 1, alpha, beta, False)
                board.pop()
                logger.debug(f"Maximizing: Move {move} has eval {eval}\n")
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    logger.debug(f"Pruning at depth:{depth}\n")
                    break
            return max_eval, best_move

        else:
            min_eval = float("inf")
            for move in board.legal_moves:
                board.push(move)
                eval, _ = self.minmax(board, depth - 1, alpha, beta, True)
                board.pop()
                logger.debug(f"Minimizing: Move {move} has eval {eval}\n")
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    logger.debug(f"Pruning at depth:{depth}\n")
                    break
            return min_eval, best_move


    def get_best_move(self, board: chess.Board, depth: int = 5) -> chess.Move:
        _, best_move = self.minmax(board, depth, -float("inf"), float("inf"), True)
        return best_move

    def is_checkmate_likely(self, board):
        # Naive implemenation should either be removed or improved
        return len(list(board.legal_moves)) <= 3 
