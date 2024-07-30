import chess
import chess.engine
import logging
from prober import Prober

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", filemode='w', format='%(asctime)s, %(msecs)d, %(message)s', level=logging.DEBUG)

# source: chessprogrammingwiki [https://www.chessprogramming.org/Simplified_Evaluation_Function]
# realtive piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

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
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
}

class MinMaxEngine:
    """
    MinMaxEngine is the brains alongside the prober. It uses MinMax with Alpha-Beta Pruning to find the best move within a given depth
    """
    def __init__(self, prober) -> None:
        self.prober = prober
        self.transposition_table = {}
        self.killer_moves = {i:[None, None] for i in range(100)} # Moves that cause alpha beta cutoff
        self.prober = Prober() # Experiment to have the wdl influence the eval to avoid dumb moves

    def evaluate_position(self, board: chess.Board) -> int:
        """
        returns an evaluation for a given position. The higher the better it is for the side to move. Positive values favour white and negative favour black.

        Parameters:
            board: Representation of the position with chess.Board

        Returns: 
            evaluation: 9999 if mate for white and -9999 for black.
        """

        if board.is_checkmate():
            return float('-inf') if board.turn else float('inf')
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
                    #sq_index = chess.square_mirror(square)  # mirror the square for piece-square table indexing
                    eval += PIECE_SQUARE_TABLES.get(piece_type, [0]*64)[square]
                for square in board.pieces(piece_type, chess.BLACK):
                    # sq_index = chess.square_mirror(square)  # mirror the square for piece-square table indexing
                    eval -= PIECE_SQUARE_TABLES.get(piece_type, [0]*64)[chess.square_mirror(square)]
                wdl = self.prober.tablebase.probe_wdl(board)
                eval *= wdl
                # print("***********************")
                # print(board)
                # print(wdl)
                # print(eval)
                # print("***********************")

            return eval

    def minmax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximising_player: bool) -> tuple[int, chess.Move]:
        board_fen = board.fen()
        if board_fen in self.transposition_table:
            logger.debug(f"In transposition table: {board_fen}")
            return self.transposition_table[board_fen]

        if depth == 0 or board.is_game_over():
            eval = self.evaluate_position(board)
            logger.debug(f"Evaluating position at depth {depth}: {eval}")
            return eval, None

        best_move = None

        if is_maximising_player:
            max_eval = -float("inf")
            for move in sorted(board.legal_moves, key=lambda move: self.move_ordering_heuristic(board, move, depth), reverse=True):
                board.push(move)
                eval, _ = self.minmax(board, depth - 1, alpha, beta, False)
                board.pop()
                logger.debug(f"Maximizing: Move {move} has eval {eval}\n")
                
                if eval == float('inf') or eval == float('-inf'):
                    return eval, move
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    logger.debug(f"Pruning at depth:{depth}\n")
                    self.update_killer_moves(depth, move)
                    break
            self.transposition_table[board_fen] = (max_eval, best_move)
            return max_eval, best_move

        else:
            min_eval = float("inf")
            for move in sorted(board.legal_moves, key=lambda move: self.move_ordering_heuristic(board, move, depth), reverse=True):
                board.push(move)
                eval, _ = self.minmax(board, depth - 1, alpha, beta, True)
                board.pop()
                logger.debug(f"Minimizing: Move {move} has eval {eval}\n")

                # should be removed?
                if eval == 9999 or eval == -9999:
                    return eval, move

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    logger.debug(f"Pruning at depth:{depth}\n")
                    self.update_killer_moves(depth, move)
                    break
            self.transposition_table[board_fen] = (min_eval, best_move)
            return min_eval, best_move


    def get_best_move(self, board: chess.Board, depth: int = 6) -> chess.Move:
        _, best_move = self.minmax(board, depth, -float("inf"), float("inf"), True)
        return best_move
    
    def move_ordering_heuristic(self, board: chess.Board, move: chess.Move, depth: int) -> int:
        """
        Simple heuristic for move ordering: captures first, then checks, then others.
        """
        if board.gives_check(move):
            return 10  # High priority for checks
        elif board.is_capture(move):
            return 5  # Then captures
        elif move in self.killer_moves[depth]:
            return 3 # Then killer moves
        else:
            return 0  # Other moves
        
    def update_killer_moves(self, depth: int, move: chess.Move) -> None:
        """
        Update the killer moves table with the move that caused a beta cutoff.
        """
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop()
