import re
import chess
import chess.syzygy
import logging

class Prober:

    def __init__(self):
        self.TABLEBASE_PATH = "../../endgame_tablebase"

        # Initialise the tablebase reader
        self.tablebase = chess.syzygy.open_tablebase(self.TABLEBASE_PATH)

    def invert_wdl(self, wdl: int) -> int:
        """Inverts the WDL value to reflect the perspective of the opponent."""
        if wdl is None:
            return None
        return -wdl
    
    def are_kings_connected(self, fen):
        """Returns True if kings connected, false otherwise."""
        board = chess.Board(fen)
        white_king_pos = board.king(chess.WHITE)
        black_king_pos = board.king(chess.BLACK)
        
        if white_king_pos is None or black_king_pos is None:
            return False
        
        white_king_rank = chess.square_rank(white_king_pos)
        white_king_file = chess.square_file(white_king_pos)
        black_king_rank = chess.square_rank(black_king_pos)
        black_king_file = chess.square_file(black_king_pos)
        
        rank_diff = abs(white_king_rank - black_king_rank)
        file_diff = abs(white_king_file - black_king_file)
        
        return rank_diff <= 1 and file_diff <= 1
    
    def is_valid_fen(self, fen):
        """Check if a given FEN is valid"""
        # Regular expression to validate the general structure of an FEN string
        fen_pattern = re.compile(
            r"^([rnbqkpRNBQKP1-8]{1,8}/){7}[rnbqkpRNBQKP1-8]{1,8} [wb] (K?Q?k?q?|-) ([a-h][36]|-) \d+ \d+$"
        )
        
        if not fen_pattern.match(fen):
            return False

        try:
            board = chess.Board(fen)
        except ValueError:
            return False

        if not board.is_valid():
            print("Invalid board")
            return False

        if len(board.pieces(chess.KING, chess.WHITE)) != 1 or len(board.pieces(chess.KING, chess.BLACK)) != 1:
            print("Wrong number of kings in given position")
            return False
        
        total_pieces = sum(len(board.pieces(piece_type, color)) 
                        for piece_type in chess.PIECE_TYPES 
                        for color in [chess.WHITE, chess.BLACK])
        
        if total_pieces > 6:
            print("Position contains too many pieces. Must contain at most 6 pieces (including Kings)")
            return False

        return True

    def evaluate_position(self, fen: str) -> tuple[int, str]:
        """
        Finds the WDL and the best move correlated to it. Outputs the given position in the terminal
        
        Parameters:
            fen: FEN string of a valid position containing no more than 6 pieces

        Returns:
            WDL: Number indicating the evaluation of the position for the side to move.
                {-2: Loss, -1: Blessed loss, 0: Equal, 1: Cursed win, 2: Win}
            Best Move: Respective best move for the given position in accordance with the WDL
        """

        if (type(fen) != str):
            print("Invalid input, must be string")
            return None, None
        
        if (not self.is_valid_fen(fen)):
            return None, None

        if (self.are_kings_connected(fen) or not self.is_valid_fen(fen)):
            return None, None

        board = chess.Board(fen)
        print("Board Position:")
        print(board)

        initial_wdl: int = self.tablebase.probe_wdl(board)
        if initial_wdl is None:
            print("Position not found in tablebase.")
            return None, None

        logging.debug("Initial WDL: %s", initial_wdl)
        

        best_move: str = None
        best_wdl: int = -float('inf')  # Initialize to the lowest possible value

        # Probe for the best move
        for move in board.legal_moves:
            board.push(move)
            move_wdl: int = self.tablebase.probe_wdl(board)
            board.pop()

            # Invert the WDL to reflect the perspective of the side to move
            move_wdl = self.invert_wdl(move_wdl)
            logging.debug(f"Evaluating move: {move}, Move WDL (inverted): {move_wdl}")


            # Update best move if this move has a better WDL value
            if move_wdl is not None and move_wdl > best_wdl:
                best_wdl = move_wdl
                best_move = move

        print(f"WDL: {initial_wdl}, Best Move: {best_move}")

        if (best_wdl == initial_wdl):
            return initial_wdl, best_move
        return None, None

if __name__ == "__main__":
    # Test output TODO: find a way to randomise FEN (maybe regex?)
    fen: str = "8/3p4/3p4/K5k1/P2P4/8/8/8 w - - 0 1"

    fentest = "8/4k3/2Q5/R7/8/8/4K3/8 w - - 0 1"
    prober = Prober()
    initial_wdl, best_move = prober.evaluate_position(fentest)
