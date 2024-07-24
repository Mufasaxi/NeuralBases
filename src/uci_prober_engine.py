import chess
from prober import Prober
from minmax_engine import MinMaxEngine

class UCIProberEngine:
    """
    UCIProberEngine class represents a chess engine that communicates using the UCI (Universal Chess Interface) protocol.
    It can handle commands like 'position', 'go', 'isready', and 'quit'.
    """

    def __init__(self) -> None:
        """
        Initializes the UCIProberEngine with a Prober instance and a chess.Board instance.
        """
        self.prober: Prober = Prober()
        self.board: chess.Board = chess.Board()
        self.minmax_engine = MinMaxEngine(self.prober)

    def handle_uci(self) -> None:
        """
        Handles the 'uci' command by printing engine information required by the UCI protocol.
        """
        print("id name ProberEngine")
        print("id author Mustafa")
        print("uciok")

    def handle_isready(self)-> None:
        """
        Handles the 'isready' command by printing 'readyok', indicating readiness to proceed.
        """
        print("readyok")

    def handle_position(self, command:str) -> None:
        """
        Handles the 'position' command to set up the board with a specific position.

        Parameters:
            command: The full 'position' command received from UCI protocol.

        Example:
        - position startpos moves e2e4 e7e5
        - position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1

        Note: Since this is made for use with the prober only endgame positions (with fewer than 6 pieces) should be used
        """
        parts = command.split()
        if "startpos" in parts:
            self.board.set_fen(chess.STARTING_FEN)
            moves_index = parts.index("moves")
            moves = parts[moves_index + 1:]
        else:
            fen_parts = command.split("fen")[1].split(" moves")
            self.board.set_fen(fen_parts[0].strip())
            if len(fen_parts) > 1:
                moves = fen_parts[1].strip().split()
            else:
                moves = []

        for move in moves:
            try:
                self.board.push_uci(move)
            except chess.InvalidMoveError:
                print(f"Invalid move syntax: {move}")
                return
            except chess.IllegalMoveError:
                print(f"Illegal move: {move}")
                return

        if not self.prober.is_valid_fen(self.board.fen()):
            self.board = chess.Board()

    def handle_go(self) -> int:
        """
        Handles the 'go' command to make a move for the engine based on current board position.
        """
        if self.board.is_checkmate():
            print("info string Checkmate detected.")
            return 0
        if self.board.is_stalemate():
            print("info string Stalemate detected.")
            return 0
        if self.board.is_insufficient_material():
            print("info string Insufficient material detected.")
            return 0
        if self.board.is_seventyfive_moves():
            print("info string Seventy-five move rule draw.")
            return 0
        if self.board.is_fivefold_repetition():
            print("info string Fivefold repetition draw.")
            return 0
        if self.board.is_repetition():
            print("info string Threefold repetition draw.")
            return 0

        best_move = self.minmax_engine.get_best_move(self.board)
        if best_move is not None:
            print(f"bestmove {best_move.uci()}")
            self.board.push(best_move)
            print(self.board)
            return 2 if self.board.is_checkmate() else 1
        return 0

    def handle_selfplay(self) -> str:
        """
        Handles the 'selfplay' command to play a game between two instances of the engine.

        The method alternates making moves between two engines until the game is over.
        """
        print(self.board)
        while not self.board.is_game_over():
            print(self.board)
            print()
            if self.board.is_game_over() or self.handle_go() == 0:
                break
            self.handle_go()
            print()

        result = self.board.result()
        print(f"Game over. Result: {result}")
        return

    def uci_loop(self):
        """
        Main loop to handle input commands from UCI interface until 'quit' command is received.

        Commands handled:
        - uci
        - isready
        - position
        - go
        - selfplay
        - quit
        """
        while True:
            try:
                command = input().strip()
                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    self.handle_isready()
                elif command.startswith("position"):
                    self.handle_position(command)
                elif command.startswith("go"):
                    self.handle_go()
                elif command == "selfplay":
                    self.handle_selfplay()
                    break
                elif command == "quit":
                    break
            except (EOFError, KeyboardInterrupt):
                break

if __name__ == "__main__":
    engine = UCIProberEngine()
    engine.uci_loop()
