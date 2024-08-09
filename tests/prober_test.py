from src.tablebase.prober import Prober # This doesn't work for some reason running "python -m pytest tests/prober_test.py" in cmd from root works.

prober: Prober = Prober()

# Positive Test Cases

def test_only_kings() -> None:
    kings_fen = "8/8/4k3/8/8/2K5/8/8 w - - 0 1"
    kings_wdl, _ = prober.evaluate_position(kings_fen)
    assert kings_wdl == 0

def test_equal_pawns() -> None:
    pawns_fen = "8/8/5k1p/6p1/2P5/3P4/4K3/8 w - - 0 1"
    pawns_wdl, _ = prober.evaluate_position(pawns_fen)
    assert pawns_wdl == 0

def test_equal_pieces() -> None:
    pieces_fen = "8/4n3/2b2k2/8/1BN5/8/4K3/8 w - - 0 1"
    pieces_wdl, _ = prober.evaluate_position(pieces_fen)
    assert pieces_wdl == 0

def test_winning() -> None:
    winning_fen = "8/8/5k2/8/1B6/6R1/3QK3/8 w - - 0 1"
    winning_wdl, _ = prober.evaluate_position(winning_fen)
    assert winning_wdl > 0

def test_losing() -> None:
    losing_fen = "8/8/2r2k2/3n2q1/8/8/4K3/8 w - - 0 1"
    losing_wdl, _ = prober.evaluate_position(losing_fen)
    assert losing_wdl < 0


# Negative Test Cases

def test_nonstring_input() -> None:
    invalid_fen = 8/8/8/8/8/8/8/8
    invalid_fen_wdl, invalid_fen_best_move = prober.evaluate_position(invalid_fen)
    assert (invalid_fen_wdl, invalid_fen_best_move) == (None, None)

def test_invalid_position() -> None:
    invalid_position = "8/8/8/3k4/4K3/8/8/8 w - - 0 1"
    invalid_position_wdl, invalid_position_best_move = prober.evaluate_position(invalid_position)
    assert (invalid_position_wdl, invalid_position_best_move) == (None, None)

def test_too_many_pieces() -> None:
    too_many_pieces = "8/6P1/2K1P3/8/2p5/3N3k/1n6/r7 w - - 0 1"
    too_many_pieces_wdl, too_many_pieces_best_move = prober.evaluate_position(too_many_pieces)
    assert (too_many_pieces_wdl, too_many_pieces_best_move) == (None, None)

def test_too_few_pieces() -> None:
    too_few_pieces = "8/8/8/8/2K5/8/8/8 w - - 0 1"
    too_few_pieces_wdl, too_few_pieces_best_move = prober.evaluate_position(too_few_pieces)
    assert (too_few_pieces_wdl, too_few_pieces_best_move) == (None, None)


# Extreme Test Cases with all queens to see if there would be a noticeable delay
def test_four_queens() -> None:
    four_queens = "8/7q/q7/Q7/3k4/Q5K1/8/8 w - - 0 1"
    four_queens_wdl, four_queens_best_move = prober.evaluate_position(four_queens)
    assert (four_queens_wdl, four_queens_best_move) != (None, None)