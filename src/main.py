from prober import Prober
from uci_prober_engine import UCIProberEngine

def main() -> None:
    # tablebase_prober = Prober()

    # fen = "8/8/p7/8/p4P2/8/k1P5/5K2 w - - 0 1"
    # tablebase_prober.evaluate_position(fen)
    
    engine = UCIProberEngine()
    engine.uci_loop()


if __name__ == "__main__":
    main()