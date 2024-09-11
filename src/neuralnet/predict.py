from model import EndgameModel
from fenParser import board_to_matrix
import chess
import chess.syzygy
import torch
import time

def load_model(model_path: str) -> EndgameModel:
    """
    Takes the stored model weights and initialises a model using them.

    Parameters:
        model_path: relative path to the stored model weights (usually a .pth file).

    Returns:
        EndgameModel: An instance of the EndgameModel initialised with the saved training weights.
    """
    model = EndgameModel()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Takes an FEN string (ex: "2K5/4q3/8/3B4/k7/8/8/8 w - - 0 1"), and converts it to a pytorch Tensor of the matrix board representation.

    Parameters:
        fen: FEN string of the form "2K5/4q3/8/3B4/k7/8/8/8 w - - 0 1"
    
    Returns:
        Tensor: Tensor of the matrix board representation of the respective FEN string        
    """
    board_matrix = board_to_matrix(chess.Board(fen))
    input_tensor = torch.FloatTensor(board_matrix).unsqueeze(0)
    return input_tensor

def predict(model: EndgameModel, input_tensor: torch.Tensor) -> int:
    """"
    Uses the trained model to predict the WDL value of a given Tensor of an FEN string

    Parameters:
        model: EndgameModel initialised with training weights.
        input_tensor: Pytorch Tensor of the FEN string of the position to be probed
    
    Returns:
        WDL: Number indicating the evaluation of the position for the side to move.
                {-2: Loss, -1: Blessed loss, 0: Equal, 1: Cursed win, 2: Win}
    """
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
    return predicted.item() - 2

if __name__ == '__main__':
    fen_string = input("Enter an FEN: ") #"2k5/4q3/8/3b4/k7/8/8/8 w - - 0 1"
    input_tensor = fen_to_tensor(fen_string)
    
    time_start_50Epoch_prediction = time.monotonic()
    model = load_model("endgame_cnn50epochs.pth") 
    prediction = predict(model, input_tensor)
    time_end_50Epoch_prediction = time.monotonic()
    prediction_time_50Epoch = time_end_50Epoch_prediction - time_start_50Epoch_prediction

    print(f"Input FEN: {fen_string}") 
    print(chess.Board(fen_string))
    print(f"Predicted WDL (50 Epoch Model): {prediction} ({prediction_time_50Epoch:.3f} s)")
    
    time_start_10Epoch_prediction = time.monotonic()
    model = load_model("endgame_cnn.pth")
    prediction = predict(model, input_tensor)
    time_end_10Epoch_prediction = time.monotonic()
    prediction_time_10Epoch = time_end_10Epoch_prediction - time_start_10Epoch_prediction
    print(f"Predicted WDL (10 Epoch Model): {prediction} ({prediction_time_10Epoch:.3f} s)")
    
    time_start_tablebase = time.monotonic()
    tablebase = chess.syzygy.open_tablebase("D:/musta/Github Projects/NeuralBases/endgame_tablebase")
    time_end_tablebase = time.monotonic()
    probing_time_tablebase = time_end_tablebase - time_start_tablebase
    wdl = tablebase.probe_wdl(chess.Board(fen_string))
    print(f"Actual WDL: {wdl} ({probing_time_tablebase:.3f} s)")