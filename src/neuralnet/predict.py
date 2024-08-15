from model import EndgameModel
from fenParser import board_to_matrix
import chess
import torch

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
    model = load_model("endgame_cnn50epochs.pth")
    fen_string = input("Enter an FEN: ") #"2k5/4q3/8/3b4/k7/8/8/8 w - - 0 1"
    input_tensor = fen_to_tensor(fen_string)
    prediction = predict(model, input_tensor)
    print(f"Input FEN: 2K5/4q3/8/3B4/k7/8/8/8 w - - 0 1")
    print(chess.Board(fen_string))
    print(f"Predicted WDL (50 Epoch Model): {prediction}")
    mode = load_model("endgame_cnn.pth")
    prediction = predict(model, input_tensor)
    print(f"Predicted WDL (10 Epoch Model): {prediction}")