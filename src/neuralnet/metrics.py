import re
import matplotlib.pyplot as plt
import numpy as np
from predict import load_model, predict, fen_to_tensor
from fenGenerator import FenGen
import chess
import chess.syzygy

def extract_data(filename: str) -> tuple[list[int], list[int], list[int], list[int]]:
    epochs = []
    train_loss = []
    val_loss = []
    val_accuracy = []

    with open(filename, 'r') as file:
        content = file.read()
        pattern = r"Epoch (\d+)/\d+:\nTrain Loss: ([\d.]+)\nVal Loss: ([\d.]+), Val Accuracy: ([\d.]+)"
        matches = re.findall(pattern, content)

        for match in matches:
            epochs.append(int(match[0]))
            train_loss.append(float(match[1]))
            val_loss.append(float(match[2]))
            val_accuracy.append(float(match[3]))

    return epochs, train_loss, val_loss, val_accuracy

def validate_data(data):
    return np.array(data, dtype=float)

def plot_loss(filename: str) -> None:
    epochs, train_loss, val_loss, val_accuracy = extract_data(filename)

    epochs = validate_data(epochs)
    train_loss = validate_data(train_loss)
    val_loss = validate_data(val_loss)
    val_accuracy = validate_data(val_accuracy)

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_loss, label='Train Loss', marker='o')

    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_accuracy(filename: str) -> None:
    epochs, train_loss, val_loss, val_accuracy = extract_data(filename)

    epochs = validate_data(epochs)
    train_loss = validate_data(train_loss)
    val_loss = validate_data(val_loss)
    val_accuracy = validate_data(val_accuracy)

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='^')
    
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_predictions():
    fifty_epoch_model = load_model("endgame_cnn50epochs.pth")
    ten_epoch_model = load_model("endgame_cnn.pth")
    tablebase = chess.syzygy.open_tablebase("D:/musta/Github Projects/NeuralBases/endgame_tablebase")
    fenGen = FenGen()
    
    test_FENs = [fenGen.make_fen() for _ in range(1000)]
    tensor_test_FENs = [fen_to_tensor(fen) for fen in test_FENs]

    tablebase_wdl = [tablebase.probe_wdl(chess.Board(fen)) for fen in test_FENs]

    ten_epoch_model_wdl = [predict(ten_epoch_model, fen) for fen in tensor_test_FENs]
    fifty_epoch_model_wdl = [predict(fifty_epoch_model, fen) for fen in tensor_test_FENs]

    ten_epoch_model_hits = [1 if x[0]==x[1] else 0 for x in zip(ten_epoch_model_wdl, tablebase_wdl)]
    fifty_epoch_model_hits = [1 if x[0]==x[1] else 0 for x in zip(fifty_epoch_model_wdl, tablebase_wdl)]

    ten_epoch_model_pracitcal_accuracy = sum(ten_epoch_model_hits) / len(ten_epoch_model_wdl)
    fifty_epoch_model_pracitcal_accuracy = sum(fifty_epoch_model_hits) / len(fifty_epoch_model_wdl)

    print(f"No. of tested FENs: {len(test_FENs)}")
    print(f"10 Epoch Accuracy: {ten_epoch_model_pracitcal_accuracy}")
    print(f"50 Epoch Accuracy: {fifty_epoch_model_pracitcal_accuracy}")


# try:
if __name__ == '__main__':
    compare_predictions()
    plot_loss("10EpochModel.txt")
    plot_loss("50EpochModel.txt")
    plot_accuracy("10EpochModel.txt")
    plot_accuracy("50EpochModel.txt")