import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EndgamesDataset, load_data, create_data
from model import EndgameModel
from fenParser import board_to_matrix
import chess
import os

# TODO: Add type hinting and formatting to model train and dataset files
# TODO: Add option to insert FEN from terminal

def train(model, train_loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    for i , (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)

        # print(f"Batch {i}: input shape: {inputs.shape}, labels shape: {labels.shape}, outputs shape: {outputs.shape}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return running_loss / len(val_loader), accuracy

def load_model(model_path):
    model = EndgameModel()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def fen_to_tensor(fen):
    board_matrix = board_to_matrix(chess.Board(fen))
    input_tensor = torch.FloatTensor(board_matrix).unsqueeze(0)
    return input_tensor

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
    return predicted.item() - 2


def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    if os.path.exists("X_data.npy") and os.path.exists("y_data.npy"):
        X, y = load_data("X_data.npy", "y_data.npy")
    else: 
        create_data()
        X, y = load_data("X_data.npy", "y_data.npy")

    split = int(0.8 * len(y))
    train_dataset = EndgamesDataset(X[:split], y[:split])
    val_dataset = EndgamesDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndgameModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimiser, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss: .4f}")
        print(f"Val Loss: {val_loss: .4f}, Val Accuracy: {val_accuracy: .4f}")

    torch.save(model.state_dict(), 'endgame_cnn.pth')



if __name__ == "__main__":
    model = load_model("endgame_cnn.pth")
    fen_string = "2K5/4q3/8/3B4/k7/8/8/8 w - - 0 1"
    input_tensor = fen_to_tensor(fen_string)
    prediction = predict(model, input_tensor)

    print(f"Input FEN: 2K5/4q3/8/3B4/k7/8/8/8 w - - 0 1")
    print(f"Predicted WDL: {prediction}")