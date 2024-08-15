import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EndgamesDataset, load_data, create_data
from model import EndgameModel
from fenParser import board_to_matrix
import os
import logging

# TODO: Add type hinting and formatting to model train and dataset files
# TODO: Add option to insert FEN from terminal

def train(model, train_loader, criterion, optimiser, device) -> int:
    """
    Trains an instance of the EndgameModel using the given training data and the given criterion, and optimiser.

    Parameters:
        model: Instance of EndgameModel
        train_loader: DataLoader for training data
        criterion: Loss function
        optimiser: Any available optimiser from torch.optim
        device: Device on which the model will be trained. Can be "cpu" or "cuda" depending on the available resources

    Returns:
        train_loss: Training loss of one training iteration
    """
    model.train()
    running_loss = 0.0
    for i , (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)
        logging.debug(f"Batch {i}: input shape: {inputs.shape}, labels shape: {labels.shape}, outputs shape: {outputs.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device) -> int:
    """
    Evaluates the trained model using the validation data using the same criterion, and optimiser.

    Parameters:
        model: The trained model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device on which the model will be trained. Can be "cpu" or "cuda" depending on the available resources

    Returns:
        val_loss: Validation loss of one validation iteration
    """
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

def main():
    """
    Trains, and validates the model with the respective data. Outputs the training and validation loss for each epoch alongside the validation accuracy
    """
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50

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

    torch.save(model.state_dict(), 'endgame_cnn50EPOCHS.pth')


if __name__ == "__main__":
    main()