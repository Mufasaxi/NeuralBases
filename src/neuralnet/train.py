import multiprocessing
from functools import partial
from tqdm import tqdm
import time
import numpy as np
import os
from fenParser import create_nn_input
from fenGenerator import FenGen

from dataset import EndgamesDataset
from model import EndgameModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def process_batch(_, batch_size, fenGen):
    fens = [fenGen.make_fen() for _ in range(batch_size)]
    return create_nn_input(fens)

def create_np_inputs():
    total_fens = 1000000
    batch_size = 1000
    num_batches = total_fens // batch_size

    # Define file paths for saving/loading
    x_file = 'X_data.npy'
    y_file = 'y_data.npy'

    # Check if saved files already exist
    if os.path.exists(x_file) and os.path.exists(y_file):
        print("Loading existing data...")
        X = np.load(x_file)
        y = np.load(y_file)
    else:
        print("Generating new data...")
        fenGen = FenGen()

        # Create a partial function with the batch size and fenGen
        process_batch_partial = partial(process_batch, batch_size=batch_size, fenGen=fenGen)

        # Use multiprocessing to parallelize the work
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(process_batch_partial, range(num_batches)), total=num_batches))

        # Combine the results
        X = np.concatenate([r[0] for r in results])
        y = np.concatenate([r[1] for r in results])

        # Save the results
        print("Saving data...")
        np.save(x_file, X)
        np.save(y_file, y)

    print(f"Total FENs processed: {len(X)}")
    return X, y

def train(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = EndgamesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    print(f"Using device: {device}")

    model = EndgameModel(5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = 0.0001)


    epochs = 50
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs= inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimiser.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.util.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            running_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(f'Epoch {epoch + 1 + 50}/{epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')

        torch.save(model.state_dict(), "../../models/TORCH50EPOCHS.pth")
        

def main():
    X, y = create_np_inputs()
    train(X, y)


if __name__ == "__main__":
    main()