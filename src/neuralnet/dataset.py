import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
from fenGenerator import FenGen
from fenParser import create_nn_input


class EndgamesDataset(Dataset):

    def __init__(self, X, y):
        """
        Defines the Tensors for the data

        Parameters:
        X: array for the inputs, that will be stored as  pytorch FloatTensor
        y: array of the outputs for each value in X, that will be stored as pytorch LongTensor

        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y+2)

    def __len__(self) -> int:
        """
        Returns the lenght of the y tensor (which should be the same as the length of X)
        """
        return len(self.y)
    
    def __getitem__(self, index: int) -> tuple[list[int], int]:
        """
        Returns the i-th element of X and y as a tuple
        """
        return self.X[index], self.y[index]
    
def process_batch(_, batch_size, fenGen):
    fens = [fenGen.make_fen() for _ in range(batch_size)]
    return create_nn_input(fens)

def create_data():
    """
    Creates 1000000 sample endgame positions and their respective WDL. X contains the matrix board representation of an endgame FEN, while y contains the respective WDL evaluation.

    Both X and y are saved as .npy files named "X_data.npy" and "y_data.npy" respectively.
    """
    total_fens = 1000000
    batch_size = 1000
    num_batches = total_fens // batch_size

    # Define file paths for saving/loading
    x_file = 'X_data.npy'
    y_file = 'y_data.npy'

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

    
def load_data(x_path, y_path):
    """
    Loads the saved .npy into np arrays and returns them

    Parameters:
        x_path: path to X_data.npy file
        y_path: path to y_data.npy file

    Returns:
    NP Arrays of X and y as tuple
    """
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y