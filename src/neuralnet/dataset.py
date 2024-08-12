from torch.utils.data import Dataset

class EndgamesDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.X[index], self.y[index]