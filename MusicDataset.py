import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

# Define a PyTorch dataset with padding
class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label