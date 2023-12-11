import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


# Define a PyTorch dataset with padding
class MusicDataset(Dataset):
    def __init__(self, features, labels, genre_to_index, max_length, num_classes):
        self.features = [self.pad_feature(feature, max_length) for feature in features]
        self.labels = torch.tensor([int(genre_to_index[label]) for label in labels], dtype=torch.long)
        self.labels_one_hot = F.one_hot(self.labels, num_classes=num_classes).long()  # One-hot encode

        
        # for idx, feature in enumerate(self.features):
        #     print(f'Original shape of feature {idx}: {features[idx].shape}')
        #     print(f"Padded shape of feature {idx}: {feature.shape}")
        #     print(f'labels one hot {idx}: {self.labels_one_hot[idx]}')
        #     print(f'Labels original {idx}: {self.labels[idx]}')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels_one_hot[idx]

    def pad_feature(self, feature, max_length):
        # Truncate or zero-pad the sequence to the specified max_length
        if feature.shape[1] > max_length:
            start_index = max(0, (feature.shape[1] - max_length) // 2)
            feature = feature[:, start_index:start_index + max_length]
        elif feature.shape[1] < max_length:
            # Zero-pad if the sequence is shorter than max_length
            pad_width = max_length - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        return feature
