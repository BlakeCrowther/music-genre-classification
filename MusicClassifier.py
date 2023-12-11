import torch.nn as nn

# Define a simple neural network
class MusicClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x