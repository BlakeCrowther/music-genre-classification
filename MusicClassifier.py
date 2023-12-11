import torch.nn as nn

# Define a simple neural network
class MusicClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation=nn.ReLU()):
        super(MusicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    