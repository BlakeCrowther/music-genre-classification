def get_splits(size):
    set_size_mask = tracks['set', 'subset'] <= size
    
    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'
    
    X_train = features.loc[set_size_mask & train, 'mfcc']
    X_test = features.loc[set_size_mask & test, 'mfcc']
    y_train = tracks.loc[set_size_mask & train, ('track', 'genre_top')]
    y_test = tracks.loc[set_size_mask & test, ('track', 'genre_top')]
    
    return X_train, X_test, y_train, y_test

def preprocess_splits(X_train, X_test, y_train, y_test):
    """
    Preprocesses the training and test data. This includes scaling the features and encoding the labels.
    
    Parameters:
    X_train, X_test: Feature data for training and test sets.
    y_train, y_test: Label data for training and test sets.

    Returns:
    X_train_tensor, X_test_tensor: Scaled and tensor-converted feature data for training and test sets.
    y_train_tensor, y_test_tensor: Encoded and tensor-converted label data for training and test sets.
    """
    # Fit scaler then transform X_train and X_test
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    # Fit the encoder and transform the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert numpy arrays to tensors
    X_train_tensor = torch.tensor(X_scaled_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_scaled_test, dtype=torch.float)

    # Convert the encoded labels to PyTorch tensors
    y_train_tensor = torch.tensor(y_train_encoded)
    y_test_tensor = torch.tensor(y_test_encoded)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def get_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=32):
    """
    Creates data loaders for the training and test sets.

    Parameters:
    X_train_tensor, X_test_tensor: Tensor-converted feature data for training and test sets.
    y_train_tensor, y_test_tensor: Tensor-converted label data for training and test sets.
    batch_size: The number of samples per batch.

    Returns:
    train_loader, test_loader: Data loaders for the training and test sets.
    """
    # Create training and test sets
    train_set = MusicDataset(X_train_tensor, y_train_tensor)
    test_set = MusicDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_model(input_size, hidden_size, num_classes):
    """
    Creates a new instance of the MusicClassifier model.

    Parameters:
    input_size: The number of input features.
    hidden_size: The number of neurons in the hidden layer.
    num_classes: The number of output classes.

    Returns:
    model: A new instance of the MusicClassifier model.
    """
    model = MusicClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains the model for a specified number of epochs.

    Parameters:
    model: The model to train.
    train_loader: The data loader for the training data.
    criterion: The loss function.
    optimizer: The optimization algorithm.
    num_epochs: The number of epochs to train for.

    Returns:
    model: The trained model.
    """
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    return model

def evaluate_model(model, dataloader, label_encoder):
    """
    Evaluates the model's performance on the provided data.

    Parameters:
    model: The model to evaluate.
    dataloader: The data loader for the data to evaluate on.
    label_encoder: The label encoder used to decode the labels.

    Returns:
    accuracy: The model's accuracy on the provided data.
    results: A DataFrame containing the true and predicted labels for each batch.
    """
    model.eval()
    # Initialize counters for total number of labels and correct predictions
    total = 0
    correct = 0
    # Initialize a DataFrame to store the true and predicted labels for each batch
    results = pd.DataFrame(columns=['Index', 'True Labels', 'Predicted Labels'])
    # Iterate over the data loader
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # Decode the true and predicted labels
        decoded_true = label_encoder.inverse_transform(labels)
        decoded_predicted = label_encoder.inverse_transform(predicted)
        # Append batch results to DataFrame
        results.loc[i] = [i, decoded_true, decoded_predicted]
        # Update the total number of labels and the number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Compute the accuracy of the model    
    accuracy = correct / total
    
    return accuracy, results