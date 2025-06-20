import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import database

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
INPUT_SIZE = 1000
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintDataset(Dataset):
    """Dataset class for website fingerprinting traces."""

    def __init__(self, traces, labels, scaler=None, fit_scaler=False):
        """
        Args:
            traces: List or array of trace data
            labels: List or array of labels
            scaler: Optional StandardScaler for normalization
            fit_scaler: Whether to fit the scaler on this data
        """
        self.traces = np.array(traces, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)

        if scaler is not None:
            if fit_scaler:
                self.traces = scaler.fit_transform(self.traces)
            else:
                self.traces = scaler.transform(self.traces)

        self.traces = torch.FloatTensor(self.traces)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


def load_data_from_json(json_path):
    """Load data from JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        traces = []
        labels = []
        websites = list(data.keys())

        for i, website in enumerate(websites):
            website_traces = data[website]
            traces.extend(website_traces)
            labels.extend([i] * len(website_traces))

        return traces, labels, websites

    except FileNotFoundError:
        print(f"JSON file {json_path} not found. Please run data collection first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None, None, None


def load_data_from_database(db_path="webfingerprint.db"):
    """Load data from SQLite database."""
    try:
        traces, labels, websites = database.get_traces_with_labels(db_path)
        print(f"Loaded {len(traces)} traces from {len(websites)} websites: {websites}")
        return traces, labels, websites
    except Exception as e:
        print(f"Error loading database: {e}")
        return None, None, None


def preprocess_data(traces, labels, test_size=0.2, normalize=True):
    """Preprocess the traces and labels."""
    # Convert labels to numpy array
    y = np.array(labels, dtype=np.int64)

    # Ensure all traces have the same length
    max_len = max(len(trace) for trace in traces)
    padded_traces = []
    for trace in traces:
        # Convert trace to numpy array first
        trace_array = np.array(trace, dtype=np.float32)
        if len(trace_array) < max_len:
            # Pad with zeros
            padded_trace = np.pad(
                trace_array, (0, max_len - len(trace_array)), "constant"
            )
        else:
            # Truncate if too long
            padded_trace = trace_array[:max_len]
        padded_traces.append(padded_trace)

    X = np.array(padded_traces, dtype=np.float32)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


class FingerprintClassifier(nn.Module):
    """Basic CNN classifier for website fingerprinting."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool1d(2)

        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After 3 pooling operations

        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ComplexFingerprintClassifier(nn.Module):
    """More complex CNN classifier with additional layers and techniques."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        # Pooling layers
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # Convolutional layers with batch norm, ReLU and pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))

        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train(
    model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path
):
    """Train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}"
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), model_save_path)

    return train_losses, train_accuracies, test_accuracies, best_accuracy


def evaluate(model, test_loader, website_names):
    """Evaluate the model and generate detailed report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(all_labels, all_predictions, target_names=website_names)
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    return accuracy


def main():
    """Main training function."""
    print("Website Fingerprinting - Training Models")
    print("=" * 50)

    # Try to load data from database first, then JSON
    traces, labels, websites = load_data_from_database()

    if traces is None:
        print("Database not found, trying JSON file...")
        traces, labels, websites = load_data_from_json(DATASET_PATH)

    if traces is None:
        print("No data found. Please run data collection first.")
        return

    print(f"Loaded {len(traces)} traces from {len(websites)} websites")

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(traces, labels)

    # Create datasets and data loaders
    train_dataset = FingerprintDataset(X_train, y_train)
    test_dataset = FingerprintDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model parameters
    input_size = X_train.shape[1]
    num_classes = len(websites)

    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Websites: {websites}")

    # Train Basic CNN
    print("\n" + "=" * 50)
    print("Training Basic CNN Model")
    print("=" * 50)

    basic_model = FingerprintClassifier(input_size, HIDDEN_SIZE, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(basic_model.parameters(), lr=LEARNING_RATE)

    basic_model_path = os.path.join(MODELS_DIR, "basic_cnn_model.pth")
    basic_losses, basic_train_acc, basic_test_acc, basic_best_acc = train(
        basic_model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        EPOCHS,
        basic_model_path,
    )

    print(f"Basic CNN Best Accuracy: {basic_best_acc:.4f}")

    # Load best basic model and evaluate
    basic_model.load_state_dict(torch.load(basic_model_path))
    print("\nBasic CNN Evaluation:")
    basic_accuracy = evaluate(basic_model, test_loader, websites)

    # Train Complex CNN
    print("\n" + "=" * 50)
    print("Training Complex CNN Model")
    print("=" * 50)

    complex_model = ComplexFingerprintClassifier(input_size, HIDDEN_SIZE, num_classes)
    optimizer = optim.Adam(complex_model.parameters(), lr=LEARNING_RATE)

    complex_model_path = os.path.join(MODELS_DIR, "complex_cnn_model.pth")
    complex_losses, complex_train_acc, complex_test_acc, complex_best_acc = train(
        complex_model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        EPOCHS,
        complex_model_path,
    )

    print(f"Complex CNN Best Accuracy: {complex_best_acc:.4f}")

    # Load best complex model and evaluate
    complex_model.load_state_dict(torch.load(complex_model_path))
    print("\nComplex CNN Evaluation:")
    complex_accuracy = evaluate(complex_model, test_loader, websites)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Basic CNN Accuracy:   {basic_accuracy:.4f}")
    print(f"Complex CNN Accuracy: {complex_accuracy:.4f}")

    better_model = "Complex CNN" if complex_accuracy > basic_accuracy else "Basic CNN"
    print(f"Better model: {better_model}")

    # Save the better model as the main model
    if complex_accuracy > basic_accuracy:
        torch.save(complex_model.state_dict(), "model.pth")
        print("Complex CNN saved as main model (model.pth)")
    else:
        torch.save(basic_model.state_dict(), "model.pth")
        print("Basic CNN saved as main model (model.pth)")


if __name__ == "__main__":
    main()
