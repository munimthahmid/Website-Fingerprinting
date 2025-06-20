"""
Bonus Task 4: Advanced Training Script (Fast Models Only)
Contains advanced training functions with learning rate scheduling and early stopping.
Streamlined to train only the fastest models for quick results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import get_traces_with_labels
from train import (
    FingerprintClassifier,
    ComplexFingerprintClassifier,
    preprocess_data,
    FingerprintDataset,
)
from models.advanced_models import (
    AdvancedResNetClassifier,
    AttentionCNN,
)
from data_augmentation import augment_data


def train_advanced(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    model_save_path,
    use_scheduler=True,
    early_stopping_patience=10,
):
    """Advanced training with learning rate scheduling and early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # Learning rate scheduler
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
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

        train_accuracy = correct / total

        # Validation
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

        print(
            f"Epoch {epoch+1}/{epochs}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}"
        )

        # Learning rate scheduling
        if use_scheduler:
            scheduler.step(test_accuracy)

        # Save best model and early stopping
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with accuracy: {best_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_accuracy, train_accuracy


def evaluate_model(model, test_loader, website_names):
    """Evaluate a model and return detailed metrics."""
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

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Per-class accuracy
    class_accuracies = {}
    for i, name in enumerate(website_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(
                np.array(all_predictions)[class_mask]
                == np.array(all_labels)[class_mask]
            )
            class_accuracies[name] = class_acc

    return accuracy, class_accuracies


def train_fast_models():
    """Train only the fastest models for quick results."""
    print("=== Bonus Task 4: Fast Advanced Model Training ===")
    print("Goal: Beat the 74.6% accuracy with fast models only")
    print("Training: Advanced ResNet + AttentionCNN (Est. 10-15 min)")

    # Load data from database
    print("\n1. Loading data from database...")
    traces, labels, website_names = get_traces_with_labels("../webfingerprint.db")
    print(f"Loaded {len(traces)} traces from {len(website_names)} websites")

    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(traces, labels)

    # Apply data augmentation
    print("\n3. Applying data augmentation...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, augment_factor=1)
    print(f"Augmented training data: {len(X_train)} -> {len(X_train_aug)} traces")

    # Create datasets
    train_dataset = FingerprintDataset(X_train_aug, y_train_aug)
    test_dataset = FingerprintDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model parameters
    input_size = len(X_train[0])
    hidden_size = 128
    num_classes = len(website_names)

    print(
        f"Input size: {input_size}, Hidden size: {hidden_size}, Classes: {num_classes}"
    )

    # Initialize only fast models
    models_to_train = {
        "advanced_resnet": AdvancedResNetClassifier(
            input_size, hidden_size, num_classes
        ),
        "attention_cnn": AttentionCNN(input_size, hidden_size, num_classes),
    }

    # Train individual models
    print("\n4. Training fast models...")
    results = {}

    for model_name, model in models_to_train.items():
        print(f"\n--- Training {model_name} ---")

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        save_path = f"../saved_models/{model_name}_model.pth"

        # Train model
        best_acc, train_acc = train_advanced(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            epochs=50,
            model_save_path=save_path,
            use_scheduler=True,
            early_stopping_patience=10,
        )

        # Load best model and evaluate
        model.load_state_dict(torch.load(save_path))
        accuracy, class_accuracies = evaluate_model(model, test_loader, website_names)

        results[model_name] = {
            "accuracy": accuracy,
            "class_accuracies": class_accuracies,
        }

        print(f"{model_name} final accuracy: {accuracy:.4f}")
        for website, acc in class_accuracies.items():
            print(f"  {website}: {acc:.4f}")

    # Summary
    print("\n=== RESULTS SUMMARY ===")
    baseline_accuracy = 0.746  # 74.6% from Complex CNN
    print(f"Baseline (Complex CNN): {baseline_accuracy:.3f}")

    for model_name, result in results.items():
        acc = result["accuracy"]
        improvement = acc - baseline_accuracy
        print(f"{model_name:20}: {acc:.3f} ({improvement:+.3f})")

        if acc > baseline_accuracy:
            print(f"  ✅ BEATS BASELINE by {improvement:.3f}!")
        else:
            print(f"  ❌ Below baseline by {abs(improvement):.3f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(
        f"\nBest model: {best_model[0]} with {best_model[1]['accuracy']:.3f} accuracy"
    )

    if best_model[1]["accuracy"] > baseline_accuracy:
        print("🎉 SUCCESS: Bonus Task 4 completed - baseline beaten!")
    else:
        print("❌ Need more work to beat the baseline.")

    return results


if __name__ == "__main__":
    train_fast_models()
