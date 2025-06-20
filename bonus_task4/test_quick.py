"""
Quick test script for advanced models - reduced data and epochs for fast local testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import get_traces_with_labels
from train import preprocess_data, FingerprintDataset, ComplexFingerprintClassifier
from models.advanced_models import (
    AdvancedResNetClassifier,
    TransformerClassifier,
    AttentionCNN,
)


def quick_test():
    print("=== QUICK TEST: Advanced Models ===")
    print("Testing with reduced data and epochs for speed")

    # Load data
    traces, labels, website_names = get_traces_with_labels("../webfingerprint.db")
    print(f"Loaded {len(traces)} traces from {len(website_names)} websites")

    # Use only subset of data for quick test
    subset_size = min(300, len(traces))  # Use only 300 traces max
    traces = traces[:subset_size]
    labels = labels[:subset_size]
    print(f"Using subset: {subset_size} traces for quick test")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(traces, labels)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create datasets
    train_dataset = FingerprintDataset(X_train, y_train)
    test_dataset = FingerprintDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model parameters
    input_size = len(X_train[0])
    hidden_size = 64  # Reduced for speed
    num_classes = len(website_names)

    # Test models (reduced complexity)
    models_to_test = {
        "baseline_complex": ComplexFingerprintClassifier(
            input_size, hidden_size, num_classes
        ),
        "advanced_resnet": AdvancedResNetClassifier(
            input_size, hidden_size, num_classes
        ),
        "attention_cnn": AttentionCNN(input_size, hidden_size, num_classes),
        # Skip transformer for quick test (most complex)
    }

    results = {}
    baseline_acc = 0

    for model_name, model in models_to_test.items():
        print(f"\n--- Quick Training: {model_name} ---")

        # Quick training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for just 5 epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on: {device}")

        for epoch in range(5):  # Very few epochs
            model.train()
            for traces_batch, labels_batch in train_loader:
                traces_batch, labels_batch = traces_batch.to(device), labels_batch.to(
                    device
                )

                optimizer.zero_grad()
                outputs = model(traces_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

        # Quick evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for traces_batch, labels_batch in test_loader:
                traces_batch, labels_batch = traces_batch.to(device), labels_batch.to(
                    device
                )
                outputs = model(traces_batch)
                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        accuracy = correct / total
        results[model_name] = accuracy

        if model_name == "baseline_complex":
            baseline_acc = accuracy

        print(f"{model_name}: {accuracy:.3f} accuracy")

    # Summary
    print(f"\n=== QUICK TEST RESULTS ===")
    print(f"Baseline (Complex): {baseline_acc:.3f}")

    for model_name, acc in results.items():
        if model_name != "baseline_complex":
            improvement = acc - baseline_acc
            status = "✅ BETTER" if improvement > 0 else "❌ WORSE"
            print(f"{model_name:20}: {acc:.3f} ({improvement:+.3f}) {status}")

    print(f"\n✅ Models are working! Ready for full training.")
    print(f"💡 For full training: Use Google Colab GPU or reduce epochs/data locally")


if __name__ == "__main__":
    quick_test()
