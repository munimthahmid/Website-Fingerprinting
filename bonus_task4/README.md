# Bonus Task 4: Advanced Website Fingerprinting Models

This folder contains advanced machine learning models and techniques to beat the 74.6% accuracy baseline of the Complex CNN model.

## Structure

```
bonus_task4/
├── models/
│   ├── __init__.py
│   └── advanced_models.py      # ResNet, Transformer, Attention CNN, Ensemble
├── data_augmentation.py        # Data augmentation techniques
├── train_advanced.py          # Advanced training script
└── README.md                  # This file
```

## Models Included

1. **AdvancedResNetClassifier**: ResNet-style architecture with skip connections, batch normalization, and attention mechanism
2. **TransformerClassifier**: Transformer-based model with positional encoding and multi-head attention
3. **AttentionCNN**: CNN with attention mechanism for better feature extraction
4. **EnsembleClassifier**: Combines multiple models for improved performance

## Data Augmentation Techniques

- **Noise injection**: Add Gaussian noise to traces
- **Time shifting**: Random temporal shifting of traces
- **Amplitude scaling**: Random scaling of trace amplitudes
- **Dropout simulation**: Random element dropout to simulate measurement errors
- **Smoothing**: Moving average smoothing

## Usage

To train the advanced models:

```bash
cd bonus_task4
python train_advanced.py
```

This will:

1. Load data from the database
2. Apply data augmentation
3. Train all advanced models
4. Create an ensemble model
5. Compare results against the 74.6% baseline

## Goal

Beat the 74.6% accuracy of the Complex CNN model through:

- Advanced architectures (ResNet, Transformer)
- Data augmentation
- Ensemble methods
- Learning rate scheduling
- Early stopping
