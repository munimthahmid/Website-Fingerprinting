"""
Bonus Task 4: Data Augmentation Techniques
Contains functions for augmenting fingerprinting traces to improve model performance.
"""

import numpy as np
import random


def add_noise(trace, noise_factor=0.01):
    """Add Gaussian noise to trace."""
    noise = np.random.randn(*trace.shape) * noise_factor
    return trace + noise


def time_shift(trace, shift_range=50):
    """Randomly shift trace in time."""
    shift = random.randint(-shift_range, shift_range)
    if shift > 0:
        return np.concatenate([np.zeros(shift), trace[:-shift]])
    elif shift < 0:
        return np.concatenate([trace[-shift:], np.zeros(-shift)])
    return trace


def scale_amplitude(trace, scale_range=(0.8, 1.2)):
    """Randomly scale trace amplitude."""
    scale_factor = random.uniform(*scale_range)
    return trace * scale_factor


def add_dropout(trace, dropout_prob=0.05):
    """Randomly set some elements to zero (simulating measurement dropouts)."""
    mask = np.random.rand(*trace.shape) > dropout_prob
    return trace * mask.astype(np.float32)


def smooth_trace(trace, window_size=3):
    """Apply simple moving average smoothing."""
    if window_size < 2:
        return trace

    # Simple moving average
    smoothed = np.convolve(trace, np.ones(window_size) / window_size, mode="same")
    return smoothed


def augment_data(traces, labels, augment_factor=2):
    """Apply data augmentation to increase dataset size."""
    augmented_traces = []
    augmented_labels = []

    for trace, label in zip(traces, labels):
        # Original data
        augmented_traces.append(trace)
        augmented_labels.append(label)

        # Generate augmented versions
        for _ in range(augment_factor):
            aug_trace = trace.copy()

            # Apply random augmentations (with probability)
            if random.random() > 0.7:
                aug_trace = add_noise(
                    aug_trace, noise_factor=random.uniform(0.005, 0.02)
                )
            if random.random() > 0.7:
                aug_trace = time_shift(aug_trace, shift_range=random.randint(20, 80))
            if random.random() > 0.7:
                aug_trace = scale_amplitude(aug_trace, scale_range=(0.85, 1.15))
            if random.random() > 0.8:
                aug_trace = add_dropout(
                    aug_trace, dropout_prob=random.uniform(0.01, 0.05)
                )
            if random.random() > 0.9:
                aug_trace = smooth_trace(aug_trace, window_size=random.choice([3, 5]))

            augmented_traces.append(aug_trace)
            augmented_labels.append(label)

    return np.array(augmented_traces), np.array(augmented_labels)


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation technique."""
    batch_size = x.shape[0]

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
