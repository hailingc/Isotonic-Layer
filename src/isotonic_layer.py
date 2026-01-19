"""
Isotonic Layer Module

This module implements an Isotonic Layer for neural networks, which produces
monotonically non-decreasing outputs with respect to its inputs. The layer uses
a bucketing approach to approximate isotonic regression, making it differentiable
and suitable for end-to-end training with gradient descent.

Key Features:
    - Enforces monotonicity through ReLU-constrained weights
    - Supports multiple parallel isotonic units
    - Configurable input range and bucket granularity
    - Outputs probabilities via sigmoid activation

Example:
    >>> layer = IsotonicLayer(units=1, lower_bound=-10.0, upper_bound=10.0)
    >>> x = torch.randn(32, 1)  # Batch of 32 inputs
    >>> y = layer(x)  # Monotonic outputs in (0, 1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class IsotonicLayer(nn.Module):
    """
    A neural network layer that produces monotonically non-decreasing outputs.
    
    The IsotonicLayer discretizes the input space into buckets and learns a 
    non-negative weight for each bucket. By constraining weights to be non-negative
    (via ReLU) and accumulating contributions from lower buckets, the layer 
    guarantees monotonicity in its output.
    
    Architecture:
        1. Input is clipped to [lower_bound, upper_bound]
        2. Input is mapped to bucket indices based on bucket_width
        3. An activation vector accumulates full bucket_width for lower buckets
           and a fractional amount for the current bucket
        4. The activation vector is multiplied by non-negative weights
        5. A sigmoid function produces the final probability output
    
    Attributes:
        units (int): Number of parallel isotonic units (output dimensions).
        lower_bound (float): Minimum value of the input range.
        upper_bound (float): Maximum value of the input range.
        bucket_width (float): Width of each discretization bucket.
        num_buckets (int): Total number of buckets spanning the input range.
        weights (nn.Parameter): Learnable non-negative weights for each bucket.
        bias (nn.Parameter): Learnable bias term for each unit.
    
    Args:
        units (int, optional): Number of parallel isotonic outputs. Default: 1.
        lower_bound (float, optional): Lower bound of the input range. Default: -17.0.
        upper_bound (float, optional): Upper bound of the input range. Default: 8.0.
        bucket_width (float, optional): Width of each bucket for discretization.
            Smaller values provide finer granularity but increase computation.
            Default: 0.05.
        weight_init_factor (float, optional): Initial value for all bucket weights.
            Default: 0.5.
    
    Example:
        >>> # Create an isotonic layer for probability calibration
        >>> layer = IsotonicLayer(
        ...     units=1,
        ...     lower_bound=-5.0,
        ...     upper_bound=5.0,
        ...     bucket_width=0.1,
        ...     weight_init_factor=0.3
        ... )
        >>> logits = torch.randn(64, 1)  # Raw model logits
        >>> calibrated_probs = layer(logits)  # Calibrated, monotonic probabilities
    """
    
    def __init__(
        self,
        units: int = 1,
        lower_bound: float = -17.0,
        upper_bound: float = 8.0,
        bucket_width: float = 0.05,
        weight_init_factor: float = 0.5
    ):
        """Initialize the IsotonicLayer with the specified configuration."""
        super(IsotonicLayer, self).__init__()
        
        self.units = units
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bucket_width = bucket_width
        self.num_buckets = int((upper_bound - lower_bound) / bucket_width) + 1
        
        # Residue offset ensures proper alignment at the lower boundary
        self.residue = lower_bound - bucket_width
        
        # Learnable parameters
        # weights: Non-negative (after ReLU) contribution of each bucket
        self.weights = nn.Parameter(
            torch.ones(units, self.num_buckets) * weight_init_factor
        )
        # bias: Additive bias for each output unit
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the monotonic output for the given input tensor.
        
        The forward pass performs the following steps:
            1. Expands 1D input to match the number of units
            2. Clips input values to the valid range [lower_bound, upper_bound]
            3. Computes bucket indices for each input
            4. Constructs activation vectors that accumulate bucket contributions
            5. Applies non-negative weights and computes the weighted sum
            6. Returns sigmoid of the result for probability output
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,) or 
                (batch_size, units). Values outside [lower_bound, upper_bound]
                are clipped to the boundary values.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, units) with values
                in the range (0, 1). Outputs are guaranteed to be monotonically
                non-decreasing with respect to inputs.
        
        Note:
            The monotonicity guarantee holds because:
            - Bucket weights are constrained to be non-negative via ReLU
            - Higher input values activate more buckets
            - Each additional bucket adds a non-negative contribution
        """
        # Handle 1D input by expanding to (batch_size, units)
        if x.dim() == 1:
            x = x.unsqueeze(1).expand(-1, self.units)
        
        # batch_size = x.shape[0]
        device = x.device
        
        # Clip input to valid range with small epsilon to avoid boundary issues
        x_clipped = torch.clamp(
            x,
            self.lower_bound + 1e-9,
            self.upper_bound - 1e-9
        )
        
        # Compute bucket indices for each input value
        bucket_indices = (
            (x_clipped - self.lower_bound + self.bucket_width) / self.bucket_width
        ).long()
        bucket_indices = torch.clamp(bucket_indices, 0, self.num_buckets - 1)
        
        # Create range vector for vectorized comparison
        # Shape: (1, 1, num_buckets)
        bucket_range = torch.arange(
            self.num_buckets, device=device
        ).view(1, 1, -1)
        
        # Expand indices for broadcasting
        # Shape: (batch_size, units, 1)
        expanded_indices = bucket_indices.unsqueeze(2)
        
        # Build activation vector:
        # - Full bucket_width for all buckets below the current bucket
        # - Zero for buckets at or above the current bucket (will be filled with delta)
        activation_vector = torch.where(
            bucket_range < expanded_indices,
            torch.tensor(self.bucket_width, device=device),
            torch.tensor(0.0, device=device)
        )
        
        # Compute fractional activation for the current bucket
        # This is the distance from the bucket's lower edge to the input value
        delta = (
            (x_clipped - self.lower_bound + self.bucket_width) 
            - (bucket_indices.float() * self.bucket_width)
        )
        
        # Insert the fractional activation at the current bucket position
        final_activation = activation_vector.clone()
        final_activation.scatter_(2, expanded_indices, delta.unsqueeze(2))
        
        # Apply non-negative constraint to weights via ReLU
        non_negative_weights = F.relu(self.weights)
        
        # Compute weighted sum of activations plus residue and bias
        logits = (
            torch.sum(final_activation * non_negative_weights, dim=2) 
            + self.residue 
            + self.bias
        )
        
        # Apply sigmoid to produce probability output
        return torch.sigmoid(logits)


def train_and_plot():
    """
    Demonstrate the IsotonicLayer by fitting it to a non-monotonic target function.
    
    This function:
        1. Creates an IsotonicLayer model
        2. Generates synthetic training data with a non-monotonic ground truth
        3. Trains the model using binary cross-entropy loss
        4. Visualizes the learned monotonic approximation vs ground truth
    
    The demonstration shows how the isotonic layer approximates a non-monotonic
    function (a parabola that peaks and descends) with the best possible
    monotonically non-decreasing function.
    
    The ground truth function is:
        y = x² for x < 0.95
        y = (1.9 - x)² for x >= 0.95
    
    This creates a curve that rises, peaks near x=0.95, then falls.
    The isotonic layer learns to approximate this with a monotonic function
    that rises and then plateaus (since it cannot decrease).
    """
    torch.manual_seed(42)
    
    # Model Configuration
    model = IsotonicLayer(
        units=1,
        lower_bound=-17.0,
        upper_bound=8.0,
        bucket_width=0.5
    )
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()
    
    # Generate Synthetic Training Data
    # x_train: Random values in [0, 1)
    # y_train: Non-monotonic parabolic function
    x_train = torch.rand(1000).unsqueeze(1)
    y_train = torch.where(
        x_train < 0.95,
        x_train ** 2,
        (1.9 - x_train) ** 2
    )
    
    # Training Loop
    print("Training model...")
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Transform x from [0,1] to unbounded space via logit
        x_input = torch.logit(x_train)
        
        # Forward pass
        predictions = model(x_input)
        
        # Compute loss and backpropagate
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    
    # Visualization
    model.eval()
    with torch.no_grad():
        # Create smooth input range for plotting
        x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
        x_input = torch.logit(x_plot)
        
        # Get model predictions
        y_pred = model(x_input)
        
        # Compute ground truth for the plotting range
        y_ground_truth = torch.where(
            x_plot < 0.95,
            x_plot ** 2,
            (1.9 - x_plot) ** 2
        )
        
        # Convert tensors to numpy arrays for matplotlib
        x_numpy = x_plot.squeeze().numpy()
        y_pred_numpy = y_pred.squeeze().numpy()
        y_ground_truth_numpy = y_ground_truth.squeeze().numpy()
        
        # Create the visualization
        plt.figure(figsize=(10, 6))
        
        # Plot ground truth (non-monotonic target)
        plt.plot(
            x_numpy,
            y_ground_truth_numpy,
            'g--',
            label='Ground Truth (Non-Monotonic Curve)',
            linewidth=2
        )
        
        # Plot model prediction (monotonic approximation)
        plt.plot(
            x_numpy,
            y_pred_numpy,
            'b-',
            label='Isotonic Layer Prediction',
            linewidth=2,
            alpha=0.8
        )
        
        # Scatter plot of training samples
        sample_indices = torch.randperm(len(x_train))[:500]
        plt.scatter(
            x_train[sample_indices],
            y_train[sample_indices],
            color='red',
            alpha=0.5,
            label='Training Samples',
            s=20
        )
        
        plt.title("Isotonic Layer Fitting a Non-Monotonic Curve")
        plt.xlabel("Input x")
        plt.ylabel("Output y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()


if __name__ == "__main__":
    train_and_plot()
