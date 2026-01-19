import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Isotonic Layer Definition
# ---------------------------------------------------------
class IsotonicLayer(nn.Module):
    def __init__(self, units=1, lb=-17.0, ub=8.0, step=0.05, w_init_factor=0.5):
        super(IsotonicLayer, self).__init__()
        self.units = units
        self.lb = lb
        self.ub = ub
        self.step = step
        self.num_buckets = int((ub - lb) / step) + 1
        self.residue = lb - step

        self.v = nn.Parameter(torch.ones(units, self.num_buckets) * w_init_factor)
        self.b = nn.Parameter(torch.zeros(units))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1).expand(-1, self.units)

        batch_size = x.shape[0]
        device = x.device

        x_clipped = torch.clamp(x, self.lb + 1e-9, self.ub - 1e-9)
        indx = ((x_clipped - self.lb + self.step) / self.step).long()
        indx = torch.clamp(indx, 0, self.num_buckets - 1)

        range_vec = torch.arange(self.num_buckets, device=device).view(1, 1, -1)
        expand_indx = indx.unsqueeze(2)
        
        activation_vector = torch.where(range_vec < expand_indx, 
                                        torch.tensor(self.step, device=device), 
                                        torch.tensor(0.0, device=device))

        delta = (x_clipped - self.lb + self.step) - (indx.float() * self.step)
        final_activation = activation_vector.clone()
        final_activation.scatter_(2, expand_indx, delta.unsqueeze(2))

        weights = F.relu(self.v)
        logits = torch.sum(final_activation * weights, dim=2) + self.residue + self.b
        
        return torch.sigmoid(logits)

# ---------------------------------------------------------
# 2. Train and Visualize
# ---------------------------------------------------------
def train_and_plot():
    torch.manual_seed(42)
    
    # Configuration
    model = IsotonicLayer(units=1, lb=-17.0, ub=8.0, step=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()
    
    # Generate Training Data
    x_train = torch.rand(1000).unsqueeze(1)
    y_train = torch.where(x_train < 0.95, x_train ** 2, (1.9 - x_train) ** 2)  # Ground Truth
    
    # Train
    print("Training model...")
    for epoch in range(1000):
        optimizer.zero_grad()
        x_input = torch.logit(x_train)
        preds = model(x_input)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
    
    # ---------------------------------------------------------
    # Visualization Logic
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Create a smooth range of inputs for plotting
        # Flattened for plotting [0, 0.01, 0.02, ... 1.0]
        x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
        x_input = torch.logit(x_plot)
        # Get Model Predictions
        y_pred = model(x_input)
        
        # Get Ground Truth for this range
        y_gt = torch.where(x_plot < 0.95, x_plot ** 2, (1.9 - x_plot) ** 2)
        
        # Convert to numpy for matplotlib
        x_np = x_plot.squeeze().numpy()
        y_pred_np = y_pred.squeeze().numpy()
        y_gt_np = y_gt.squeeze().numpy()
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Ground Truth (Green dashed line)
        plt.plot(x_np, y_gt_np, 'g--', label='Ground Truth (Non Monotonic Curve)', linewidth=2)
        
        # 2. Plot Model Prediction (Blue line)
        plt.plot(x_np, y_pred_np, 'b-', label='Isotonic Layer Prediction', linewidth=2, alpha=0.8)
        
        # 3. Scatter plot of a subset of training data (Red dots)
        # Taking a small random sample to keep graph clean
        indices = torch.randperm(len(x_train))[:500]
        plt.scatter(x_train[indices], y_train[indices], color='red', alpha=0.5, label='Training Samples', s=20)

        plt.title("Isotonic Layer Fitting An Non Monotonic Curve")
        plt.xlabel("Input x")
        plt.ylabel("Output y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Show plot
        plt.show()

if __name__ == "__main__":
    train_and_plot()