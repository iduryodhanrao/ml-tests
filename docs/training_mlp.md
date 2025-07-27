Training a neural network using a **Multi-Layer Perceptron (MLP)** involves feeding data through layers of neurons, calculating errors, and updating weights so the model learns meaningful patterns. Let’s walk through the full lifecycle with a clear example.

---

## 🧱 1. **Architecture Setup**

Suppose we’re building an MLP to predict customer churn from tabular data with 10 input features. Our network looks like:

- **Input layer**: 10 neurons (for each feature)
- **Hidden layer 1**: 64 neurons + ReLU
- **Hidden layer 2**: 32 neurons + ReLU
- **Output layer**: 1 neuron + Sigmoid (for binary output)

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
```

---

## 🚦 2. **Forward Pass**

- Input data is fed into the network.
- Each layer computes:  
  $$ z = w \cdot x + b $$
- The result passes through activation functions (e.g. ReLU, Sigmoid).
- Output prediction is generated.

🎯 *Example*: If input is `[0.5, 1.2, ..., 0.9]` → output might be `0.78` → 78% chance of churn.

---

## 📉 3. **Loss Calculation**

Use a loss function to quantify prediction error:
```python
loss_fn = nn.BCELoss()  # For binary classification
loss = loss_fn(predictions, actual_labels)
```
This calculates how far off the prediction is from the true label.

---

## 🔁 4. **Backpropagation**

- Gradients of the loss w.r.t each weight are computed.
- Errors are propagated backward from output to input layer.
- Each weight gets updated to minimize the loss.

---

## 📈 5. **Optimization Step**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()    # Clear old gradients
loss.backward()          # Compute new gradients
optimizer.step()         # Apply weight updates
```

🎯 The model improves by adjusting weights to reduce the loss on training data.

---

## 🔄 6. **Repeat Across Epochs**

Repeat the process over multiple epochs:
- Shuffle the data
- Batch inputs
- Train, validate, and evaluate performance

---

## ✅ Example Use Case: Loan Default Prediction

- Input: Customer demographics, income, debt-to-income ratio
- Output: Probability of default
- Metric: Accuracy, Precision, Recall, AUC

You’ll train the MLP using historical data, then deploy it to flag risky applicants in real time.

---

A simple training run and visualize how gradients and loss evolve across epochs. This example uses PyTorch with synthetic data so we can see the core mechanics clearly.
---

## 📊 **Gradient & Loss Visualization with MLP**

Here’s a sketch of what we’ll do:
1. Simulate tabular data with 10 features.
2. Build a small MLP model.
3. Train over multiple epochs.
4. Visualize loss and gradient norms.

---

### 🧪 Step 1: Generate Synthetic Data

```python
import torch
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, n_classes=2)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---

### 🧱 Step 2: Define the MLP

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```

---

### 🧠 Step 3: Train & Track Gradients

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

loss_history = []
grad_history = []

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    grad_history.append(grad_norm)
    loss_history.append(loss.item())

    optimizer.step()
```

---

### 📈 Step 4: Plot Results

```python
import matplotlib.pyplot as plt

plt.plot(loss_history, label='Loss')
plt.plot(grad_history, label='Gradient Norm')
plt.xlabel('Epoch')
plt.title('Training Curve')
plt.legend()
plt.show()
```

---

## 🔍 What This Shows

- **Loss** curve tells us how well the model is minimizing error.
- **Gradient norm** highlights how strongly weights are being updated (can diagnose vanishing or exploding gradients).

You’ll typically see the loss drop early, then flatten as the model converges. 


