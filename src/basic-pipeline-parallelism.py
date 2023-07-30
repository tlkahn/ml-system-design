# [[file:../README.org::*Pipeline parallelism][Pipeline parallelism:1]]
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchgpipe import GPipe

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
batch_size = 100
microbatch_size = 25
accumulation_steps = batch_size // microbatch_size
learning_rate = 0.001
num_epochs = 5

# Load the MNIST dataset
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Split the SimpleNet model into two pipeline stages
model = nn.Sequential(
    nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)

# Wrap the model with GPipe
model = GPipe(
    model,
    balance=[2, 3],
    devices=[0, 1] if device == "cuda" else [0, 0],
    chunks=accumulation_steps,
)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data, targets in train_loader:
        # Move data and targets to the appropriate device
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        loss.backward()

        # Perform an optimizer step
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
# Pipeline parallelism:1 ends here
