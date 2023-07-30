# [[file:../README.org::*Pipeline parallelism][Pipeline parallelism:2]]
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
train_loader = DataLoader(
    dataset=train_dataset, batch_size=microbatch_size, shuffle=True
)


# Define the neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, stage):
        if stage == 1:
            return self.stage1(x)
        elif stage == 2:
            return self.stage2(x)


# Initialize the model, loss function, and optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data = data.view(-1, 28 * 28)

        optimizer.zero_grad()

        # Forward pass - Stage 1
        stage1_output = model(data, stage=1)
        stage1_output = stage1_output.detach().requires_grad_()

        # Forward pass - Stage 2
        scores = model(stage1_output, stage=2)
        loss = criterion(scores, targets)

        # Backward pass - Stage 2
        loss.backward()

        # Backward pass - Stage 1
        stage1_output_grad = stage1_output.grad
        stage1_output.backward(stage1_output_grad)

        # Accumulate gradients and perform an optimizer step
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
# Pipeline parallelism:2 ends here
