# [[file:../README.org::*Pipeline parallelism][Pipeline parallelism:2]]
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from multiprocessing import Process, Pipe

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
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Linear(64, 10),
        )

    def forward(self, x, stage):
        if stage == 1:
            return self.stage1(x)
        elif stage == 2:
            return self.stage2(x)
        elif stage == 3:
            return self.stage3(x)
        elif stage == 4:
            return self.stage4(x)


# Pipeline stage function
def run_stage(stage, input_conn, output_conn, model, optimizer, criterion):
    while True:
        data, targets, stage_input = input_conn.recv()
        if data is None:
            break

        if stage != 1:
            data = stage_input

        stage_output = model(data, stage=stage)

        if stage != 4:
            output_conn.send((targets, stage_output))
        else:
            loss = criterion(stage_output, targets)
            loss.backward()

            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()


# Initialize the model, loss function, and optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create input and output connections for each stage
input_conns = [Pipe(False) for _ in range(4)]
output_conns = [Pipe(False) for _ in range(4)]

# Start pipeline stage processes
processes = []
for i in range(4):
    optimizer_stage = optimizer if i == 3 else None
    p = Process(
        target=run_stage,
        args=(
            i + 1,
            input_conns[i][1],
            output_conns[i][0],
            model,
            optimizer_stage,
            criterion,
        ),
    )
    p.start()
    processes.append(p)

# Training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data = data.view(-1, 28 * 28)

        optimizer.zero_grad()

        # Forward pass through the pipeline
        input_conns[0][0].send((data, targets, None))
        for j in range(1, 4):
            _, stage_output = output_conns[j - 1][1].recv()
            input_conns[j][0].send((None, None, stage_output))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Terminate the pipeline stage processes
for i in range(4):
    input_conns[i][0].send((None, None, None))
    processes[i].join()
# Pipeline parallelism:2 ends here
