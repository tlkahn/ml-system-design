# [[file:../README.org::*Architecture parallelism][Architecture parallelism:1]]
import torch
import torch.nn as nn
import torch.optim as optim


# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Save a checkpoint
torch.save(
    {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.1,
    },
    "checkpoint.pth",
)

# Load a checkpoint
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model.train()  # Set the model in train mode
# Continue training...
# Architecture parallelism:1 ends here
