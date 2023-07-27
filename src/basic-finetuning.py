# [[file:../README.org::*Finetuning][Finetuning:1]]
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Load a pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Modify the model architecture for the new task
num_classes = 100  # Number of target classes
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# Load the target dataset
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder("path/to/train_data", transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4
)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

# Fine-tune the model
num_epochs = 10
resnet18.train()
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
# Finetuning:1 ends here
