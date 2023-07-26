# [[file:../README.org::*CBOW][CBOW:2]]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample input data
data = [
    ["hello", "world"],
    ["goodbye", "world"],
    ["hello", "goodbye"],
    ["world", "hello"],
]

# Vocabulary
vocab = list(set([word for sentence in data for word in sentence]))
vocab_size = len(vocab)

# Word-to-index mapping
word_to_index = {word: i for i, word in enumerate(vocab)}

# Context window size
window_size = 2

# Generate training data
training_data = []
for sentence in data:
    for i, target_word in enumerate(sentence):
        context_words = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(sentence):
                context_words.append(word_to_index[sentence[j]])
                training_data.append((context_words, word_to_index[target_word]))


class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, target = self.data[index]
        return torch.tensor(context), torch.tensor(target)


# CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x).sum(dim=1)
        hidden = torch.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output


# Training parameters
embedding_dim = 10
hidden_dim = 10
epochs = 100
batch_size = 64
learning_rate = 0.1

# Create CBOW model instance
model = CBOW(vocab_size, embedding_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Create DataLoader for training data
train_dataset = CBOWDataset(training_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0

    for context, target in train_loader:
        optimizer.zero_grad()

        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")

# Test the model
test_sentence = ["hello", "world"]
context = []
for i, target_word in enumerate(test_sentence):
    context_words = []
    for j in range(i - window_size, i + window_size + 1):
        if j != i and 0 <= j < len(test_sentence):
            context_words.append(word_to_index[test_sentence[j]])
            context.append(context_words)

model.eval()

with torch.no_grad():
    context_tensor = torch.tensor(context)
    output = model(context_tensor)
    predicted_word_index = torch.argmax(output, dim=1).item()
    predicted_word = vocab[predicted_word_index]

print("Predicted word:", predicted_word)
# CBOW:2 ends here
