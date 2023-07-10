import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample input data
data = [['hello', 'world'],
        ['goodbye', 'world'],
        ['hello', 'goodbye'],
        ['world', 'hello']]

# Vocabulary
vocab = list(set([word for sentence in data for word in sentence]))
vocab_size = len(vocab)

# Word-to-index mapping
word_to_index = {word: i for i, word in enumerate(vocab)}

# Generate training data
training_data = []
for sentence in data:
    for i, target_word in enumerate(sentence):
        context_words = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(sentence):
                context_words.append(word_to_index[sentence[j]])
        training_data.append((word_to_index[target_word], context_words))


class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target, context = self.data[index]
        return torch.tensor(target), torch.tensor(context)


# Skip-gram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.fc(embedded)
        return output


# Training parameters
embedding_dim = 10
epochs = 100
batch_size = 64
learning_rate = 0.1

# Create Skip-gram model instance
model = SkipGram(vocab_size, embedding_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Create DataLoader for training data
train_dataset = SkipGramDataset(training_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0

    for target, context in train_loader:
        optimizer.zero_grad()

        output = model(target)
        loss = criterion(output.view(-1, vocab_size), context.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")

# Test the model
test_word = 'hello'
test_index = word_to_index[test_word]

model.eval()

with torch.no_grad():
    output = model(torch.tensor([test_index]))
    predicted_word_index = torch.argmax(output).item()
    predicted_word = vocab[predicted_word_index]

print("Predicted word:", predicted_word)
