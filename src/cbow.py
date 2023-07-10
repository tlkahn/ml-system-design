import numpy as np

# Sample input data
data = [
    ["hello", "world"],
    ["goodbye", "world"],
    ["hello", "goodbye"],
    ["world", "hello"],
]

# Vocabulary
vocab = set([word for sentence in data for word in sentence])
vocab_size = len(vocab)

# Word-to-index mapping
word_to_index = {word: i for i, word in enumerate(vocab)}

# Context window size
window_size = 2

# Generate training data
X_train = []
y_train = []

for sentence in data:
    for i, target_word in enumerate(sentence):
        context_words = []

        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(sentence):
                context_words.append(sentence[j])

        X_train.append(context_words)
        y_train.append(target_word)

# Convert training data to one-hot vectors
X_train_onehot = np.zeros((len(X_train), vocab_size), dtype=np.float32)
y_train_onehot = np.zeros((len(y_train), vocab_size), dtype=np.float32)

for i, context_words in enumerate(X_train):
    for word in context_words:
        X_train_onehot[i, word_to_index[word]] = 1

    y_train_onehot[i, word_to_index[y_train[i]]] = 1

# Initialize weights
input_dim = vocab_size
hidden_dim = 10
output_dim = vocab_size

W1 = np.random.randn(input_dim, hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim)

# Training loop
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    hidden_layer = np.dot(X_train_onehot, W1)
    output_layer = np.dot(hidden_layer, W2)
    softmax_output = np.exp(output_layer) / np.sum(
        np.exp(output_layer), axis=1, keepdims=True
    )

    # Backward pass
    dW2 = np.dot(hidden_layer.T, (softmax_output - y_train_onehot))
    dW1 = np.dot(X_train_onehot.T, np.dot((softmax_output - y_train_onehot), W2.T))

    # Update weights
    W2 -= learning_rate * dW2
    W1 -= learning_rate * dW1

# Test the model
test_sentence = ["hello", "world"]
context = []
for i, target_word in enumerate(test_sentence):
    context_words = []
    for j in range(i - window_size, i + window_size + 1):
        if j != i and 0 <= j < len(test_sentence):
            context_words.append(test_sentence[j])
    context.append(context_words)

X_test = np.zeros((len(context), vocab_size), dtype=np.float32)
for i, context_words in enumerate(context):
    for word in context_words:
        X_test[i, word_to_index[word]] = 1

hidden_layer = np.dot(X_test, W1)
output_layer = np.dot(hidden_layer, W2)
softmax_output = np.exp(output_layer) / np.sum(
    np.exp(output_layer), axis=1, keepdims=True
)

predicted_word_index = np.argmax(softmax_output, axis=1)
predicted_word = [list(vocab)[idx] for idx in predicted_word_index]

print("Predicted word:", predicted_word)
