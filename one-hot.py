from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode the features
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Train a decision tree classifier on one-hot encoded data
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_encoded, y_train)

# Evaluate the model on the test set
accuracy = tree_model.score(X_test_encoded, y_test)
print("Accuracy:", accuracy)
