# [[file:README.org::*Feature hashing][Feature hashing:1]]
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data
data = [
    {"user_id": "user1", "ad_id": "ad1", "age": "25", "gender": "M", "clicked": 1},
    {"user_id": "user2", "ad_id": "ad2", "age": "30", "gender": "F", "clicked": 0},
    {"user_id": "user3", "ad_id": "ad3", "age": "35", "gender": "M", "clicked": 0},
    {"user_id": "user4", "ad_id": "ad4", "age": "40", "gender": "F", "clicked": 1},
]

# Extract features and labels
features = [{k: v for k, v in item.items() if k != "clicked"} for item in data]
labels = [item["clicked"] for item in data]

# Use FeatureHasher to handle high-cardinality categorical features
hasher = FeatureHasher(n_features=20, input_type="dict")
hashed_features = hasher.transform(features).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hashed_features, labels, test_size=0.25, random_state=42
)

# Train a logistic regression model
clf = LogisticRegression(solver="lbfgs")
clf.fit(X_train, y_train)

# Predict on the test set and calculate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Predicted values: ", y_pred)
print("Accuracy: ", accuracy)
# Feature hashing:1 ends here
