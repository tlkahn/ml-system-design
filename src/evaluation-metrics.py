# [[file:../README.org::*Metrics][Metrics:1]]
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
  - X, y, test_size=0.2, random_state=42
)

# Train a classifier
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{cm}")

# Modify predictions to increase false positives
y_pred_modified = np.copy(y_pred)
y_pred_modified[:20] = 1  # Force the first 20 instances to be positive

# Recalculate metrics
precision_modified = precision_score(y_test, y_pred_modified)
recall_modified = recall_score(y_test, y_pred_modified)
f1_modified = f1_score(y_test, y_pred_modified)

print("\nModified Metrics:")
print(f"Precision: {precision_modified}")
print(f"Recall: {recall_modified}")
print(f"F1 Score: {f1_modified}")
# Metrics:1 ends here
