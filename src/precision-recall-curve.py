# [[file:../README.org::*Metrics][Metrics:2]]
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train a logistic regression model
model = LogisticRegression(solver="liblinear", random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import precision_recall_curve

# Get predicted probabilities for the positive class
y_scores = model.predict_proba(X_test)[:, 1]

# Compute the Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure()
plt.plot(recall, precision, marker=".")
num_thresholds_to_display = 5
threshold_indices = np.linspace(
    0, len(thresholds) - 1, num_thresholds_to_display, dtype=int
)

for i in threshold_indices:
    plt.annotate(
        f"Thresh: {thresholds[i]:.2f}",
        xy=(recall[i], precision[i]),
        xytext=(recall[i] - 0.1, precision[i] + 0.02),
        arrowprops={
            "facecolor": "black",
            "arrowstyle": "wedge,tail_width=0.7",
            "lw": 1,
            "alpha": 0.5,
        },
        fontsize=9,
        color="black",
    )

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("recision-Recall Curve")

from sklearn.metrics import roc_curve, auc

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("../img/precision-recall-roc-example.png")
# Metrics:2 ends here
