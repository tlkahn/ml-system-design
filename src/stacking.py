# [[file:../README.org::*stacking][stacking:1]]
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28
)

# Base models
model1 = RandomForestClassifier(random_state=36)
model2 = GradientBoostingClassifier(random_state=98)

# Train base models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Base models' predictions
pred1 = model1.predict(X_train)
pred2 = model2.predict(X_train)

print(f"Accuracy of model 1: {accuracy_score(y_train, pred1)}")
print(f"Accuracy of model 2: {accuracy_score(y_train, pred2)}")

stacked_predictions_train = np.column_stack((pred1, pred2))

# Train meta-model
meta_model = LogisticRegression(random_state=12)
meta_model.fit(stacked_predictions_train, y_train)

# Test predictions
test_pred1 = model1.predict(X_test)
test_pred2 = model2.predict(X_test)
stacked_predictions_test = np.column_stack((test_pred1, test_pred2))

# Meta-model's final prediction
final_prediction = meta_model.predict(stacked_predictions_test)

# Accuracy
print(f"Stacking Model Accuracy: {accuracy_score(y_test, final_prediction)}")
print(f"Recall Score: {recall_score(y_test, final_prediction, average='macro')}")
print(f"F1 Score: {f1_score(y_test, final_prediction, average='macro')}")
# stacking:1 ends here
