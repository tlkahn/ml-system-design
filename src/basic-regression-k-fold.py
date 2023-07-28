# [[file:../README.org::*Cross validation][Cross validation:3]]
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Create a sample dataset
data = {
    "Feature1": np.random.rand(20),
    "Feature2": np.random.rand(20),
    "Target": np.random.rand(20),
}
df = pd.DataFrame(data)

# Discretize the target variable into bins
num_bins = 5
labels = [f"Bin_{i}" for i in range(1, num_bins + 1)]
df["Target_Bin"] = pd.cut(df["Target"], bins=num_bins, labels=labels)

# Create StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Split the dataset into folds
for train_index, test_index in stratified_kfold.split(df, df["Target_Bin"]):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    print("Train set:\n", train_set, "\nTest set:\n", test_set, "\n---")
# Cross validation:3 ends here
