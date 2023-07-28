# [[file:../README.org::*Feature hashing][Feature hashing:2]]
from sklearn.feature_extraction import FeatureHasher
import numpy as np

# Example input data
data = [
    {"color": "red", "shape": "circle"},
    {"color": "blue", "shape": "triangle"},
    {"color": "green", "shape": "square"},
]

# Create a FeatureHasher object
hasher = FeatureHasher(n_features=20, input_type="dict")

# Transform the data
hashed_data = hasher.transform(data)

# Print the transformed features
print(hashed_data.toarray())
print(np.array(hashed_data.toarray()).shape)
# Feature hashing:2 ends here
