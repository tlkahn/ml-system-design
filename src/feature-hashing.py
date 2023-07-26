# [[file:../README.org::*Feature hashing][Feature hashing:1]]
from sklearn.feature_extraction import FeatureHasher

# Example input data
data = [{'color': 'red', 'shape': 'circle'},
        {'color': 'blue', 'shape': 'triangle'},
        {'color': 'green', 'shape': 'square'}]

# Create a FeatureHasher object
hasher = FeatureHasher(n_features=10, input_type='dict')

# Transform the data
hashed_data = hasher.transform(data)

# Print the transformed features
print(hashed_data.toarray())
# Feature hashing:1 ends here
