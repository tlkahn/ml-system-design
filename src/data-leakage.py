# [[file:../README.org::*Causes][Causes:1]]
# Temporal leakage
import pandas as pd

# Load data
data = pd.read_csv("stock_prices.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Incorrect: Shuffling before splitting
shuffled_data = data.sample(frac=1)
train_data = shuffled_data[:800]
test_data = shuffled_data[800:]

# Correct: Sorting and splitting by date
sorted_data = data.sort_values(by="Date")
train_data = sorted_data[:800]
test_data = sorted_data[800:]
# Causes:1 ends here

# [[file:../README.org::*Causes][Causes:2]]
# Target leakage
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("cancer_data.csv")

# Incorrect: Including target-related feature
X = data[["Age", "Gender", "Tumor_Size", "Treatment"]]
y = data["Cancer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Correct: Excluding target-related feature
X = data[["Age", "Gender", "Tumor_Size"]]
y = data["Cancer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Causes:2 ends here

# [[file:../README.org::*Causes][Causes:3]]
# Improper preprocessing leakage
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Incorrect: Scaling before splitting, leaking global and test statistics to train data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Correct: Scaling after splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Causes:3 ends here
