# [[file:../README.org::*Cross validation][Cross validation:1]]
# import scikit-learn tree and metrics
from sklearn import tree
from sklearn import metrics

# import matplotlib and seaborn # for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split

data = load_wine()
# data.data # 178 * 13
# stats.describe(data.data)
# data.data.shape # 13 dimensions
# data.target # {0, 1, 2}
# this is our global size of label text # on the plots
matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)
# This line ensures that the plot is displayed # inside the notebook
# initialize lists to store accuracies # for training and test data
# we start with 50% accuracy train_accuracies = [0.5] test_accuracies = [0.5]
# iterate over a few depth values
train_accuracies = test_accuracies = []
train_data, test_data, train_labels, test_labels = train_test_split(
    data.data, data.target, test_size=0.3
)
for depth in range(1, 50):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(train_data, train_labels)
    train_predictions = clf.predict(train_data)
    test_predictions = clf.predict(test_data)
    train_accuracy = metrics.accuracy_score(train_labels, train_predictions)
    test_accuracy = metrics.accuracy_score(test_labels, test_predictions)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# plot train_accuracies and test_accuracies
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.savefig("../img/overfitting-demo.png")
# plt.show()
# Cross validation:1 ends here
