# [[file:../README.org::*Exploratory data analysis (EDA)][Exploratory data analysis (EDA):1]]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")
pixel_values, targets = data
targets = targets.astype(int)
single_image = pixel_values.iloc[1, :].values.reshape(28, 28)
plt.imshow(single_image, cmap="gray")
plt.savefig("../img/single_mnist_image.png")
plt.clf()

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values.iloc[:100, :])

tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:100])), columns=["x", "y", "targets"]
)
# tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
grid = sns.FacetGrid(tsne_df, hue="targets")
grid.map(sns.scatterplot, "x", "y").add_legend()
plt.savefig("../img/tsne.png")
# Exploratory data analysis (EDA):1 ends here
