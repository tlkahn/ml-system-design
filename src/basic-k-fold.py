# [[file:../README.org::*Cross validation][Cross validation:2]]
import pandas as pd
from sklearn import model_selection
from sklearn.utils import Bunch
from sklearn import datasets

# Training data is in a CSV file called train.csv
iris_data = datasets.load_iris()
# df.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR',
# df.data.shape  # (150, 4)
# df.target.shape  # (150,)
# df.target_names  # array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
# df.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# we create a new column called kfold and fill it with -1
# create a pandas dataset from df.data
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df = df.sample(frac=1).reset_index(drop=True)
# initiate the kfold class from model_selection module
kf = model_selection.KFold(n_splits=5)
# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    # trn_ and val_ are indexes of the training and validation subsets
    # like:
    #   [ 30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
    #   48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65
    #   66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83
    #   84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
    #  102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
    #  120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
    #  138 139 140 141 142 143 144 145 146 147 148 149]
    #  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    #  24 25 26 27 28 29]
    df.loc[val_, "kfold"] = fold
# save the new csv with kfold column
df.to_csv("../data/train_folds.csv", index=False)
# Cross validation:2 ends here
