import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

training = pd.read_csv("training.csv")

training.replace(np.nan, -99999, inplace=True)

training.drop(["RefId"], 1, inplace=True)


drop_list = []

for x in training.columns:
    if training[x].dtype == "object":
        drop_list.append(x)


drop_list.append("IsBadBuy")
drop_list.append("WheelTypeID")
print(drop_list)

"""
y = df["party"].values
X = df.drop(["party"], axis=1).values
"""

X = np.array(training.drop(drop_list, 1))
y = np.array(training["IsBadBuy"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=100)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
accuracy = clf.score(X_train, y_train)
print(accuracy)

