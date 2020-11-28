import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

training = pd.read_csv("training.csv")

for x in training["MMRCurrentAuctionCleanPrice"]:
    if x == "nan":
        print(x)

training.replace(np.nan, -99999, inplace=True)

training.drop(["RefId"], 1, inplace=True)

X = np.array(training.drop(["IsBadBuy", "PurchDate", "Auction", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "Nationality", "Size", "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "VNST", ], 1))  # inplace = True
y = np.array(training["IsBadBuy"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

for x in training["Auction"].iloc:
    if x != "ADESA" and x != "OTHER" and x != "MANHEIM":
        print(x)
"""
example_measures = np.array([4,2,1,1,5,6,5,4,3,2,2])
prediction = clf.predict(example_measures)
example_measures = example_measures.reshape(len(example_measures), -1)
print(prediction)
"""

"""
y = training["IsBadBuy"].values
X = training.drop(["IsBadBuy"], axis=1).values

knn = KNeighborsClassifier(6)

knn.fit(X, y)

y_pred = knn.predict(X)

new_prediction = knn.predict(X)
print("Prediction: {}".format(new_prediction))
"""