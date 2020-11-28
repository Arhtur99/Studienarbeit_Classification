import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

training = pd.read_csv("training.csv")


plt.scatter(x = training["WarrantyCost"], y = training["VehBCost"], c = training["IsBadBuy"], label = "IsBadBuy")
plt.legend()
plt.show()

groups = training.groupby('IsBadBuy')

for name, group in groups:
    plt.plot(group["KickDate"], group["VehOdo"], marker='o', linestyle='', label=name)

plt.xlabel("VehicleAge")
plt.ylabel("VehOdo")
plt.legend()
plt.show()

for x in training.iloc:
    if x["VehBCost"] > 40000:
        print(x)