import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

training = pd.read_csv("training.csv")

#####################################################
# PLOT WARRANTY UND ODOMETER SCATTER
groups = training.groupby('IsBadBuy')

for name, group in groups:
    plt.plot(group["WarrantyCost"], group["VehOdo"], marker='o', linestyle='', label=name)

plt.xlabel("WarrantyCost")
plt.ylabel("VehOdo")
plt.legend()
plt.show()
###################################################
