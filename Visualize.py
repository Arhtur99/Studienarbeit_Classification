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

#PLOT FÜR VehOdo und IsBadBuy jeweils in 6000 Meilen Schritten

data_vehodo_isbadbuy = dataframefeatures[["VehOdo", "IsBadBuy"]]
upper = 6000
lower = 0
y_val = []
x_val = []
while upper <= 120000:
    data = data_vehodo_isbadbuy.loc[(data_vehodo_isbadbuy.VehOdo >= lower) & (data_vehodo_isbadbuy.VehOdo < upper)]
    data_list = list(data.IsBadBuy.value_counts())
    
    if len(data_list) < 2:
        y_val.append(data_list[0] / data_list[0])
        
    else:
        if data_list[1] == data_list[0]:
            y_val.append(0)
        else:
            y_val.append(data_list[1] / data_list[0])
    
    x_val.append(upper)   
    lower = upper
    upper += 6000
    
plt.plot(x_val, y_val)

plt.xlabel("VehOdo")
plt.ylabel("IsBadBuy")
plt.show()


###################################################


#Next Plot



###################################################
