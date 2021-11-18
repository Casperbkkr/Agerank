import pandas as pd
import numpy as np
import os
t = 100
data = []
for i in range(50):
    name = os.path.join("Results", "Old_young_"+str(i)+".csv")
    file = pd.read_csv(name)
    data.append(file["total infected"].iloc[300])

mean = np.mean(data)
var = np.var(data)
std_dev = np.std(data)

print(mean, std_dev, var)
data2 = []
for i in range(100):
    name = os.path.join("Results", "Young_old_"+str(i)+".csv")
    file = pd.read_csv(name)
    data2.append(file["total infected"].iloc[300])

mean = np.mean(data2)
var = np.var(data2)
std_dev = np.std(data2)

print(mean, std_dev, var)