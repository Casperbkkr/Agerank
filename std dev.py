import pandas as pd
import numpy as np
import os
t = 100
data = []
for i in range(100):
    name = os.path.join("Results", "Young_old_"+str(i)+".csv")
    file = pd.read_csv(name)
    data.append(file["deceased"].iloc[200])

mean = np.mean(data)
var = np.var(data)
std_dev = np.std(data)

print(mean, std_dev, var)

for i in range(100):
    name = os.path.join("Results", "Danish_"+str(i)+".csv")
    file = pd.read_csv(name)
    data.append(file["deceased"].iloc[200])

mean = np.mean(data)
var = np.var(data)
std_dev = np.std(data)

print(mean, std_dev, var)