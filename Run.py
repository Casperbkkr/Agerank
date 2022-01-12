import pandas as pd
import numpy as np
import os

from Datasets import filenames_dictionary
from Model import model
from Parameters import parameter

#  This code can be used for collecting data.

# Timesteps to simulate the model for
timesteps = 400
# The amount of times the same model has to be run
k = 20
# The type of order for the vaccination, choose from dictionary below.
vaccination_orders = [1, 2, 3]

# The names for the datasets to be saved
file_names = {1: "Old_young", 2: "Young_old", 3: "Mix"}

# Load the parameters to be used in the model. Can be changed in the parameters.py file.
parameters = parameter()

# Load the filenames of the datasets to be used
filenames = filenames_dictionary()

# Run the model k times for all vaccination orders specified
for vacc_order in vaccination_orders:
    results = model(parameters, filenames, vacc_order, timesteps)
    file_name1 = str(file_names[vacc_order]) + "_" + str(0) + ".csv"
    file_name_2 = os.path.join("Results", file_name1)
    results.data.to_csv(file_name_2)
    print("Saved as:", file_name1)
    results_total = results.data

    for i in range(1, k):
        results = model(parameters, filenames, vacc_order, timesteps)

        file_name1 = str(file_names[vacc_order]) + "_" + str(i) + ".csv"
        file_name_2 = os.path.join("Results", file_name1)
        results.data.to_csv(file_name_2)
        print("Saved as:", file_name1)
        results_total = results_total + results.data

    results_total = results_total / k
    tracker = results_total.astype(int)
    file_name_total_1 = str(file_names[vacc_order]) + "_total.csv"
    file_name_total_2 = os.path.join("Results", file_name_total_1)
    results_total.to_csv(file_name_total_2)

    print("Saved as:", file_name_total_1)
