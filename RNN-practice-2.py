#import libraries
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    
    return np.array(data), np.array(labels)

#load dataset
df = pd.read_csv('Practice-data/EURUSD.csv')
df.set_index('Gmt time', inplace=True)

#removing data from after close of play Friday as there were no changes in 
#values
df = df[:7021]

first_80pct = int(0.8 * len(df))
next_10pct = int(0.1 * len(df))

#standardize the train dataset
dataset = df.values
data_mean = dataset[:first_80pct].mean(axis=0)
data_std = dataset[:first_80pct].mean(axis=0)
dataset = (dataset - data_mean) / data_std

#split into data sequences and target values
PAST_HISTORY = 60
FUTURE_TARGET = 3
STEP = 1

print(dataset[:, 1])