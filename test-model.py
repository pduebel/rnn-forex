# Script to test model produced by RNN-practice-2. Preprocesses historical 
# data in the same way and loads saved model, before using evaluate and
# predict methods to test how effective the model is.

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False    

def classify(current, future):
    #function to return 1, or buy, when the future value is higher than the
    #current value and return 0, or sell, when it is lower
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    #function to convert values in dataframe to pct change, scale and remove
    #inf and nan values
    df = df.drop('future', axis=1)
    
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    return df


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    #function to split data into sequences and divide into inputs and targets
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices, :-1])

        if single_step:
            labels.append(target[i])
        else:
            labels.append(target[i:i + target_size])
    
    return np.array(data), np.array(labels)


def plot_train_history(history, title):
    #function that produces plot showing training and validation losses
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

#load dataset
main_df = pd.DataFrame()

ratios = ['EURUSD', 'GBPUSD', 'USDCHF']

#rename columns to distinguish data from different ratios, combine into one
for ratio in ratios:
    dataset = f"Test-data/April/{ratio}.csv"
    df = pd.read_csv(dataset)
    df.rename(columns={"Close": f"{ratio}_close", "Volume": f"{ratio}_volume"},
              inplace=True)
    df.set_index('Local time', inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)


RATIO_TO_PREDICT = "EURUSD"
PAST_HISTORY = 60
FUTURE_TARGET = 30
STEP = 1

#create targets
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_TARGET)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

main_df = preprocess_df(main_df)

main_df_vals = main_df.values

#split into data sequences and target values
x_test, y_test = multivariate_data(main_df_vals, main_df_vals[:, -1],
                                   0, None, PAST_HISTORY,
                                   FUTURE_TARGET, STEP,
                                   single_step=True)

#x = np.reshape(x_test[0], (1, 60, 7))
print(x_test)

#load model and evaluate accuracy at predicting different historical data
#model = tf.keras.models.load_model('Saved-models/Model1')
#print(model.evaluate(x_test))
#print(model.redict(x))
