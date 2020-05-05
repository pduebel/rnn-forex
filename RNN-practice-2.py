# Combination of data series tutorial in tensorflow docs and YouTube tutorial 
# by Sentdex - takes time series data of four currency pairs, combines them 
# into one dataframe, preprocesses (scaling etc.), splits into sequences of 
# certain time length and then use RNN in tensorflow to predict whether the 
# value of the target currency pair will be higher or lower a set amount of 
# time in the future.

#This produces results no better than random - next attempt in NN-averages


#import libraries
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
        data.append(dataset[indices])

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
    dataset = f"Practice-data/{ratio}.csv"
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

main_df = main_df[:7022]

first_80pct = int(0.8 * len(main_df))
next_10pct = int(0.1 * len(main_df))

#divide dataframe into train, val and test
train_df = main_df[:first_80pct]
validation_df = main_df[first_80pct:first_80pct + next_10pct]
test_df = main_df[first_80pct + next_10pct:]

train_df = preprocess_df(train_df)
validation_df = preprocess_df(validation_df)
test_df = preprocess_df(test_df)

train_df_vals = train_df.values
validation_df_vals = validation_df.values
test_df_vals = test_df.values

#split into data sequences and target values
x_train, y_train = multivariate_data(train_df_vals, train_df_vals[:, -1], 0,
                                     None, PAST_HISTORY,
                                     FUTURE_TARGET, STEP,
                                     single_step=True)
x_val, y_val = multivariate_data(validation_df_vals, validation_df_vals[:, -1],
                                 0, None, PAST_HISTORY,
                                 FUTURE_TARGET, STEP,
                                 single_step=True)
x_test, y_test = multivariate_data(test_df_vals, test_df_vals[:, -1],
                                   0, None, PAST_HISTORY,
                                   FUTURE_TARGET, STEP,
                                   single_step=True)


BATCH_SIZE = 100
BUFFER_SIZE = 10000

#shuffle data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE)

#define model layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=x_train.shape[-2:], return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(2, activation='softmax'))

#define optimizer and loss
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EVALUATION_INTERVAL = 200
EPOCHS = 10

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

#train model
single_step_history = model.fit(train_data, 
                                epochs=EPOCHS,
                                validation_data=val_data,
                                callbacks=[early_stopping]
                                )

#produce plot of losses
plot_train_history(single_step_history, 
                   'Single Step Training and validation loss')

model.evaluate(x_test, y_test)

model.save('Saved-models\Model1')