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


def plot_train_history(history, title):
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
df = pd.read_csv('Practice-data/EURUSD.csv')
df.set_index('Gmt time', inplace=True)

#removing data from after close of play Friday as there were no changes in 
#values
df = df[:7021]

PAST_HISTORY = 60
FUTURE_TARGET = 3
STEP = 1

df['future'] = df['Close'].shift(-FUTURE_TARGET)
df['target'] = list(map(classify, df['Close'], df['future']))

first_80pct = int(0.8 * len(df))
next_10pct = int(0.1 * len(df))

train_df = df[:first_80pct]
validation_df = df[first_80pct:first_80pct + next_10pct]
test_df = df[first_80pct + next_10pct:]

train_df = preprocess_df(train_df)
validation_df = preprocess_df(validation_df)
test_df = preprocess_df(test_df)

train_df = train_df.values
validation_df = validation_df.values
test_df = test_df.values

print(train_df[:, 5])

#split into data sequences and target values
x_train, y_train = multivariate_data(train_df, train_df[:, 5], 0,
                                     None, PAST_HISTORY,
                                     FUTURE_TARGET, STEP,
                                     single_step=True)
x_val, y_val = multivariate_data(validation_df, validation_df[:, 5],
                                 0, None, PAST_HISTORY,
                                 FUTURE_TARGET, STEP,
                                 single_step=True)

BATCH_SIZE = 100
BUFFER_SIZE = 10000

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE)

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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EVALUATION_INTERVAL = 200
EPOCHS = 10

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

single_step_history = model.fit(train_data, epochs=EPOCHS,
                                validation_data=val_data,
                                validation_steps=1,
                                callbacks=[early_stopping])

plot_train_history(single_step_history, 
                   'Single Step Training and validation loss')

