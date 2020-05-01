# Based on YouTube tutorial by Sentdex - takes time series data of four currency 
# pairs, combines them into one dataframe, preprocesses (scaling and balancing 
# etc.), splits into sequences of certain time length and then use RNN in 
# tensorflow to predict whether the value of the target currency pair will be 
# higher or lower a set amount of time in the future.

#Accuracy no better than random, try slightly different preprocessing in practice 2

#import libraries
import pandas as pd
import numpy as np
import random
import time
import tensorflow as tf

from collections import deque
from sklearn import preprocessing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

#set hyperparameters
SEQ_LEN = 60
PREDICT_SEQ_LEN = 3
END_OF_WEEK = 7021
RATIO_TO_PREDICT = "BTCUSD"
EPOCHS = 10
BATCH_SIZE = 64

def classify(current, future):
    #function to return 1, or buy, when the future value is higher than the
    #current value and return 0, or sell, when it is lower
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df, balance):
    #preprocessed dataframe by converting absolute values to pct change,
    #scaling, shuffling, splitting series into sequences and balancing
    #if balance set to True
    df = df.drop('future', axis=1)

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    if balance == True:
        buys = []
        sells = []

        for seq, target in sequential_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])
        
        random.shuffle(buys)
        random.shuffle(sells)

        lower = min(len(buys), len(sells))

        buys = buys[:lower]
        sells = sells[:lower]

        sequential_data = buys + sells
        random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
        
    return np.array(x), y

#combine timeseries data from different currency pairs into one pandas dataframe
main_df = pd.DataFrame()

ratios = ['BTCUSD', 'LTCUSD', 'ETHUSD']

for ratio in ratios:
    dataset = "Practice-data/%s.csv" %(ratio)
    df = pd.read_csv(dataset)
    df.rename(columns={"Open": "%s_open" %(ratio),
                       "High": "%s_high" %(ratio),
                       "Low": "%s_low" %(ratio),
                       "Close": "%s_close" %(ratio),
                       "Volume": "%s_volume" %(ratio)
                       },
              inplace=True)
    df.set_index('Local time', inplace=True)
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

#create targets for model using classify function
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-PREDICT_SEQ_LEN)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

#split into train, validation and test samples and preprocessing using
#preprocess_df function
first_80pct = int(0.8 * len(main_df))
next_10pct = int(0.1 * len(main_df))

train_main_df = main_df[:first_80pct]
validation_main_df = main_df[first_80pct:first_80pct + next_10pct]
test_main_df = main_df[first_80pct + next_10pct:]

train_x, train_y = preprocess_df(train_main_df, True)
validation_x, validation_y = preprocess_df(validation_main_df, True)
test_x, test_y = preprocess_df(test_main_df, False)

print(f"train: {len(train_x)} validation: {len(validation_x)} test: {len(test_x)}")
print(f"Train 0: {train_y.count(0)} train 1: {train_y.count(1)}")
print(f"Validation 0: {validation_y.count(0)} validation 1: {validation_y.count(1)}")
print(f"Test 0: {test_y.count(0)} test 1: {test_y.count(1)}")

#converting targets from list to np array so works in model
train_y = np.asarray(train_y)
validation_y = np.asarray(validation_y)
test_y = np.asarray(test_y)

#defining model 
model = Sequential([
                    LSTM(128, input_shape= train_x.shape[-2:], return_sequences=True),
                    Dropout(0.2),
                    BatchNormalization(),

                    LSTM(128, return_sequences=True),
                    Dropout(0.1),
                    BatchNormalization(),

                    LSTM(128),
                    Dropout(0.2),
                    BatchNormalization(),

                    Dense(32, activation='relu'),
                    Dropout(0.2),

                    Dense(2, activation="softmax")
                    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training model
model.fit(train_x,
          train_y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(validation_x, validation_y),
          )