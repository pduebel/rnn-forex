import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing

def classify(current, future):
    #function to return 1, or buy, when the future value is higher than the
    #current value and return 0, or sell, when it is lower
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    for col in df.columns:
        if col != 'target':
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    return df.values[:, 1:], df.values[:, 0]

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

MOV_AVG1 = 10
MOV_AVG2 = 20
MOV_AVG3 = 40
FUTURE_TARGET = 60

df = pd.read_csv('Practice-data/EURUSD.csv')
df.set_index('Local time', inplace=True)
df = df[['Close']]
df = df[:7022]

df[f'{MOV_AVG1}avg'] = df[['Close']].ewm(span=MOV_AVG1).mean()
df[f'{MOV_AVG2}avg'] = df[['Close']].ewm(span=MOV_AVG2).mean()
df[f'{MOV_AVG3}avg'] = df[['Close']].ewm(span=MOV_AVG3).mean()

#df.plot()
#plt.xticks(ticks=np.arange(0, len(df['Close']), step=10), labels=None)
#plt.show()

df['future'] = df['Close'].shift(-FUTURE_TARGET)
df['target'] = list(map(classify, df['Close'], df['future']))

df[f'{MOV_AVG1}-{MOV_AVG2}_diff'] = df[f'{MOV_AVG1}avg'] - df[f'{MOV_AVG2}avg']
df[f'{MOV_AVG1}-{MOV_AVG3}_diff'] = df[f'{MOV_AVG1}avg'] - df[f'{MOV_AVG3}avg']
df[f'{MOV_AVG2}-{MOV_AVG3}_diff'] = df[f'{MOV_AVG2}avg'] - df[f'{MOV_AVG3}avg']

df.drop(['Close', 'future', f'{MOV_AVG1}avg', f'{MOV_AVG2}avg', f'{MOV_AVG3}avg'],
        axis=1, inplace=True)

first_80pct = int(0.8 * len(df))
next_10pct = int(0.1 * len(df))

train_df = df[:first_80pct]
validation_df = df[first_80pct:first_80pct + next_10pct]
test_df = df[first_80pct + next_10pct:]

train_x, train_y = preprocess_df(train_df)
validation_x, validation_y = preprocess_df(validation_df)
test_x, test_y = preprocess_df(test_df)

BATCH_SIZE = 200
BUFFER_SIZE = 10000

train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
val_data = val_data.batch(BATCH_SIZE)

test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, input_shape=(3,), activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

#define optimizer and loss
opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 20

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

#train model
single_step_history = model.fit(train_data, 
                                epochs=EPOCHS,
                                validation_data=val_data,
                                )

plot_train_history(single_step_history, 
                   'Single Step Training and validation loss')

model.evaluate(test_x, test_y)