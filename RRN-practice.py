import pandas as pd
import numpy as np
import random

from collections import deque
from sklearn import preprocessing

SEQ_LEN = 60
PREDICT_SEQ_LEN = 3
RATIO_TO_PREDICT = "EURUSD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', axis=1)

    for col in df.columns:
        if col != 'target':
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)


main_df = pd.DataFrame()

ratios = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY']
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
    df.set_index('Gmt time', inplace=True)
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-PREDICT_SEQ_LEN)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

#print(main_df[["EURUSD_close", "future", 'target']].head())

first_80pct = int(0.8 * len(main_df))
next_10pct = int(0.1 * len(main_df))

train_main_df = main_df[:first_80pct]
validation_main_df = main_df[first_80pct:first_80pct + next_10pct]
test_main_df = main_df[first_80pct + next_10pct:]

#print(len(main_df))
#print(len(train_main_df), len(validation_main_df), len(test_main_df))
#print(len(train_main_df) + len(validation_main_df) + len(test_main_df))

preprocess_df(train_main_df)