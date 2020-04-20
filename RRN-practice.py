import pandas as pd

SEQ_LEN = 60
PREDICT_SEQ_LEN = 3
RATIO_TO_PREDICT = "EURUSD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


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

print(main_df[["EURUSD_close", "future", 'target']].head())