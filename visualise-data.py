# Takes historical price data of various currency pairs and plots the close
# values of to check for anythong that may affect the accuracy of the model.

import pandas as pd
import matplotlib.pyplot as plt

main_df = pd.DataFrame()

ratios = ['EURUSD', 'GBPUSD', 'USDCHF']

#rename columns to distinguish data from different ratios, combine into one
for ratio in ratios:
    dataset = f"Test-data/April/{ratio}.csv"
    df = pd.read_csv(dataset)
    df.rename(columns={"Close": f"{ratio}_close", "Volume": f"{ratio}_volume"},
              inplace=True)
    df.set_index('Local time', inplace=True)
    df = df[[f"{ratio}_close"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

first_80pct = int(0.8 * len(main_df))
next_10pct = int(0.1 * len(main_df))

# plot data
main_df.plot(subplots=True)
plt.show()
