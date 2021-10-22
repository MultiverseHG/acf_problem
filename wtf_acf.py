import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr


# read in data
bus = pd.read_csv('cleaned_bus.csv', index_col='Month', parse_dates=True)

# compute first 12 lags of ACF with statsmodels
sm_acf = sm.tsa.acf(bus, nlags=12, fft=False)
print(sm_acf)

# compute ACF by shifting and using scipy.stats.pearsonr
# drop nulls that arise from shifting and corresponding rows of unshifted data
trimmed_acf = []
for lag in range(13):
    shifted = bus.riders.shift(lag).iloc[lag:]
    trimmed = bus.riders.iloc[lag:]
    corr = pearsonr(shifted, trimmed)[0] # [0] to grab r ([1] is p-value)
    trimmed_acf.append(corr)
trimmed_acf = np.array(trimmed_acf)
print(trimmed_acf)

# not the same - how different are they?
print(sm_acf - trimmed_acf)

# maybe acf is filling in the missing values with zeroes?
# from looking at the source code, doesn't seem like it, but let's try
zeroed_acf = []
for lag in range(13):
    shifted = bus.riders.shift(lag).fillna(0)
    corr = pearsonr(shifted, bus.riders)[0]
    zeroed_acf.append(corr)
zeroed_acf = np.array(zeroed_acf)
print(zeroed_acf)

# different again!
print(sm_acf - zeroed_acf)

# so why does statsmodels give a different ACF than calculating directly with Pearson's r?
