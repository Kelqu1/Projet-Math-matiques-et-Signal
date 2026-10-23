import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_signal_td3 import Signal 

years=74

data = pd.read_csv("strasbourg_entzheim.csv",encoding="utf-8") 

avg_temp=data["tavg"].interpolate.to_numpy()
axe_t= np.arange(len(avg_temp))

sig_avg_temp=Signal(lambda t: np.interp(t, axe_t, avg_temp), 1)
sig_avg_temp.plot(0, 365 * years)