import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_signal_td3 import Signal 

data = pd.read_csv("strasbourg_entzheim.csv",encoding="utf-8") 
end_of_data=len(data)

def create_signal(tag):
    tag_column=data[tag].interpolate().to_numpy()
    axe_t= np.arange(len(tag_column))
    return Signal(lambda t: np.interp(t, axe_t, tag_column), 1)

def get_max_value(tag):
    return data[tag].max()

def get_min_value(tag):
    return data[tag].min()

def get_mean_value(tag):
    return data[tag].mean()

#test
Tmax=create_signal("tmax")
Tmax.plot(0,end_of_data)

print(f"Neige maximale : {get_max_value('snow')}")
print(f"Pression minimale : {get_min_value('pres')}")
print(f"Pression moyenne : {get_mean_value('pres')}")