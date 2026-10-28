import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_signal_td3 import Signal 

#read the csv file
data = pd.read_csv("strasbourg_entzheim.csv",encoding="utf-8") 
end_of_data=len(data)

def create_signal(tag):
    tag_column=data[tag].interpolate().to_numpy()
    axe_t= np.arange(len(tag_column))
    return Signal(lambda t: np.interp(t, axe_t, tag_column), 1)

#functions to get stats
def get_max_value(tag):
    return data[tag].max()

def get_min_value(tag):
    return data[tag].min()

def get_mean_value(tag):
    return data[tag].mean()

def get_variance_value(tag):
    return data[tag].var()

def get_ecart_type_value(tag):
    return data[tag].std(ddof=0)

#tests
Tmax=create_signal("tmax")
Tmax.plot(0,end_of_data)

print(f"Neige maximale : {get_max_value('snow')}")
print(f"Pression minimale : {get_min_value('pres')}")
print(f"Pression moyenne : {get_mean_value('pres')}")
print(f"Variance pour la vitesse du vent : {get_variance_value('wspd')}")
print(f"Ecart type pour la la vitesse du vent : {get_ecart_type_value('wspd')}")