import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_signal_td3 import Signal 

#read the csv file
data = pd.read_csv("strasbourg_entzheim.csv",encoding="utf-8") 
end_of_data=len(data)

def new_signal(tag):
    tag_column=data[tag].interpolate().to_numpy()
    axe_t= np.arange(len(tag_column))
    return Signal(lambda t: np.interp(t, axe_t, tag_column), 1),tag_column

#Affiche le spectre en bilatéral
def spectrum_bilateral(tag_column):
    n=len(tag_column)
    amp=np.abs(np.fft.fft(tag_column))/n
    freq =np.fft.fftfreq(n,1)
    freq_max =np.max(freq)

    return freq, amp
    plt.xlim(-freq_max,freq_max)
    plt.grid(True)
    plt.stem(freq, amp, markerfmt=" ")
    plt.show()


#Affiche le spectre en unilatéral
def spectrum_unilateral(tag_column):
    n=len(tag_column)
    amp=np.abs(np.fft.fft(tag_column))/n
    freq =np.fft.fftfreq(n,1)
    freq_max =np.max(freq)

    freq = freq[:len(freq)//2+1]
    amp = amp[:len(amp)//2+1]
    amp[1:] *= 2

    return freq, amp



def rectructruct_signal(tag_column,nb_harmonique):
    Amp=np.fft.fft(tag_column)
    axe_t= np.arange(nb_harmonique)
    signal_recontruit=np.real(np.fft.ifft(Amp,nb_harmonique))
    return axe_t, signal_recontruit

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

def SerieF (t):
    #Par lecture graphique, création de la liste [freq, amp] avec 1 harmonique grâce au  spectre unilatéral
    framp = [[0, 15.274], [0.00274, 10.716], [0.00548, 1.063]]

    phi = np.pi/1.85 #déphasage du signal
    return framp[0][1] + framp[1][1] * np.sin(np.pi * 2 * framp[1][0] * t - phi) + framp[2][1] * np.sin(np.pi * 2 * framp[2][0] * t - phi)

def filtrePB(e, fc):
    """
    e : signal d'entrée
    fc: fréquence de coupure (-3dB)
    """
    s = [e[0]] # on fait l'hypothèse mais si ça ne va pas, on changera
    Te = 1 #A déterminer
    tau = 1/(2* np.pi *fc)
    A = Te / (Te + tau)
    B = tau / (Te + tau)
    for i in range(1, len(e)):
        s.append(A * e[i] + B * s[i-1])
    return s

def filtrePH(e, fc):
    """
    e : signal d'entrée
    fc: fréquence de coupure (-3dB)
    """
    s = [e[0]] # on fait l'hypothèse mais si ça ne va pas, on changera
    Te = 1 #A déterminer
    tau = 1 #A déterminer
    A = 1 + tau/Te
    B = 1 / Te
    for i in range(1, len(e)):
        s.append(A * e[i] + B * e[i-1])
    return s


#tests
Tmax,list_Tmax=new_signal("tmax")
# Tmax.plot(0,end_of_data)
# rectructruct_signal(tag_Tmax,100)
# rectructruct_signal(tag_Tmax,500)
# rectructruct_signal(tag_Tmax,1000)
# print("longeur tag :",len(tag_Tmax))
# rectructruct_signal(tag_Tmax,len(tag_Tmax))
# spectrum_bilateral(tag_Tmax)
# spectrum_unilateral(tag_Tmax)

temp = np.arange(0, len(list_Tmax))
a, signal = rectructruct_signal(list_Tmax,len(list_Tmax))
signal_FiltrePB0 = filtrePB(signal, 0.001)
signal_FiltrePB1 = filtrePB(signal, 0.003)
signal_FiltrePB2 = filtrePB(signal, 0.006)

freq, amp = spectrum_unilateral(signal)
freq0, amp0 = spectrum_unilateral(signal_FiltrePB0)
freq1, amp1 = spectrum_unilateral(signal_FiltrePB1)
freq2, amp2 = spectrum_unilateral(signal_FiltrePB2)
freq_max = np.max(freq)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(a, signal, color="r")
# ax1.plot(temp, SerieF(temp))
ax1.plot(temp, signal_FiltrePB2, "--", label="Fc à 0.006Hz", color="b")
ax1.plot(temp, signal_FiltrePB1, "--", label="Fc à 0.003Hz", color="orange")
ax1.plot(temp, signal_FiltrePB0, ":", label="Fc à 0.001Hz", color="g")
ax1.set_xlabel("Temps en jour")
ax1.set_ylabel("T max")

ax2.set_xlim(-freq_max,freq_max)
ax2.stem(freq, amp, markerfmt="x")
ax2.stem(freq0, amp0, markerfmt="x")
ax2.stem(freq1, amp1, markerfmt="x")
ax2.stem(freq2, amp2, markerfmt="x")

ax1.grid(which="both")
ax2.grid(which="both")
ax1.legend()
ax2.legend()
plt.show()



# print(f"Neige maximale : {get_max_value('snow')}")
# print(f"Pression minimale : {get_min_value('pres')}")
# print(f"Pression moyenne : {get_mean_value('pres')}")
# print(f"Variance pour la vitesse du vent : {get_variance_value('wspd')}")
# print(f"Ecart type pour la la vitesse du vent : {get_ecart_type_value('wspd')}")
