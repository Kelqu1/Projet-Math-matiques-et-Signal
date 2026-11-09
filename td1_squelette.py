import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_signal_td3 import Signal 

#read the csv file
data = pd.read_csv("strasbourg_entzheim.csv", encoding="utf-8")
data["time"] = pd.to_datetime(data["time"])

#select the data to create a numpy array
#and create a numpy array with the time column
def new_signal(tag, pas=1, annees=(2014, 2024)):
    years = data[data["time"].dt.year.between(*annees)]
    tag_column = years[tag].interpolate().to_numpy()[::pas]
    dates=years["time"].interpolate().to_numpy()[::pas]
    return tag_column,dates


#Affiche le spectre en bilatéral
def spectrum_bilateral(tag_column,pas=1):
    n=len(tag_column)
    amp=np.abs(np.fft.fft(tag_column))/n
    freq =np.fft.fftfreq(n,d=pas)

    return freq, amp


#Affiche le spectre en unilatéral
def spectrum_unilateral(tag_column,pas=1):
    n=len(tag_column)
    amp=np.abs(np.fft.fft(tag_column))/n
    freq =np.fft.fftfreq(n,d=pas)

    freq = np.abs(freq[:len(freq)//2+1])
    amp = amp[:len(amp)//2+1]
    amp[1:-1] *= 2

    return freq, amp



def reconstruct_signal(tag_column, nb_harmonique):
    Amp = np.fft.rfft(tag_column)
    idx = np.argsort(np.abs(Amp[1:]))[-nb_harmonique:] + 1
    amp_filt = np.zeros_like(Amp)
    amp_filt[0] = Amp[0]
    amp_filt[idx] = Amp[idx]
    signal_reconstruit = np.fft.irfft(amp_filt, len(tag_column))
    return signal_reconstruit


def auto_correlate(tag_collum,pas=1):
    centrered = tag_collum - np.mean(tag_collum)
    correlation= np.correlate(centrered,centrered,mode="full")[len(tag_collum)-1:]
    axe_t = np.arange(len(correlation))*pas
    return correlation,axe_t

def get_max_value(tag_column):
    return np.max(tag_column)

def get_min_value(tag_column):
    return np.min(tag_column)

def get_mean_value(tag_column):
    return np.mean(tag_column)

def get_variance_value(tag_column):
    return np.var(tag_column)

def get_ecart_type_value(tag_column):
    return np.std(tag_column) 

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

#lecture
list_Tmax,axe_t=new_signal("tmax",1)

#Q1:
plt.figure('représentation temporelle de Tmax')
plt.grid()
plt.xlabel("Temps (année)")
plt.ylabel("T max (°C)")
plt.plot(axe_t,list_Tmax)
plt.savefig("Q1_reprsentation_temporelle.png")
plt.show()

print(f"Tmax maximale : {get_max_value(list_Tmax)} °C")
print(f"Tmax minimale : {get_min_value(list_Tmax)} °C")
print(f"Tmax moyenne : {get_mean_value(list_Tmax)} °C")
print(f"Variance pour Tmax : {get_variance_value(list_Tmax)} °C²")
print(f"Ecart type pour Tmax : {get_ecart_type_value(list_Tmax)}°C")

#Q2
ig, (ax1,ax2) = plt.subplots(2,1,constrained_layout=True,num="spectre de Tmax",sharey=True)
freq, amp=spectrum_unilateral(list_Tmax)
ax1.stem(freq,amp,markerfmt="x",label="signal sans zoom")
ax2.stem(freq,amp,markerfmt="x",label="signal avec un zoom")

ax2.set_xlim(-0.0001,0.02)
ax1.set_xlabel("fréquences du signal (1/j)")
ax2.set_xlabel("fréquences du signal (1/j)")
ax2.set_ylabel("amplitudes")

ax1.grid(which="both")
ax2.grid(which="both")
ax1.legend()
ax2.legend()

plt.savefig("Q2_spectre_signal.png")
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(3,1,constrained_layout=True,num="signaux recontruits avec n harmoniques",sharex=True,sharey=True)

reconstr5=reconstruct_signal(list_Tmax,3)
ax1.plot(axe_t,list_Tmax,alpha=0.5,label="signal d'origine")
ax1.plot(axe_t,reconstr5,color="#9932CC",label="nombre d'harmoniques=3")

reconstr10=reconstruct_signal(list_Tmax,10)
ax2.plot(axe_t,list_Tmax,alpha=0.5,label="signal d'origine")
ax2.plot(axe_t,reconstr10,color='green',label="nombre d'harmoniques=10")

reconstr30=reconstruct_signal(list_Tmax,30)
ax3.plot(axe_t,list_Tmax,alpha=0.5,label="signal d'origine")
ax3.plot(axe_t,reconstr30,color="#FF7F50",label="nombre d'harmoniques=30")

plt.xlabel("Temps (année)")
plt.ylabel("T max (°C)")

ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax1.legend()
ax2.legend()
ax3.legend()
plt.savefig("Q2_reconstructions_signal_Tmax.png")
plt.show()


#Q3:
plt.figure("représentation de signaux sous échantillonnés")
plt.plot(axe_t,list_Tmax,alpha=0.5,label="Te=1")

list_sous100,axe_T100=new_signal("tmax",100)
plt.plot(axe_T100,list_sous100,linewidth=3,color="#9932CC",label="Te=100")

list_sous365,axe_T365=new_signal("tmax",365)
plt.plot(axe_T365,list_sous365,linewidth=3,color='green',label="Te=365")

list_sous500,axe_T500=new_signal("tmax",500)
plt.plot(axe_T500,list_sous500,linewidth=3,color="#FF7F50",label="Te=500")

plt.xlabel("Temps (année)")
plt.ylabel("T max selon l'échantillonnage (°C)")

plt.legend()
plt.savefig("Q3_sous_echantionnage_temporel.png")
plt.show()

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,constrained_layout=True,num="spectre selon le sous-échantionnage du signal",sharex=True,sharey=True)
freq1,spectre1 =spectrum_unilateral(list_Tmax)
ax1.set_ylabel("Amplitudes")
ax1.axvline(1/365, color="r", ls="--",label="1/365j")
ax1.stem(freq1, spectre1, markerfmt="x",label="signal avec Te=1")

freq100,spectre100 =spectrum_unilateral(list_sous100,100)
ax2.set_ylabel("Amplitudes")
ax2.axvline(1/365, color="r", ls="--",label="1/365j")
ax2.stem(freq100, spectre100, markerfmt="x",label="signal avec Te=100")

freq365,spectre365 =spectrum_unilateral(list_sous365,365)
ax3.set_ylabel("Amplitudes")
ax3.axvline(1/365, color="r", ls="--",label="1/365j")
ax3.stem(freq365, spectre365, markerfmt="x",label="signal avec Te=365")

freq500,spectre500 =spectrum_unilateral(list_sous500,500)
ax4.set_xlim(-0.0001,0.02)
ax4.set_ylim(-0.1,20)
ax4.axvline(1/365, color="r", ls="--")
ax4.stem(freq500, spectre500, markerfmt="x",label="signal avec Te=500")

ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.savefig("Q3_spectre_sous_echantionnage_temporel.png")
plt.show()

#Q4
signal_FiltrePB0 = filtrePB(list_Tmax, 0.001)
signal_FiltrePB1 = filtrePB(list_Tmax, 0.003)
signal_FiltrePB2 = filtrePB(list_Tmax, 0.006)

freq, amp = spectrum_unilateral(list_Tmax)
freq0, amp0 = spectrum_unilateral(signal_FiltrePB0)
freq1, amp1 = spectrum_unilateral(signal_FiltrePB1)
freq2, amp2 = spectrum_unilateral(signal_FiltrePB2)
freq_max = np.max(freq)

fig, (ax1, ax2,ax3) = plt.subplots(3,1,constrained_layout=True,num="représentation temporelle après  filtrage passe bas")
ax1.plot(axe_t, list_Tmax,alpha=0.5,label="signal non filtré")
# ax1.plot(axe_t, SerieF(temp))

ax1.plot(axe_t, list_Tmax,alpha=0.5,label="signal non filtré")
ax3.plot(axe_t, signal_FiltrePB2, label="signal filtré avec Fc à 0.006Hz", color="b")

ax2.plot(axe_t, list_Tmax,alpha=0.5,label="signal non filtré")
ax2.plot(axe_t, signal_FiltrePB1, label="signal filtré avec Fc à 0.003Hz", color="orange")

ax3.plot(axe_t, list_Tmax,alpha=0.5,label="signal non filtré")
ax1.plot(axe_t, signal_FiltrePB0, label="signal filtré avec Fc à 0.001Hz", color="g")

ax3.set_ylabel("T max (°C)")
ax3.set_xlabel("Temps (année)")

ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax1.legend()
ax2.legend()
ax3.legend()
plt.savefig("Q4_filtrage_temporel.png")
plt.show()

fig, (ax1, ax2,ax3,ax4) = plt.subplots(4,1,constrained_layout=True,num="représentation spectrale après un filtrage passe bas",sharex=True,sharey=True)

ax1.stem(freq, amp, markerfmt="x",label="signal non filtré")
ax2.stem(freq2, amp2, markerfmt="x",label="signal filtré avec Fc à 0.006Hz")
ax3.stem(freq1, amp1, markerfmt="x",label="signal filtré avec Fc à 0.003Hz")
ax4.stem(freq0, amp0, markerfmt="x",label="signal filtré avec Fc à 0.001Hz")

ax4.set_xlim(-0.0001,0.02)

ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.savefig("Q4_filtrage_spectre.png")
plt.show()

#Q5
plt.figure("Autocorrélation de Tmax")
plt.grid()
plt.xlabel("Décalage (jours)")
plt.ylabel("Autocorrélation ")
corelation_Tmax,lags=auto_correlate(list_Tmax)
plt.plot(lags,corelation_Tmax / corelation_Tmax[0])
plt.savefig("Q5_corelation.png")
plt.show()
