# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:16:26 2023

@author: JDION
"""


"""
Création de la classe Signal utilisée durant l'intégralité de ce cours
"""


import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

class Signal:
    
    
    def __init__(self, function, sampling_frequency):
        
        """
        Initialise une instance de la classe Signal.
    
        Args:
            function (callable): Une fonction qui définit le comportement du signal. La fonction doit accepter un argument (le temps) et retourner la valeur du signal à cet instant.
            sampling_frequency (float): La fréquence d'échantillonnage du signal en hertz.
    
        Returns:
            None
    
        Example:
            Pour créer une instance de Signal avec une fonction sinus et une fréquence d'échantillonnage de 1000 Hz :
            
            signal = Signal(np.sin, 1000)
        """
        
        
        self.function = function
        self.sampling_frequency = sampling_frequency
        
        
    def __add__(self, other):
        if self.sampling_frequency != other.sampling_frequency:
            raise ValueError("Les signaux doivent avoir la même fréquence d'échantillonnage.")

        def combined_function(t):
            return self.evaluate(t) + other.evaluate(t)

        return Signal(combined_function, self.sampling_frequency)


    def evaluate(self, t):
        
        """
        Évalue le signal à un instant donné.
    
        Args:
            t (float): Le temps auquel évaluer le signal en secondes.
    
        Returns:
            float: La valeur du signal à l'instant t.
            
        Example:
            Pour évaluer le signal à t = 2.5 secondes :
            
            value = signal.evaluate(2.5)
        """
        
        return self.function(t)
    
    
    
    def samples(self, start_time, end_time):
        
        """
        Échantillonne le signal dans la plage de temps spécifiée.

        Args:
            start_time (float): Le temps de début de l'échantillonnage en secondes.
            end_time (float): Le temps de fin de l'échantillonnage en secondes.

        Returns:
            list: Une liste des échantillons du signal dans la plage de temps spécifiée.
        
        Example:
            Pour échantillonner le signal entre 0 et 10 secondes :
            
            signal.samples(0, 10)
        """
        
        t = np.arange(start_time, end_time, 1 / self.sampling_frequency)
        return [self.evaluate(ti) for ti in t]
    

    def plot(self, start_time, end_time, sampled = False):
        
        """
        Affiche la représentation graphique du signal sur une plage de temps donnée.
    
        Args:
            start_time (float): Le temps de début de la représentation en secondes.
            end_time (float): Le temps de fin de la représentation en secondes.
            sampled (bool, optional): Si True, affiche une représentation échantillonnée (discrétisée en temps) du signal.
    
        Returns:
            None
    
        Notes:
            Cette méthode génère un graphique affichant le signal dans la plage de temps spécifiée. Le graphique inclut des étiquettes d'axe pour le temps et l'amplitude, ainsi qu'un titre "Représentation du signal". La méthode affiche également une grille pour une meilleure lisibilité.
        
        Example:
            Pour afficher le signal entre 0 et 10 secondes :
            
            signal.plot(0, 10)
        """    
        
        t = np.arange(start_time, end_time, 1 / self.sampling_frequency)
        y = self.samples(start_time, end_time)

        plt.figure(figsize=(10, 4))
        if sampled:
            plt.plot(t, y, 'b+')
        else:
            plt.plot(t, y)
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.title('Représentation du signal')
        plt.grid(True)
        plt.show()
        

    def spectrum(self, end_freq, acquisition_duration, unilateral = False, fourier_transform = False):
        
        """
        Affiche le spectre du signal.
    
        Args:
            acquisition_duration (float): La durée d'acquisition du signal en secondes (doit être assez importante pour éviter toute déformation du spectre)
            end_freq (float): La fréquence maximale à afficher.
            unilateral (bool, optional): Si True, affiche un spectre unilatéral, sinon affiche un spectre bilatéral (par défaut)
            fourier_transform (bool, optional): Si True, affiche un la transformée de Fourier, sinon affiche les coefficients de Fourier (par défaut)
    
        Returns:
            None
            
        Example:
            Pour afficher le spectre bilatéral du signal sur une durée d'acquisition de 10 secondes et une fréquence maximale de 3 Hz :
            
            signal.spectrum(end_freq=3, acquisition_duration=10)
            
            Pour afficher le spectre unilatéral du signal sur une durée d'acquisition de 2 secondes (spectre jusqu'à 20 Hz) :
            
            signal.spectrum(end_freq=20, acquisition_duration=2, unilateral=True)
            
            Pour afficher la transformée de Fourier du signal sur une durée d'acquisition de 100 secondes (spectre jusqu'à 10 Hz) :
            
            signal.spectrum(end_freq=10, acquisition_duration=100, fourier_transform=True)
        """
        
        # Calcul du nombre d'échantillons requis pour la durée d'acquisition spécifiée
        num_samples = int(acquisition_duration * self.sampling_frequency)
        
        # Création de la séquence des valeurs du signal sur la durée d'acquisition spécifiée
        t = np.linspace(-acquisition_duration / 2, acquisition_duration / 2, num_samples) ### MODIF : acquérir sur -duree/2, duree/2
        y = [self.evaluate(ti) for ti in t]

        # Calcul des fréquences correspondantes et du spectre du signal à l'aide de l'algorithme de FFT (Fast Fourier Transform)
        freqs = np.fft.fftfreq(num_samples, 1 / self.sampling_frequency)
        
        if fourier_transform:
            # Calcul de la transformée de Fourier
            spectrum = np.abs(np.fft.fft(y)) * acquisition_duration ### MODIF : enlever la multiplication par 2
        else:
            # Calcul du module coefficients de Fourier
            spectrum = np.abs(np.fft.fft(y))
        
        # Normalisation de l'amplitude en fonction de la durée d'acquisition
        spectrum /= num_samples
        
        if unilateral:
            # Doubler l'amplitude (sauf la composante continue)
            spectrum[1:] *= 2

            # Sélection des indices correspondant à l'intervalle [0, end_freq]
            end_index = int(end_freq * acquisition_duration)
            freqs = freqs[0:end_index]
            spectrum = spectrum[0:end_index]
        
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, spectrum)
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude du spectre')
        
        if unilateral:
            plt.title('Spectre unilatéral du signal')
        else:
            plt.title('Spectre bilatéral du signal')
            plt.axis(xmin = -end_freq, xmax = end_freq)
            
        plt.grid(True)
        plt.show()
                
        
    def convolution(self, other_signal, start_time, end_time):
        
        """
        Effectue une convolution entre le signal actuel et un autre signal donné.

        Args:
            other_signal (Signal): L'autre signal à convoluer avec le signal actuel.
            start_time (float): Le temps de début de la convolution en secondes.
            end_time (float): Le temps de fin de la convolution en secondes.

        Returns:
            Signal: Un nouveau signal représentant le résultat de la convolution.
        
        Example:
            Pour effectuer une convolution entre signal1 et signal2 de 0 à 100 secondes :
            
            signal1.convolution(signal2, 0, 100)
        """
        self_samples = self.samples(start_time, end_time)
        other_samples = other_signal.samples(start_time, end_time)
        
        # Convolution entre les deux signaux 
        convolution_result = np.convolve(self_samples, other_samples, mode='same') / self.sampling_frequency # mode='same' signifie que la longueur de la sortie sera la même que celle des entrées ; / self.sampling_frequency normalise le résultat pour maintenir l'unité de mesure
        
        t = np.linspace(start_time, end_time, len(convolution_result), endpoint=False)
        return Signal(lambda x: np.interp(x, t, convolution_result), self.sampling_frequency) # création d'un signal à partir d'une fonction lambda qui prend un argument x et renvoie la valeur interpolée à partir du tableau convolution_result en utilisant le tableau t
        
    
    def play(self, duration):
        
        """
        Joue le son associé au signal pendant la durée spécifiée.

        Args:
            duration (float): La durée en secondes pendant laquelle jouer le son.

        Returns:
            None
        """

        sd.play(self.samples(0, duration), self.sampling_frequency)
        sd.wait()
        
    
