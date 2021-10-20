from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property


class FFTArray:
    def __init__(self, xf, yf):
        self.__xf = xf
        self.__yf = yf
        self.__N = len(xf)
        
    @property
    def xf(self):
        return self.__xf
    
    @property
    def yf(self):
        return self.__yf
    
    @property
    def N(self):
        return self.__N

    @cached_property
    def positive_frequencies(self):
        pos_xf = self.xf[:self.N//2]
        pos_yf = 2.0/self.N * np.abs(self.yf[0:self.N//2])
        return pos_xf, pos_yf

    def plot(self):
        plt.plot(*self.positive_frequencies)
        plt.grid()
        plt.show()


def signal_fft(signal, sample_rate):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, sample_rate)
    return FFTArray(xf, yf)



if __name__ == "__main__":

    # Number of sample points
    N = 600

    # sample spacing
    T = 1.0 / 800.0

    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    
    fft_arr = signal_fft(y, T)
    
    fft_arr.plot()