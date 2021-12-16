import numpy as np
from scipy import constants as cst
import matplotlib.pyplot as plt


# power spectral density of a gaussian bunch, oscillating in different (hermite-gauss) modes, m
def hmm_gauss(freq, tau, m=0):
    return (2.0*np.pi*freq*tau/4)**(2*m) * np.exp(-(2.0*np.pi*freq*tau/4.)**2)

N = 3.5E10
Qx = 26.18
Qs = 0.0051
gamma_t = 22.8
energy = 270.0
gamma = np.sqrt(1+(energy/0.938)**2)
circum = 2*np.pi*1.1E3
frev = cst.c/circum
eta = 1/gamma_t**2 - 1/gamma**2
tau = 1.85E-9 # 4 sigmat
sigmaz = tau*cst.c/4

#### Impedance model from https://gitlab.cern.ch/IRIS/SPS_IW_model/-/tree/master/SPS_IW_model_python ###########################################
impedanceData = np.genfromtxt('./from_Xavier/SPS_Complete_imp_model_2018_Q26.txt', skip_header=1, dtype=complex)  # type: array
freqZ = np.real(impedanceData[:, 0]) * 1E6
ReZ = np.real(impedanceData[:, 2])  # vertical dipolar kick

chroma = 0.0 # Qp
chromaShift = chroma * frev / eta  # chromatic frequency shift

nSideband = int(np.floor((1E10/frev))) # not well understood, why 1e10?? gia na einai se GHz?
sidebands = np.arange(-nSideband, nSideband+0.5)

modeNumber_list = [0, 1, 2, 3]

fig, ax = plt.subplots()

for modeNumber in modeNumber_list:
    freqs = frev * (0.18 + sidebands + modeNumber * Qs)  # --> not sure why/if 0.18 or mode_number*Qs are necessary
    hs = hmm_gauss(freqs - chromaShift, tau, m=modeNumber)  # mode spectrum
    # interpolation of ReZ of the impedance such as to it at frequencies spaced as freqs
    zeffs = np.interp(np.abs(freqs), freqZ, ReZ) * np.sign(freqs) * hs
    zeffs /= np.sum(hs)
    zeff = np.sum(zeffs)
    ax.plot(freqs/1e9, zeffs)

plt.show()
