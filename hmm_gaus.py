import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as cst
import matplotlib
matplotlib.rcParams['font.size'] = 15


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

modeNumber_list = [0, 1, 2, 3]

chroma = 0.0 # Qp
chromaShift = chroma * frev / eta  # chromatic frequency shift

nSideband = int(np.floor((1E10/frev))) # not well understood, why 1e10?? gia na einai se GHz?
sidebands = np.arange(-nSideband, nSideband+0.5)


fig, ax = plt.subplots()

for modeNumber in modeNumber_list:
    # create array with frequencies, all the sidebands around the betatron frequency + the mode number *Qs (why the last term?)
    freqs = frev * (0.18 + sidebands + modeNumber * Qs) #--> not sure if 0.18 or mode_number*Qs are necessary
    #freqs = frev * sidebands
    hs = hmm_gauss(freqs - chromaShift, tau, m=modeNumber)  # mode spectrum
    ax.plot(freqs/1e9, hs, label=f'm={modeNumber}')

# place a text box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.95, r'$\mathrm{f_{Q^\prime}}=$'+f'{chromaShift/1e9:.3f} GHz', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
#ax.vlines(7000/1e9, 0, 2)
# styling
ax.set_ylabel(r'$\mathrm{h(f-f_{Q^\prime}) \ [a.u.]}$')
ax.set_xlabel('Frequency [GHz]')
ax.set_xlim(-2, 2)
ax.legend(loc=1)
ax.grid(linestyle='dashed')
plt.tight_layout()



#plt.savefig(f'hmm_gaus_chroma{chroma}.png', bbox='tight')
plt.show()