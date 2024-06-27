import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Author: Sam Brown
# Date: 5/26/2024

##### BPSK Generation #####

num_samples = 10000 # Number of samples to simulate

def pskGen(sps, fc=0, M=2, pulse='rect', beta=.25, Ns=10):
    # Generate random bits
    num_symb = int(np.ceil(num_samples/sps))
    num_bits = int(num_symb*np.log2(M))
    bits = np.random.randint(0, 2, num_bits)
    bits = np.reshape(bits, (int(np.log2(M)), num_symb))
    
    # MPSK modulation
    symbols = np.exp(2j*np.pi*np.polyval(bits, 2)/M)

    # Upsample
    pcm = np.zeros((num_symb, sps), dtype=complex)
    pcm[:, 0] = symbols
    pcm = pcm.flatten()

    # Pulse Shaping
    if pulse == 'rect':
        samples = np.convolve(pcm, np.ones(sps), mode='same')
    elif pulse == 'srrc':
        t_vals = np.arange(-Ns*sps/2, Ns*sps/2, 1)
        h = np.zeros(t_vals.shape)
        for i, t in enumerate(t_vals):
            if t == 0:
                h[i] = 1 + beta*(4/np.pi - 1)
            elif abs(t) == sps/(4*beta):
                h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta)) + 
                                          (1-2/np.pi)*np.cos(np.pi/(4*beta)))
            else:
                h[i] = (np.sin(np.pi*t/sps*(1-beta)) + 
                        4*beta*t/sps*np.cos(np.pi*t/sps*(1+beta)))/(np.pi*t/sps*(1-(4*beta*t/sps)**2))
        h /= np.sqrt(sps)
        samples = np.convolve(pcm, h, mode='same')
        # samples = np.convolve(pcm, h, mode='same') # If we want raised cosine
        
    # Upconvert samples
    samples = np.multiply(samples[0:num_samples], np.exp(1j*2*np.pi*fc*np.arange(num_samples)))

    return samples

##### Build Signal Components #####

signal = pskGen(sps=10, fc=.3, M=2, pulse='rect') + pskGen(sps=10, fc=.1, M=4, pulse='rect')

##### Add Noise #####

SNR_dB = 0
p_sig = np.mean(np.abs(signal)**2)
noise = np.random.randn(num_samples) * np.sqrt(p_sig*10**(-SNR_dB/10)/2) + 1j*np.random.randn(num_samples) * np.sqrt(p_sig*10**(-SNR_dB/10)/2)
signal += noise

plt.figure(figsize=(8, 5))
freq_vals = np.linspace(-0.5, 0.5, num_samples)
signal_fft = np.fft.fftshift(np.fft.fft(signal))
# plt.subplot(1, 3, 3)
plt.plot(freq_vals, 20*np.log10(np.abs(signal_fft)))
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
max_val = np.max(20*np.log10(np.abs(signal_fft)))
plt.ylim([max_val-60, max_val+10])
plt.grid()

plt.title("Transmitted Waveform Spectrum")
plt.tight_layout()



##### Generate the Spectral Correlation Function #####

a_res = 0.00025
a_vals = np.arange(-1, 1, a_res)
smoothing_len = 256
window = np.hanning(smoothing_len)

X = np.fft.fft(signal)
X = np.fft.fftshift(X)

SCF = np.zeros((len(a_vals), num_samples), dtype=complex)
SCF_conj = np.zeros((len(a_vals), num_samples), dtype=complex)

for i, a in enumerate(a_vals):
    SCF[i, :] = np.roll(X, -int(np.round(a*num_samples/2)))*np.conj(np.roll(X, int(np.round(a*num_samples/2))))
    # SCF[i, :abs(round(a*num_samples/2))] = 0
    # SCF[i, -abs(round(a*num_samples/2))-1:] = 0
    SCF[i, :] = np.convolve(SCF[i, :], window, mode='same')
    
    SCF_conj_slice = np.roll(X, int(np.round(a*num_samples/2))-1)*np.flip(np.roll(X, int(np.round(a*num_samples/2))))
    SCF_conj_slice[:abs(round(a*num_samples/2))] = 0
    SCF_conj_slice[-abs(round(a*num_samples/2))-1:] = 0
    SCF_conj[i, :] = np.convolve(SCF_conj_slice, window, mode='same')

'''
plt.figure()
symb_rate = 0.1
a_ind = np.where((np.mod(a_vals, symb_rate) < a_res/2) & (abs(a_vals) < 3*symb_rate))
for i in a_ind[0]:
    plt.plot(freq_vals, 10*np.log10(np.abs(SCF[i, :])))
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
max_val = np.max(10*np.log10(np.abs(SCF[a_ind[0], :])))
plt.ylim([max_val-60, max_val+10])
plt.grid()
plt.legend(["a = " + str(round(a_vals[i], 3)) for i in a_ind[0]])
plt.title("Spectral Correlation Function Slices")
'''
# plt.figure()
# plt.plot(10*np.log10(SCF_conj[int(len(a_vals)*.5), :]))


dym_range_dB = 20
max_val = np.max(np.abs(SCF[np.where(a_vals > a_res),:]))
linear_scale = True

plt.set_cmap("viridis")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
if linear_scale:
    plt.imshow(np.abs(SCF), aspect='auto', extent=(-0.5, 0.5, -1, 1), vmax=max_val)
else:
    plt.imshow(10*np.log10(np.abs(SCF)), aspect='auto', extent=(-0.5, 0.5, -1, 1),
            vmax=10*np.log10(max_val), vmin=10*np.log10(max_val)-dym_range_dB)

plt.ylim([0, 0.5])
plt.xlabel("Normalized Frequency")
plt.ylabel("Cycle Frequency")
plt.colorbar()
plt.title("Non-Conjugate SCF")

max_val = np.max(np.abs(SCF_conj))

plt.subplot(1, 2, 2)
if linear_scale:
    plt.imshow(np.abs(SCF_conj), aspect='auto', extent=[-0.5, 0.5, -1, 1],
           vmax=max_val)
else:
    plt.imshow(10*np.log10(np.abs(SCF_conj)), aspect='auto', extent=[-0.5, 0.5, -1, 1], 
            vmax=10*np.log10(max_val), vmin=10*np.log10(max_val)-dym_range_dB)
plt.xlabel("Normalized Frequency")
plt.ylabel("Cycle Frequency")
plt.ylim([-1, 1])
plt.colorbar()
plt.title("Conjugate SCF")
plt.tight_layout()

plt.show()