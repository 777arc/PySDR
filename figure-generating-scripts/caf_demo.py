import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Author: Sam Brown
# Date: 5/28/2024

##### BPSK Generation #####

num_samples = 100 # Number of samples to simulate

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

signal = pskGen(sps=10, fc=0.05, M=2, pulse='srrc')

##### Add Noise #####

SNR_dB = 50
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

##### Generate the Autocorrelation Function #####

tau_vals = np.arange(-num_samples/2, num_samples/2, 1)
t_vals = np.arange(-num_samples/2, num_samples/2, 1)

acf = np.zeros((num_samples, num_samples), dtype=complex)


for i, tau in enumerate(tau_vals):
    acf[i, :] = np.sum(np.multiply(np.roll(signal, -int(np.floor(tau/2))-1), np.roll(np.conj(signal), int(np.floor(tau/2)))))
    
    
    
    # plt.figure()
    # plt.plot(t_vals, np.abs(acf[i, :]))
    # plt.show()
    

    

plt.figure()
plt.plot(t_vals, np.abs(acf[:, 50]))
plt.show()

dym_range_dB = 20
max_val = np.max(abs(acf))
linear_scale = False

plt.set_cmap("viridis")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
if linear_scale:
    plt.imshow(np.abs(acf), aspect='auto', 
           vmax=max_val)
else:
    plt.imshow(10*np.log10(np.abs(acf)), aspect='auto', 
            vmax=10*np.log10(max_val), vmin=10*np.log10(max_val)-dym_range_dB)

plt.xlabel("Time ($t$)")
plt.ylabel("Time Delay ($\\tau$)")
plt.colorbar()
plt.title("Auto-Correlation Function")



plt.show()