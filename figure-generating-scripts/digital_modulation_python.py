# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
### No noise
num_symbols = 1000

x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols

fig1, (_ax1) = plt.subplots(1, 1, figsize=(4, 4))
fig1.subplots_adjust(hspace=0.4)
_ax1.plot(np.real(x_symbols), np.imag(x_symbols), '.')
_ax1.set_ylabel("Q")
_ax1.set_xlabel("I")
_ax1.set_ylim(bottom=-1, top=1)
_ax1.set_xlim(left=-1, right=1)
_ax1.grid()
#if os.path.isdir('../_images'):
#    fig1.savefig('../_images/qpsk_python.svg', bbox_inches='tight')
fig1

# %%
### Add AWGN
n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
noise_power = 0.01
_r = x_symbols + n * np.sqrt(noise_power)

fig2, (_ax1) = plt.subplots(1, 1, figsize=(4, 4))
fig2.subplots_adjust(hspace=0.4)
_ax1.plot(np.real(_r), np.imag(_r), '.')
_ax1.set_ylabel("Q")
_ax1.set_xlabel("I")
_ax1.set_ylim(bottom=-1, top=1)
_ax1.set_xlim(left=-1, right=1)
_ax1.grid()
#if os.path.isdir('../_images'):
#    fig2.savefig('../_images/qpsk_python2.svg', bbox_inches='tight')
fig2

# %%
### Phase noise
_phase_noise = np.random.randn(len(x_symbols)) * 0.15 # adjust multiplier for "strength" of phase noise
_r = x_symbols * np.exp(1j*_phase_noise)

fig3, (_ax1) = plt.subplots(1, 1, figsize=(4, 4))
fig3.subplots_adjust(hspace=0.4)
_ax1.plot(np.real(_r), np.imag(_r), '.')
_ax1.set_ylabel("Q")
_ax1.set_xlabel("I")
_ax1.set_ylim(bottom=-1, top=1)
_ax1.set_xlim(left=-1, right=1)
_ax1.grid()
#if os.path.isdir('../_images'):
#    fig3.savefig('../_images/phase_jitter.svg', bbox_inches='tight')
fig3

# %%
### Phase noise plus AWGN
_phase_noise = np.random.randn(len(x_symbols)) * 0.2 # adjust multiplier for "strength" of phase noise
_r = x_symbols * np.exp(1j*_phase_noise) + n * np.sqrt(noise_power)

fig4, (_ax1) = plt.subplots(1, 1, figsize=(4, 4))
fig4.subplots_adjust(hspace=0.4)
_ax1.plot(np.real(_r), np.imag(_r), '.')
_ax1.set_ylabel("Q")
_ax1.set_xlabel("I")
_ax1.set_ylim(bottom=-1.2, top=1.2)
_ax1.set_xlim(left=-1.2, right=1.2)
_ax1.grid()
#if os.path.isdir('../_images'):
#    fig4.savefig('../_images/phase_jitter_awgn.svg', bbox_inches='tight')
fig4

# %%
# Kept in its own cell, otherwise marimo would display the figures twice
plt.show()
