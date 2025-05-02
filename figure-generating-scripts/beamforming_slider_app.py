import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

Nr = 8 # Number of elements
N_fft = 1024
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

fig = plt.figure('Beamforming Slider App', figsize=(10, 6))
gs = fig.add_gridspec(Nr + 3, 5, hspace=-0.3, wspace=0.6)
plt.subplots_adjust(left=0, bottom=0, right=0.95, top=1.02)

# We will hold the magnitude and phase values in the slider objects (they will act as our state)
mag_sliders = []
phase_sliders = []
for i in range(Nr):
    mag_sliders.append(Slider(fig.add_subplot(gs[i+1, 1]), '', 0, 1.0, valinit=1, valstep=0.01))
    phase_sliders.append(Slider(fig.add_subplot(gs[i+1, 2]), '', -2*np.pi, 2*np.pi, valinit=0, valstep=0.01))

def beam_pattern():
    w_mags = np.asarray([x.val for x in mag_sliders])
    w_phases = np.asarray([x.val for x in phase_sliders])
    w = w_mags * np.exp(1j * w_phases)
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10((np.abs(np.fft.fftshift(np.fft.fft(w_padded)*Nr))**2)/N_fft) # magnitude of fft in dB
    return w_fft_dB

# Sliders should trigger update
def update(val):
    w_fft_dB = beam_pattern()
    polar_plot.set_ydata(w_fft_dB)
    rect_plot.set_ydata(w_fft_dB)
    fig.canvas.draw_idle()
for i in range(Nr):
    mag_sliders[i].on_changed(update)
    phase_sliders[i].on_changed(update)

for i in range(Nr):
    def update(val):
        w_fft_dB = beam_pattern()
        polar_plot.set_ydata(w_fft_dB)
        rect_plot.set_ydata(w_fft_dB)
        fig.canvas.draw_idle()
    phase_sliders[i].on_changed(update)

ax1 = fig.add_subplot(gs[0:5, 3:5], projection = 'polar')
polar_plot, = ax1.plot(theta_bins, np.zeros(N_fft))
ax1.set_theta_zero_location('N') # type: ignore # make 0 degrees point up
ax1.set_theta_direction(-1) # type: ignore # increase clockwise
ax1.set_thetamin(-90) # type: ignore # only show top half
ax1.set_thetamax(90) # type: ignore
ax1.set_ylim((-30, 10))
ax1.set_yticks(np.arange(-30, 11, 10)) # Only label every 10 dB

ax2 = fig.add_subplot(gs[5:9, 3:5])
rect_plot, = ax2.plot(theta_bins * 180 / np.pi, np.zeros(N_fft))
plt.axis((-90, 90, -40, 10))
plt.xlabel('Theta [Degrees]')
plt.grid()

# Text labels
ax4 = fig.add_subplot(gs[0, 0], frameon=False, xticks=[], yticks=[])
ax4.text(0.5, 0.5, 'Element', horizontalalignment='center', verticalalignment='center', fontsize=12)
for i in range(Nr):
    ax4 = fig.add_subplot(gs[i+1, 0], frameon=False, xticks=[], yticks=[])
    ax4.text(0.5, 0.5, str(i), horizontalalignment='center', verticalalignment='center', fontsize=12)
ax3 = fig.add_subplot(gs[0, 1], frameon=False, xticks=[], yticks=[])
ax3.text(0.5, 0.5, 'Magnitude', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax4 = fig.add_subplot(gs[0, 2], frameon=False, xticks=[], yticks=[])
ax4.text(0.5, 0.5, 'Phase', horizontalalignment='center', verticalalignment='center', fontsize=12)

# Reset button
ax_button = fig.add_subplot(gs[Nr + 2, 1:3])
reset_button = Button(ax_button, 'Reset', color='0.8', hovercolor='0.5')
def button_press(event):
    for i in range(Nr):
        mag_sliders[i].reset()
        phase_sliders[i].reset()
reset_button.on_clicked(button_press)

# Initialize the plot
w_fft_dB = beam_pattern()
polar_plot.set_ydata(w_fft_dB)
rect_plot.set_ydata(w_fft_dB)
fig.canvas.draw_idle()

plt.show() # blocking function
