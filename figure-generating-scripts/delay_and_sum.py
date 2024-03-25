import imageio # pip install imageio==2.11.0
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.gridspec import GridSpec
import drawsvg as draw # pip install drawsvg

Nr = 8 # elements
N = 10000 # samples to simulate
sample_rate = 1e6
d = 0.5

num_frames = 60
theta_scan = np.linspace(np.pi/-2, np.pi/2, num_frames)
filenames = []
for frame_i in range(0, num_frames):
    # Simulate received signal signal, high SNR
    t = np.arange(N)/sample_rate
    tx = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    theta = 0 / 180 * np.pi # angle of arrival
    a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1,1) # Nr x 1
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = a @ tx + 0.01*n

    # Decide on the weights
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[frame_i])) # steering vector at theta

    # Theoretical beam shape using FFT
    N_fft = 1024
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

    # Plot
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(6, 8, figure=fig)
    ax = fig.add_subplot(gs[0:5, :], polar=True)

    ax.plot(theta_bins, w_fft_dB)
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_ylim(-40, 0)  # dB scale
    ax.yaxis.set_ticks(np.arange(-40, 10, 10))
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.text(2.3, -34, 'dB')


    for i in range(Nr):
        ax2 = fig.add_subplot(gs[5, i])
        ax2.plot([-2, 2], [0, 0], 'k', linewidth=1)
        ax2.plot([0, 0], [-2, 2], 'k', linewidth=1)
        ax2.add_patch(plt.Circle((0, 0), 1, color='lightgrey', fill=False, linestyle='dotted', linewidth=1)) # circle
        ax2.plot([0, w[i].real], [0, w[i].imag], linewidth=2, color='red') # line
        ax2.plot(w[i].real, w[i].imag, color='red', marker='o', markersize=5) # marker
        # Hide border
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.xaxis.set_ticks([])
        ax2.yaxis.set_ticks([])
        ax2.text(-0.1, -1.6, str(i), fontsize=12)
        ax2.grid(False)
        ax2.axis([-1.1, 1.1, -1.1, 1.1])
    ax2.text(-10, -2, "Element", fontsize=12)

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=1.1, wspace=0.2, hspace=-0.2)

    #plt.show()
    #exit()

    filename = '/tmp/delay_and_sum' + str(frame_i) + '.png' # 7kB for beam pattern. text appears to use font, not path
    fig.savefig(filename, pad_inches=0, dpi=75)
    plt.close()
    filenames.append(filename)

time.sleep(2)

# Create animated gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
for filename in reversed(filenames):
    images.append(imageio.imread(filename))

# requires specific version if imageio! see top.  new version doesnt have fps arg, and it doesnt repeat for some reason
imageio.mimsave('../_images/delay_and_sum.gif', images, fps=8)
