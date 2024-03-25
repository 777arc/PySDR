import imageio # sudo pip install imageio==2.11.0
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

Nr = 8 # elements
N = 10000 # samples to simulate
sample_rate = 1e6
d = 0.5

num_frames = 10 # for first animnation
#num_frames = 30 # for second animation
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
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(0)) # steering vector at boresight
    tapering = np.random.uniform(0, 1, Nr) # random tapering
    #tapering = np.hamming(Nr) * frame_i / (num_frames - 1) + np.ones(Nr) * (num_frames - frame_i - 1)/(num_frames - 1) # switch between rect and hamming window slowly
    w *= tapering

    '''
    # Conventional DOA
    theta_scan = np.linspace(np.pi/-2, np.pi/2, 1000)
    doa_results_dB = []
    for theta_i in range(len(theta_scan)):
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i]))
        r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
        doa_results_dB.append(10*np.log10(np.mean(np.abs(r_weighted)**2))) # energy detector
    doa_results_dB = doa_results_dB - np.max(doa_results_dB) # normalize to 0 dB at peak
    '''

    # Theoretical beam shape using FFT
    N_fft = 1024
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

    # Plot
    #fig, [ax, ax2, ax3] = plt.subplots(3, 1, subplot_kw={'projection': 'polar'})
    fig = plt.figure()
    gs = GridSpec(3, 3, figure=fig)
    ax = fig.add_subplot(gs[0:2, :], polar=True)

    #ax.plot(theta_scan, doa_results_dB) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.plot(theta_bins, w_fft_dB)
    #ax.legend(['Conventional DOA', 'Theoretical Beam Shape'])
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_ylim(-40, 0)  # dB scale
    ax.yaxis.set_ticks(np.arange(-40, 10, 10))
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.text(2.3, -32.5, 'dB')
    #plt.show()
    #exit()

    # Add text showing n
    #ax.text(np.pi/4, 5, str(frame_i), fontsize=16, color='red')

    # set the x-spine (see below for more info on `set_position`)
    #ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    #ax.spines['right'].set_color('none')
    #ax.yaxis.tick_left()

    # set the y-spine
    #ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    #ax.spines['top'].set_color('none')
    #ax.xaxis.tick_bottom()

    # Turn off tick numbering/labels
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    #ax.axis([min(xdata), max(xdata), -1.5, 1.5])

    # Replace 2nd axis with non-polar plot show taper levels as bars w/ handles
    ax2 = fig.add_subplot(gs[2, :])
    for i in range(Nr):
        ax2.plot([i, i], [0, 1], color='grey', linewidth=1)
    ax2.plot(tapering, linewidth=0, marker='_', markersize=20, markeredgewidth=2, markeredgecolor='red')
    ax2.set_xlabel('Element')
    ax2.set_ylabel('Tapering Level')
    # Hide border
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.set_ticks_position('none') # hides ticks but leaves tick labels
    ax2.axis([-0.5, Nr-0.5, -0.1, 1.1])

    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=1.1, wspace=0.2, hspace=-0.2)
    
    #plt.show()
    #exit()

    filename = '/tmp/spatial_tapering_' + str(frame_i) + '.png'
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    filenames.append(filename)

time.sleep(2)

# Create animated gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

if False: # only use for 2nd animation
    for filename in reversed(filenames):
        images.append(imageio.imread(filename))

# requires specific version if imageio! see top.  new version doesnt have fps arg, and it doesnt repeat for some reason
imageio.mimsave('../_images/spatial_tapering_animation.gif', images, fps=2)
#imageio.mimsave('../_images/spatial_tapering_animation2.gif', images, fps=10)
