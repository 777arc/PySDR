import numpy as np
import matplotlib.pyplot as plt
import imageio

# THE OUTPUT HAS BEEN MANUALLY PUT THROUGH COMPRESSION! using https://ezgif.com/optimize

sample_rate = 1e6
carrier_freq = sample_rate * 0.015
data_signal_freq = sample_rate * 0.0015
samples_per_frame = 500
samples_added_per_frame = 5
frames = 134
total_samples = samples_per_frame + (samples_added_per_frame * frames)

# Had to switch to generating the entire time at once or else it causes weird aliasing and sine wave issues

t = np.arange(0, total_samples) / sample_rate

carrier = np.cos(2 * np.pi * carrier_freq * t)
data_signal = np.cos(2 * np.pi * data_signal_freq * t)
am_signal = carrier * (data_signal + 1)

k = 0.5 # sensitivity
phi = 2 * np.pi * carrier_freq * t + k * np.cumsum(data_signal + 1) # phase
fm_signal = np.cos(phi) # modulated signal


sample_indx = 0
filenames = []
for i in range(frames):


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 7))
    fig.subplots_adjust(hspace=0)
    ax1.plot(carrier[sample_indx:sample_indx+samples_per_frame], 'g-')
    ax1.set_ylabel('Carrier')
    ax1.set_ylim([-1.5, 1.5])

    ax2.plot(data_signal[sample_indx:sample_indx+samples_per_frame], 'k-')
    ax2.set_ylabel('Data Signal')
    ax2.set_ylim([-1.5, 1.5])

    ax3.plot(am_signal[sample_indx:sample_indx+samples_per_frame], 'r-')
    ax3.set_ylabel('AM')
    ax3.set_ylim([-2.5, 2.5])
    
    ax4.plot(fm_signal[sample_indx:sample_indx+samples_per_frame], 'b-')
    ax4.set_ylabel('FM')
    ax4.set_ylim([-1.5, 1.5])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax1.yaxis.label.set_size(15)
    ax2.yaxis.label.set_size(15)
    ax3.yaxis.label.set_size(15)
    ax4.yaxis.label.set_size(15)

    #plt.show()

    filename = '/tmp/am_fm_' + str(i) + '.png'
    print(i)
    fig.savefig(filename, bbox_inches='tight')
    filenames.append(filename)
    plt.close(fig)

    sample_indx += samples_added_per_frame


# Create animated gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/tmp/am_fm_animation.gif', images, fps=20)
