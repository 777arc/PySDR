import matplotlib.pyplot as plt
import imageio
import numpy as np

# THE OUTPUT HAS BEEN MANUALLY PUT THROUGH COMPRESSION! using https://ezgif.com/optimize

radians_per_frame = 0.1
frames = int(2*np.pi/radians_per_frame)

filenames = []
theta_pos = 0
theta_neg = 0
for i in range(frames):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot([theta_pos, theta_pos], [0, 1], color='blue', linewidth=2)
    ax.plot([theta_neg, theta_neg], [0, 1], color='red', linewidth=2)
    ax.set_rmax(1)
    ax.grid(True)

    ax.legend(['Positive Frequency', 'Negative Frequency'], loc='upper right')

    #plt.show()

    filename = '/tmp/negative_freq_animation' + str(i) + '.png'
    print(i)
    fig.savefig(filename, bbox_inches='tight')
    filenames.append(filename)
    plt.close(fig)

    theta_pos += radians_per_frame
    theta_neg -= radians_per_frame


# Create animated gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/tmp/negative_freq_animation.gif', images, fps=20)
