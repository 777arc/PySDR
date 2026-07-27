import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6

# Generate tone plus noise
t = np.arange(1024*1000)/sample_rate # time vector
f = 50e3 # freq of tone
x = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))

if False:
    plt.plot(x[0:200])
    plt.xlabel("Time")
    #plt.show()
    plt.savefig('../_images/spectrogram_time.svg', bbox_inches='tight')
    exit()

fft_size = 1024
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

# Time starts at the top and goes down, eg x[0] will be in the top row displayed
plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.savefig('../_images/spectrogram.svg', bbox_inches='tight')
#plt.show()
