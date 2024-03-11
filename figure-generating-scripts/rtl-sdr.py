from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
print(sdr.sample_rate)
sdr.center_freq = 100e6   # Hz
sdr.freq_correction = 60  # PPM
print(sdr.valid_gains_db)
sdr.gain = 49.6
print(sdr.gain)

if False:
    x = sdr.read_samples(2048) # get rid of initial empty samples

    samples = np.zeros(0, np.complex128)
    for i in range(4):
        x = sdr.read_samples(512*10)
        samples = np.concatenate((samples, x)) # add the new samples to the end of the array
        if i % 2 == 0:
            sdr.center_freq = 70e6
        else:
            sdr.center_freq = 100e6 # try to find one frequency with lots of power, and another with little power
    sdr.close()
    plt.plot(samples.real)
    #plt.savefig("../_images/rtlsdr-retuning.svg", bbox_inches='tight')
    plt.show()
    exit()

if False:
    x = sdr.read_samples(4096)
    sdr.close()
    plt.plot(x.real)
    plt.plot(x.imag)
    plt.legend(["I", "Q"])
    #plt.savefig("../_images/rtlsdr-gain.svg", bbox_inches='tight')
    plt.show()
    exit()

if True:
    fft_size = 512
    num_rows = 500
    x = sdr.read_samples(2048) # get rid of initial empty samples
    x = sdr.read_samples(fft_size*num_rows) # get all the samples we need for the spectrogram

    print(np.max(x))

    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

    extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
              (sdr.center_freq + sdr.sample_rate/2)/1e6,
              len(x)/sdr.sample_rate, 0]
    plt.imshow(spectrogram, aspect='auto', extent=extent)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    #plt.savefig("../_images/rtlsdr-waterfall.svg", bbox_inches='tight')
    plt.show()

sdr.close()