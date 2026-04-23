import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

#### STEP 1: Generate PRN for each satellite ####

# Don't get too caught up in how generate_PRN() works, first try to understand the input and output

chip_rate = 1.023e6 # chips / sec (part of the GPS spec)
num_chips = 1023 # chips per C/A code period
gps_svs = list(range(1, 33)) # 1–32 are the main ones, it's possible there are 33-37

##### C/A Code Generation #####
# The GPS C/A code is a Gold code formed by XOR-ing two 10-stage maximal-length shift registers (G1 and G2)
# 1023-chip C/A PRN code for SV sv (1-32) as float32, 1's and -1's, so BPSK
def generate_PRN(sv: int) -> np.ndarray:
    # G1 LFSR: polynomial x^10 + x^3 + 1, all-ones init, output at stage 10
    reg = np.ones(10, dtype=np.int8)
    G1 = np.empty(num_chips, dtype=np.int8)
    for i in range(num_chips):
        G1[i] = reg[9] # output
        fb = reg[2] ^ reg[9] # taps at 3 and 10
        reg = np.roll(reg, 1)
        reg[0] = fb

    # G2 LFSR: polynomial x^10+x^9+x^8+x^6+x^3+x^2+1, all-ones init
    reg = np.ones(10, dtype=np.int8)
    G2 = np.empty(num_chips, dtype=np.int8)
    for i in range(num_chips):
        G2[i] = reg[9] # output
        fb = reg[1]^reg[2]^reg[5]^reg[7]^reg[8]^reg[9]  # taps at 2,3,6,8,9,10
        reg = np.roll(reg, 1)
        reg[0] = fb

    # G2 is delayed by a satellite-specific number of chips, specified in IS-GPS-200, Table 3-Ia
    G2_DELAY = [ # G2 phase delay (chips) for gps_svs 1–32
        5,   6,   7,   8,  17,  18, 139, 140,     #  1–8
        141, 251, 252, 254, 255, 256, 257, 258,   #  9–16
        469, 470, 471, 472, 473, 474, 509, 512,   # 17–24
        513, 514, 515, 516, 859, 860, 861, 862    # 25–32
    ]
    g2_delayed = np.roll(G2, G2_DELAY[sv - 1]) 
    
    bits = G1 ^ g2_delayed # bitwise XOR, still 0s and 1s
    return (1 - 2.0 * bits) # convert to BPSK, +1s and −1s.  At this point it's 1 sample per chip

example_prn = generate_PRN(12)
print(example_prn[0:10])

if False:
    plt.plot(example_prn)
    plt.title("Example PRN Code for SV 12")
    plt.xlabel("Chip Index")
    plt.ylabel("Amplitude")
    plt.show()
    exit()


#### STEP 2: Read in example IQ recording of GPS ####

filename = "/home/marc/PySDR/figure-generating-scripts/GPS_L1_recording_10ms_4MHz_cf32.iq"
sample_rate = 4e6
samples_per_code = int(round(sample_rate / chip_rate * num_chips))  # Exact number of samples in one 1 ms code period at 4 MHz
print(f"samples_per_code: {samples_per_code}")
num_integrations = 10 # non-coherent power integrations (so 10 ms total), determines how much of the IQ recording we read in and process!
n_needed = samples_per_code * num_integrations
x = np.fromfile(filename, dtype=np.complex64, count=n_needed)

# At this point, have them plot the spectrogram

#### STEP 3: Try correlating for one of them ####

template = generate_PRN(12)
template_upsampled = resample_poly(template, samples_per_code, num_chips)
corr = np.abs(np.correlate(template_upsampled, x))**2

if False:
    plt.plot(corr)
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Magnitude")
    plt.title("Correlation for SV 12")
    plt.show()
    exit()

# Have them calc the time between spikes

#### Step 4: Loop through all satellites ####

print(" SV        PMR [dB]")
for sv in gps_svs:
    template = generate_PRN(sv)

    # Non-coherent integration: accumulate squared correlation magnitude
    corr_integrated = np.zeros(samples_per_code)
    for k in range(num_integrations):
        x_sub = x[k * samples_per_code:(k + 1) * samples_per_code]
        template_upsampled = resample_poly(template, samples_per_code, num_chips)
        corr = np.correlate(x_sub, template_upsampled, mode='same') / samples_per_code
        corr_integrated += np.abs(corr)**2

    # Normalize by mean and convert to dB
    pmr_db = 10.0 * np.log10(np.max(corr_integrated) / np.mean(corr_integrated))
    print(f"{sv:>3}  {pmr_db:>9.1f} dB")


# TODO plot corr curve of a few satellites to see the difference in delay



