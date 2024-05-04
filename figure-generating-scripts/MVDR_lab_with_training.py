import matplotlib.pyplot as plt
import numpy as np

# Array params
center_freq = 3.3e9
d = 0.045 * center_freq / 3e8 #0.5
print("d:", d)

# Load "training data" which is just A and B, then calc Rinv
filename = '/mnt/c/Users/marclichtman/Downloads/3p3G_A_B.npy'
X_A_B = np.load(filename)
Nr = X_A_B.shape[0]
R_training = X_A_B @ X_A_B.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
Rinv_training = np.linalg.pinv(R_training) # pseudo-inverse tends to work better than a true inverse

print("R_training:", R_training)

# Add Signal C, which is at -19.5 deg
filename = '/mnt/c/Users/marclichtman/Downloads/3p3G_A_B_C.npy'
X = np.load(filename)

# Perform DOA to find angle of arrival of C
theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 10000) # between -90 and +90 degrees
results = []
R = X @ X.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
print("R:", R)
Rinv = np.linalg.pinv(R) # pseudo-inverse tends to work better than a true inverse
for theta_i in theta_scan:
   a = np.exp(-2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta_i)) # steering vector in the desired direction theta_i
   a = a.reshape(-1,1) # make into a column vector
   power = 1/(a.conj().T @ Rinv @ a).squeeze() # MVDR power equation
   power_dB = 10*np.log10(np.abs(power)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize to 0 dB at peak

# Pull out angle of C, after zeroing out the angles that include the interferers
results_temp = np.copy(results)
results_temp[int(len(results)*0.4):] = -9999*np.ones(int(len(results)*0.6))
max_angle = theta_scan[np.argmax(results_temp)] # radians
print("max_angle:", max_angle)

# Calc MVDR weights using training Rinv
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # steering vector in the desired direction theta
s = s.reshape(-1,1) # make into a column vector (size 3x1)
w = (Rinv_training @ s)/(s.conj().T @ Rinv_training @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
w = w.squeeze() # length 16

# Calc beam pattern
w = np.conj(w) # or else our answer will be negative/inverted
N_fft = 2048
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

# Plot beam pattern and DOA results
plt.plot(theta_bins * 180 / np.pi, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
plt.plot(theta_scan * 180 / np.pi, results, 'r')
plt.vlines(ymax=np.max(results), ymin=np.min(results) , x=max_angle*180/np.pi, color='g', linestyle='--')
plt.xlabel("Angle [deg]")
plt.ylabel("Magnitude [dB]")
plt.title("Beam Pattern and DOA Results, With Training")
plt.grid()
plt.savefig("../_images/DOA_with_training.svg", bbox_inches='tight')
plt.show()
