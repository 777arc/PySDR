import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from matplotlib.animation import FuncAnimation
sample_rate = 1e6
d = 0.5 # half wavelength spacing
N = 1e5 # number of samples to simulate
N = int(N) # convert to int
t_v = np.arange(N)/sample_rate # time vector

# more complex scenario taken from DOA code
Nr = 16 # 8 elements
theta1 = (10 / 180) * np.pi # convert to radians
theta2 = (50 / 180) * np.pi
theta3 = (-60 / 180) * np.pi
s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # Nrx1
s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# we'll use 3 different frequencies
if False:
    soi = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1) # 1xN
else:

    gold_code = np.array([-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1], dtype=complex) # Gold code sequence-127 
    """     gold_code = np.array([-1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1]) # Gold code sequence-31 
    """
     
    soi_samples_per_symbol = 32
    soi = np.repeat(gold_code, soi_samples_per_symbol) # Gold code is 31 bits, so 31*8 = 248 samples
    num_sequence_repeats = int(N / soi.shape[0]) + 1 # number of times to repeat the sequence
    soi = np.tile(soi, num_sequence_repeats) # repeat the sequence
    soi = soi[:N] # trim to N samples
    print(soi.shape)

    soi_in = soi.reshape(1, -1) # 1xN

tone2 = np.exp(2j*np.pi*0.02e6*t_v).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t_v).reshape(1,-1)
d_s_inp = s1 @ soi_in




n_c_s = 0.01 # noise covariance scaling factor
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r_training = s2 @ tone2 + 0.1 * s3 @ tone3
r_training = r_training + n_c_s*n # 8xN

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = d_s_inp + 0.5* s2 @ tone2 + 0.1 * s3 @ tone3
r = r + n_c_s*n # Nr x N


#plt.ion()  # Turn on interactive mode
N_fft = 512
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))

# Initialize with zeros
w_padded = np.zeros(N_fft) + 1e-8  # avoid log(0)
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) 
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

# Initial plot
'''
line1, = ax.plot(theta_bins, w_fft_dB)
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_xticks(np.linspace(-np.pi/2, np.pi/2, 37)) 
ax.set_xticklabels(np.round(np.rad2deg(np.linspace(-np.pi/2, np.pi/2, 37)), 1)) # show degrees
ax.set_xticks(np.linspace(-np.pi/2, np.pi/2, 37), minor=True)
ax.set_ylim([-30, 1])  # Set y-axis limits
ax.grid(True)
ax.set_title('LMS Algorithm Weight Evolution')
'''

# Setup the animation process
w_LMS_current = np.zeros((Nr,1), dtype=complex) # initialize weights to zero


iterations_to_show = N  # Show fewer iterations for demonstration


def w_LMS_calc(d, x_n, w, mu=0.000001): # d is the desired signal, x_n is the input signal, w is the weights vector, mu is the step size
    # LMS algorithm to calculate weights
    y_n = w.conj().T @ x_n # output of the filter
    y_n = y_n.reshape(-1,1) # make into a column vector
    e_n = d - y_n # error signal
    #error_log.append(e_n.squeeze()) # mean square error
    print(e_n)
    w += 2*mu * e_n * np.conj(x_n) # update weights
    return w    



error_log = []
for n in range(0, iterations_to_show): 
    x_n = r[:,n].reshape(-1,1) # make into a column vector 
    w_LMS_current = w_LMS_calc(soi[n], x_n, w_LMS_current, mu=0.0005)
    
    # Prepare data for plotting
    w_soi = w_LMS_current.reshape(-1) # make into a row vector
    w_padded = np.concatenate((w_soi, np.zeros(N_fft - Nr))) # zero pad
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    
    # Update plot data
    #line1.set_ydata(w_fft_dB)
    
    # Update plot title with iteration number
    #ax.set_title(f'LMS Algorithm Weight Evolution - Iteration {n}')
    
    # Draw the updated plot
    #fig.canvas.draw()
    #plt.pause(0.1)  # Short pause to allow plot to update

plt.plot(error_log)
plt.show()
exit()


# Keep the plot window open after animation completes
plt.ioff()
plt.show()
