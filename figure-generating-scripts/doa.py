import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N)/sample_rate
f_tone = 0.02e6
tx = np.exp(2j*np.pi*f_tone*t)

# Simulate three omnidirectional antennas in a line with 1/2 wavelength between adjancent ones, receiving a signal that arrives at an angle

d = 0.5
Nr = 3
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
print(a)

# we have to do a matrix multiplication of a and tx, which currently are both 1D, so we have to make them 2D with reshape
a = a.reshape(-1,1)
print(a.shape) # 3x1
tx = tx.reshape(-1,1)
print(tx.shape) # 10000x1

# so how do we use this? simple:
r = a @ tx.T # matrix multiply. dont get too caught up by the transpose a, the important thing is we're multiplying the array factor by the tx signal
print(r.shape) # 3x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

# Plot the real part of the first 200 samples of all three elements
if False:
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
    ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
    ax1.set_ylabel("Samples")
    ax1.set_xlabel("Time")
    ax1.grid()
    ax1.legend(['0','1','2'], loc=1)
    plt.show()
    #fig.savefig('../_images/doa_time_domain.svg', bbox_inches='tight')
    exit()
# note the phase shifts, they are also there on the imaginary portions of the samples

# So far this has been simulating the recieving of a signal from a certain angle of arrival
# in your typical DOA problem you are given samples and have to estimate the angle of arrival(s)
# there are also problems where you have multiple receives signals from different directions and one is the SOI while another might be a jammer or interferer you have to null out

# One thing we didnt both doing- lets add noise to this recieved signal.
# AWGN with a phase shift applied is still AWGN so we can add it after or before the array factor is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.1*n

if False:
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
    ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
    ax1.set_ylabel("Samples")
    ax1.set_xlabel("Time")
    ax1.grid()
    ax1.legend(['0','1','2'], loc=1)
    plt.show()
    #fig.savefig('../_images/doa_time_domain_with_noise.svg', bbox_inches='tight')
    exit()

# OK lets use this signal r but pretend we don't know which direction the signal is coming in from, lets try to figure it out
# The "conventional" beamforming approach involves scanning through (sampling) all directions from -pi to +pi (-180 to +180) 
# and at each direction we point the array towards that angle by applying the weights associated with pointing in that direction
# which will give us a single 1D array of samples, as if we recieved it with 1 antenna
# we then calc the mean of the magnitude squared as if we were doing an energy detector
# repeat for a ton of different angles and we can see which angle gave us the max

if False:
    # signal from hack-a-sat 4 where we wanted to find the direction of the least energy because there were jammers
    N = 880 # num samples
    r = np.zeros((Nr,N), dtype=np.complex64)
    r[0, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_0.bin', dtype=np.complex64)
    r[1, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_1.bin', dtype=np.complex64)
    r[2, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_2.bin', dtype=np.complex64)


# conventional beamforming
if False:
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        #print(theta_i)
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # look familiar?
        r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i. remember r is 3x10000 so we can leave w as normal (row) vector
        results.append(np.mean(np.abs(r_weighted)**2)) # energy detector

    print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
    ax1.plot([20],[np.max(results)],'r.')
    ax1.text(-5, np.max(results) + 0.7, '20 degrees')
    ax1.set_xlabel("Theta [Degrees]")
    ax1.set_ylabel("DOA Metric")
    ax1.grid()
    plt.show()
    #fig.savefig('../_images/doa_conventional_beamformer.svg', bbox_inches='tight')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rgrids([0,2,4,6,8]) 
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_conventional_beamformer_polar.svg', bbox_inches='tight')

    exit()

# sweeping angle of arrival
if False:
    theta_txs = np.concatenate((np.repeat(-90, 10), np.arange(-90, 90, 2), np.repeat(90, 10)))
    
    theta_scan = np.linspace(-1*np.pi, np.pi, 300)
    results = np.zeros((len(theta_txs), len(theta_scan)))
    for t_i in range(len(theta_txs)):
        print(t_i)

        theta = theta_txs[t_i] / 180 * np.pi
        a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
        a = a.reshape(-1,1) # 3x1
        tone = np.exp(2j*np.pi*0.02e6*t)
        tone = tone.reshape(-1,1) # 10000x1
        r = a @ tone.T

        for theta_i in range(len(theta_scan)):
            w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i]))
            r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
            results[t_i, theta_i]  = np.mean(np.abs(r_weighted)**2) # energy detector

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={'projection': 'polar'})
    fig.set_tight_layout(True)
    line, = ax.plot(theta_scan, results[0,:])
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    text = ax.text(0.4, 12, 'fillmein', fontsize=16)
    text2 = ax.text(np.pi/-2, 19, 'broadside →', fontsize=16)
    text3 = ax.text(np.pi/2, 12, '← broadside', fontsize=16)
    def update(i):
        i = int(i)
        print(i)
        results_i = results[i,:] / np.max(results[i,:]) * 9 # had to add this in for the last animation because it got too large
        line.set_ydata(results_i)
        d_str = str(np.round(theta_txs[i],2))
        text.set_text('AoA = ' + d_str + '°')
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(theta_txs)), interval=100)
    anim.save('../_images/doa_sweeping_angle_animation.gif', dpi=80, writer='imagemagick')
    exit()




# varying d animations
if False:
    #ds = np.concatenate((np.repeat(0.5, 10), np.arange(0.5, 4.1, 0.05))) # d is large
    ds = np.concatenate((np.repeat(0.5, 10), np.arange(0.5, 0.02, -0.01))) # d is small
    
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000)
    results = np.zeros((len(ds), len(theta_scan)))
    for d_i in range(len(ds)):
        print(d_i)

        # Have to recalc r
        a = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta))
        a = a.reshape(-1,1)
        r = a @ tx.T

        # DISABLE FOR THE FIRST TWO ANIMATIONS
        if True:
            theta1 = 20 / 180 * np.pi
            theta2 = -40 / 180 * np.pi
            a1 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta1))
            a1 = a1.reshape(-1,1)
            a2 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta2))
            a2 = a2.reshape(-1,1)
            freq1 = np.exp(2j*np.pi*0.02e6*t)
            freq1 = freq1.reshape(-1,1)
            freq2 = np.exp(2j*np.pi*-0.02e6*t)
            freq2 = freq2.reshape(-1,1)
            # two tones at diff frequencies and angles of arrival (not sure it actually had to be 2 diff freqs...)
            r = a1 @ freq1.T + a2 @ freq2.T

        for theta_i in range(len(theta_scan)):
            w = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta_scan[theta_i]))
            r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
            results[d_i, theta_i]  = np.mean(np.abs(r_weighted)**2) # energy detector

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.set_tight_layout(True)
    line, = ax.plot(theta_scan, results[0,:])
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90) 
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    text = ax.text(0.6, 12, 'fillmein', fontsize=16)
    def update(i):
        i = int(i)
        print(i)
        results_i = results[i,:] #/ np.max(results[i,:]) * 10 # had to add this in for the last animation because it got too large
        line.set_ydata(results_i)
        d_str = str(np.round(ds[i],2))
        if len(d_str) == 3:
            d_str += '0'
        text.set_text('d = ' + d_str)
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(ds)), interval=100)
    #anim.save('../_images/doa_d_is_large_animation.gif', dpi=80, writer='imagemagick')
    #anim.save('../_images/doa_d_is_small_animation.gif', dpi=80, writer='imagemagick')
    anim.save('../_images/doa_d_is_small_animation2.gif', dpi=80, writer='imagemagick')
    exit()



# Capons beamformer
if True:
    if False: # use for doacompons2
        # more complex scenario
        Nr = 8 # 8 elements
        theta1 = 20 / 180 * np.pi # convert to radians
        theta2 = 25 / 180 * np.pi
        theta3 = -40 / 180 * np.pi
        a1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1))
        a1 = a1.reshape(-1,1)
        a2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2))
        a2 = a2.reshape(-1,1)
        a3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3))
        a3 = a3.reshape(-1,1)
        # we'll use 3 different frequencies
        tone1 = np.exp(2j*np.pi*0.01e6*t)
        tone1 = tone1.reshape(1,-1)
        tone2 = np.exp(2j*np.pi*0.02e6*t)
        tone2 = tone2.reshape(1,-1)
        tone3 = np.exp(2j*np.pi*0.03e6*t)
        tone3 = tone3.reshape(1,-1)
        r = a1 @ tone1 + a2 @ tone2 + 0.1 * a3 @ tone3
        n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
        r = r + 0.04*n

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))
        a = a.reshape(-1,1) # 3x1

        # Calc covariance matrix
        R = r @ r.conj().T # gives a Nr x Nr covariance matrix of the samples

        Rinv = np.linalg.pinv(R) # still 3x3

        w = 1/(a.conj().T @ Rinv @ a) # denominator is 1x3 * 3x3 * 3x1
        metric = np.abs(w.squeeze()) # take magnitude
        metric = 10*np.log10(metric)

        results.append(metric) 

    results /= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_capons.svg', bbox_inches='tight')
    #fig.savefig('../_images/doa_capons2.svg', bbox_inches='tight')
    exit()


# plugging complex scenario into simple DOA approach
if False:
    # more complex scenario
    Nr = 8 # 8 elements
    theta1 = 20 / 180 * np.pi # convert to radians
    theta2 = 25 / 180 * np.pi
    theta3 = -40 / 180 * np.pi
    a1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1))
    a1 = a1.reshape(-1,1)
    a2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2))
    a2 = a2.reshape(-1,1)
    a3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3))
    a3 = a3.reshape(-1,1)
    # we'll use 3 different frequencies
    tone1 = np.exp(2j*np.pi*0.01e6*t)
    tone1 = tone1.reshape(-1,1)
    tone2 = np.exp(2j*np.pi*0.02e6*t)
    tone2 = tone2.reshape(-1,1)
    tone3 = np.exp(2j*np.pi*0.03e6*t)
    tone3 = tone3.reshape(-1,1)
    r = a1 @ tone1.T + a2 @ tone2.T + 0.1 * a3 @ tone3.T
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.04*n

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # look familiar?
        r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
        metric = np.mean(np.abs(r_weighted)**2) # energy detector
        metric = 10*np.log10(metric)
        results.append(metric) 

    results /= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_complex_scenario.svg', bbox_inches='tight')
    exit()



# MUSIC with complex scenario
if False:
    Nr = 8 # 8 elements
    theta1 = 20 / 180 * np.pi # convert to radians
    theta2 = 25 / 180 * np.pi
    theta3 = -40 / 180 * np.pi
    a1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1))
    a1 = a1.reshape(-1,1)
    a2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2))
    a2 = a2.reshape(-1,1)
    a3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3))
    a3 = a3.reshape(-1,1)
    # we'll use 3 different frequencies
    tone1 = np.exp(2j*np.pi*0.01e6*t)
    tone1 = tone1.reshape(-1,1)
    tone2 = np.exp(2j*np.pi*0.02e6*t)
    tone2 = tone2.reshape(-1,1)
    tone3 = np.exp(2j*np.pi*0.03e6*t)
    tone3 = tone3.reshape(-1,1)
    r = a1 @ tone1.T + a2 @ tone2.T + 0.1 * a3 @ tone3.T
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.04*n

    # MUSIC Algorithm (part that doesn't change with theta_i)
    num_expected_signals = 3 # Try changing this!
    R = r @ r.conj().T # Calc covariance matrix, it's Nr x Nr
    w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]

    if False:
        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
        ax1.plot(10*np.log10(np.abs(w)),'.-')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue [dB]')
        plt.show()
        #fig.savefig('../_images/doa_eigenvalues.svg', bbox_inches='tight') # I EDITED THIS ONE IN INKSCAPE!!!
        exit()

    eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
    v = v[:, eig_val_order] # sort eigenvectors using this order
    V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64) # Noise subspace is the rest of the eigenvalues
    for i in range(Nr - num_expected_signals):
        V[:, i] = v[:, i]

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))
        a = a.reshape(-1,1)
        metric = 1 / (a.conj().T @ V @ V.conj().T @ a) # The main MUSIC equation
        metric = np.abs(metric.squeeze()) # take magnitude
        metric = 10*np.log10(metric) # convert to dB
        results.append(metric) 

    results /= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_music.svg', bbox_inches='tight')
    exit()


# MUSIC animation changing angle of two
if False:
    Nr = 8 # 8 elements
    num_expected_signals = 2

    theta2s = np.arange(15,21,0.05) / 180 * np.pi
    theta_scan = np.linspace(-1*np.pi, np.pi, 2000)
    results = np.zeros((len(theta2s), len(theta_scan)))
    for theta2s_i in range(len(theta2s)):
        theta1 = 18 / 180 * np.pi # convert to radians
        theta2 = theta2s[theta2s_i]
        a1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1))
        a1 = a1.reshape(-1,1)
        a2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2))
        a2 = a2.reshape(-1,1)
        tone1 = np.exp(2j*np.pi*0.01e6*t)
        tone1 = tone1.reshape(-1,1)
        tone2 = np.exp(2j*np.pi*0.02e6*t)
        tone2 = tone2.reshape(-1,1)
        r = a1 @ tone1.T + a2 @ tone2.T
        n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
        r = r + 0.01*n
        R = r @ r.conj().T # Calc covariance matrix, it's Nr x Nr
        w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
        eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
        v = v[:, eig_val_order] # sort eigenvectors using this order
        V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64) # Noise subspace is the rest of the eigenvalues
        for i in range(Nr - num_expected_signals):
            V[:, i] = v[:, i]
        for theta_i in range(len(theta_scan)):
            a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i]))
            a = a.reshape(-1,1)
            metric = 1 / (a.conj().T @ V @ V.conj().T @ a) # The main MUSIC equation
            metric = np.abs(metric.squeeze()) # take magnitude
            metric = 10*np.log10(metric) # convert to dB
            results[theta2s_i, theta_i] = metric

        results[theta2s_i,:] /= np.max(results[theta2s_i,:]) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.set_tight_layout(True)
    line, = ax.plot(theta_scan, results[0,:])
    ax.set_thetamin(0)
    ax.set_thetamax(30) 
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    def update(i):
        i = int(i)
        print(i)
        results_i = results[i,:] #/ np.max(results[i,:]) * 10 # had to add this in for the last animation because it got too large
        line.set_ydata(results_i)
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(theta2s)), interval=100)
    anim.save('../_images/doa_music_animation.gif', dpi=80, writer='imagemagick')
    exit()