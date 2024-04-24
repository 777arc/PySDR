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
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
#print(s)

# we have to do a matrix multiplication of s and tx, which currently are both 1D, so we have to make them 2D with reshape
s = s.reshape(-1,1)
#print(s.shape) # 3x1
tx = tx.reshape(-1,1)
#print(tx.shape) # 10000x1

# so how do we use this? simple:
r = s @ tx.T # matrix multiply. dont get too caught up by the transpose s, the important thing is we're multiplying the steering vector by the tx signal
#print(r.shape) # 3x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

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
# AWGN with a phase shift applied is still AWGN so we can add it after or before the steering vector is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.5*n

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
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
        r_weighted = w.conj().T @ r # apply our weights. remember r is 3x10000
        results.append(10*np.log10(np.var(r_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
    results -= np.max(results) # normalize

    # print angle that gave us the max value
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
    #ax.set_rgrids([0,2,4,6,8]) 
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
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
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
        s = s.reshape(-1,1) # 3x1
        tone = np.exp(2j*np.pi*0.02e6*t)
        tone = tone.reshape(-1,1) # 10000x1
        r = s @ tone.T

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
        s = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta))
        s = s.reshape(-1,1)
        r = s @ tx.T

        # DISABLE FOR THE FIRST TWO ANIMATIONS
        if True:
            theta1 = 20 / 180 * np.pi
            theta2 = -40 / 180 * np.pi
            s1 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta1)).reshape(-1,1)
            s2 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
            freq1 = np.exp(2j*np.pi*0.02e6*t).reshape(-1,1)
            freq2 = np.exp(2j*np.pi*-0.02e6*t).reshape(-1,1)
            # two tones at diff frequencies and angles of arrival (not sure it actually had to be 2 diff freqs...)
            r = s1 @ freq1.T + s2 @ freq2.T

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



# MVDR/Capons beamformer
if True:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_ylim([-10, 0])

    # theta is the direction of interest, in radians, and r is our received signal
    def w_mvdr(theta, r):
        s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta
        s = s.reshape(-1,1) # make into a column vector (size 3x1)
        R = (r @ r.conj().T)/r.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
        Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
        w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
        return w

    def power_mvdr(theta, r):
        s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
        s = s.reshape(-1,1) # make into a column vector (size 3x1)
        #R = (r @ r.conj().T)/r.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
        R = np.cov(r)
        print(R)
        Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
        return 1/(s.conj().T @ Rinv @ s).squeeze()
    
    if False: # use for doacompons2
        # more complex scenario
        Nr = 8 # 8 elements
        theta1 = 20 / 180 * np.pi # convert to radians
        theta2 = 25 / 180 * np.pi
        theta3 = -40 / 180 * np.pi
        s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
        s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
        s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
        # we'll use 3 different frequencies.  1xN
        tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
        tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
        tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
        r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
        n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
        r = r + 0.05*n # 8xN
        ax.set_ylim([-30, 0])

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        #w = w_mvdr(theta_i, r) # 3x1
        #r_weighted = w.conj().T @ r # apply weights
        #power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        #results.append(power_dB)
        results.append(10*np.log10(power_mvdr(theta_i, r))) # compare to using equation for MVDR power, should match, SHOW MATH OF WHY THIS HAPPENS!
    results -= np.max(results) # normalize
    print(theta_scan[np.argmax(results)] * 180/np.pi) # Angle at peak, in degrees

    
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    
    ax.set_thetamin(-90)
    ax.set_thetamax(90) 

    #fig.savefig('../_images/doa_capons.svg', bbox_inches='tight')
    #fig.savefig('../_images/doa_capons2.svg', bbox_inches='tight')
    plt.show()
    exit()


# plugging complex scenario into conventional DOA approach
if False:
    # more complex scenario
    Nr = 8 # 8 elements
    theta1 = 20 / 180 * np.pi # convert to radians
    theta2 = 25 / 180 * np.pi
    theta3 = -40 / 180 * np.pi
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
    s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
    # we'll use 3 different frequencies.  1xN
    tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
    tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
    r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.05*n # 8xN

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
        r_weighted = w.conj().T @ r # apply our weights. remember r is 3x10000
        results.append(10*np.log10(np.var(r_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
    results -= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    ax.set_thetamin(-90)
    ax.set_thetamax(90) 
    plt.show()
    fig.savefig('../_images/doa_complex_scenario.svg', bbox_inches='tight')
    exit()



# MUSIC with complex scenario
if False:
    # more complex scenario
    Nr = 8 # 8 elements
    theta1 = 20 / 180 * np.pi # convert to radians
    theta2 = 25 / 180 * np.pi
    theta3 = -40 / 180 * np.pi
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
    s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
    # we'll use 3 different frequencies.  1xN
    tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
    tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
    r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.05*n # 8xN

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
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1)
        metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
        metric = np.abs(metric.squeeze()) # take magnitude
        metric = 10*np.log10(metric) # convert to dB
        results.append(metric)
    results -= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    plt.show()
    fig.savefig('../_images/doa_music.svg', bbox_inches='tight')
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
        s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1)
        s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
        tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(-1,1)
        tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(-1,1)
        r = s1 @ tone1.T + s2 @ tone2.T
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
            s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i])).reshape(-1,1)
            metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
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



# Radar style scenario using MVDR, with a training phase, and comparing it to normal DOA approach (NORMAL SEEMS TO WORK BETTER SO IM LEAVING RADAR STYLE OUT FOR NOW)
if False:
    # 1 jammer 1 SOI, generating two different received signals so we can isolate jammer for the training step
    # Jammer is complex baseband noise
    # Signal is complex baseband noise
    N = 1000
    Nr = 32 # number of elements
    theta_jammer = 20 / 180 * np.pi
    theta_soi =    30 / 180 * np.pi
    s_jammer = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_jammer)).reshape(-1,1) # Nr x 1
    s_soi =    np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1)

    # Generate the signal with just jammer, before SOI turns on
    jamming_signal = np.random.randn(1,  N) + 1j*np.random.randn(1,  N)
    system_noise =   np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    s_jammer = np.sqrt(1000) * s_jammer @ jamming_signal + system_noise

    # Generate the signal after SOI turns on
    jamming_signal = np.random.randn(1,  N) + 1j*np.random.randn(1,  N)
    soi_signal =     np.random.randn(1,  N) + 1j*np.random.randn(1,  N)
    system_noise =   np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r_both =   np.sqrt(1000) * s_jammer @ jamming_signal + np.sqrt(10) * s_soi @ soi_signal + system_noise

    # "Training" step, with just jammer present
    Rinv_jammer = np.linalg.pinv(r_jammer @ r_jammer.conj().T) # Nr x Nr, inverse covariance matrix estimate using the received samples

    # Plot beam pattern when theta = SOI, note that this process doesnt actually involve using r_both
    if True:
        N_fft = 1024
        theta_i = theta_soi
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1) # steering vector
        w = (Rinv_jammer @ s)/(s.conj().T @ Rinv_jammer @ s) # MVDR
        w = np.conj(w) # or else our answer will be negative/inverted
        w = w.squeeze()
        w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
        theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
        fig, ax = plt.subplots()
        ax.plot([theta_jammer * 180/np.pi]*2, [-50, np.max(w_fft_dB)], 'r:') # position of jammer
        ax.plot([theta_soi * 180/np.pi]*2, [-50, np.max(w_fft_dB)], 'g:') # position of SOI
        ax.plot(theta_bins * 180/np.pi, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.set_xlabel("Theta [Degrees]")
        ax.set_ylabel("Beam Pattern [dB]")
        plt.show()

    # Now perform DOA by processing r_both.  We still get a spike in the direction of the jammer, since its treaing the jammer as the SOI at that theta, but the important thing is we were able to also find the SOI spike
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # sweep theta between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1) # steering vector in the desired direction theta (size Nr x 1)
        w = (Rinv_jammer @ s)/(s.conj().T @ Rinv_jammer @ s) # MVDR/Capon equation!  Note which R's are being used where
        r_weighted = w.conj().T @ r_both # apply weights to the signal that contains both jammer and SOI
        power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        results.append(power_dB)

    results -= np.max(results) # normalize

    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig, ax = plt.subplots()
    ax.plot(theta_scan * 180/np.pi, results)
    #ax.plot([theta_soi * 180/np.pi, theta_soi * 180/np.pi], [-30, -20],'g--')
    #ax.plot([theta_jammer * 180/np.pi, theta_jammer * 180/np.pi], [-30, -20],'r--')
    ax.set_xlabel("Theta [Degrees]")
    ax.set_ylabel("DOA Metric")
    #ax.set_theta_zero_location('N') # make 0 degrees point up
    #ax.set_theta_direction(-1) # increase clockwise
    #ax.set_rlabel_position(55)  # Move grid labels away from other labels
    #ax.set_ylim([-40, 0]) # only plot down to -40 dB
    #plt.show()
    #fig.savefig('../_images/doa_radar_scenario.svg', bbox_inches='tight')

    # Now compare to just doing MVDR DOA on r_both
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # sweep theta between -180 and +180 degrees
    results = []
    Rinv_both = np.linalg.pinv(r_both @ r_both.conj().T) # Nr x Nr, inverse covariance matrix estimate using the received samples
    for theta_i in theta_scan:
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1) # steering vector in the desired direction theta (size Nr x 1)
        w = (Rinv_both @ s)/(s.conj().T @ Rinv_both @ s) # MVDR/Capon equation!  Note which R's are being used where
        r_weighted = w.conj().T @ r_both # apply weights to the signal that contains both jammer and SOI
        power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        results.append(power_dB)
    results -= np.max(results) # normalize
    ax.plot(theta_scan * 180/np.pi, results)
    ax.set_xlabel("Theta [Degrees]")
    ax.set_ylabel("DOA Metric")
    ax.legend(['Radar Style', 'Normal DOA Approach'])
    plt.show()

    exit()


# Create quiescent antenna pattern using FFT of weights, changing number of elements is really the only thing that will change the pattern
if False:
    N_fft = 512
    theta = theta_degrees / 180 * np.pi # doesnt need to match SOI, we arent processing samples, this is just the direction we want to point at
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
    w = np.conj(w) # or else our answer will be negative/inverted
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    
    # Map the FFT bins to angles in radians
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians
    
    # find max so we can add it to plot
    theta_max = theta_bins[np.argmax(w_fft_dB)]
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
    ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
    plt.show()

    fig.savefig('../_images/doa_quiescent.svg', bbox_inches='tight')
    exit()


'''
Wiener filter approach NEVER GOT THIS WORKING
Notes:
    dont use np.dot unless its two 1Ds
    why FFT?
    There's also the multistage wiener approach which has a cool diagram
    make the simple wiener diagram first
''' 
if False:
    # 2 element, 1 jammer 1 SOI, two different r's so we can isolate jammer first
    Nr = 2
    theta1 = 20 / 180 * np.pi # Jammer
    theta2 = 30 / 180 * np.pi # SOI
    s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
    s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
    tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
    tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
    r_jammer = s1 @ tone1 + 0.05*(np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N))
    r_both = s1 @ tone1 + s2 @ tone2 + 0.05*(np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N))

    def w_wiener(r):
        Rx_0 = r[0,:]
        Rx_1 = r[1,:]

        #Rxx_hat = (1/N) * np.sum(np.conj(Rx_1) * Rx_1) # scalar (appears to be real only)
        Rxx_hat = np.correlate(Rx_1, Rx_1) / N # same as above
        Rxx_hat = Rxx_hat.squeeze() # converts the 1D array of length-1 to a scalar

        #rxz_hat = np.sum(Rx_1 * np.conj(Rx_0)) / N # scalar
        rxz_hat = np.correlate(Rx_1, Rx_0) / N # same as above
        rxz_hat = rxz_hat.squeeze()
        
        w_hat = (1 / Rxx_hat) * rxz_hat # scalar
        #w_vector = np.array([[1], [-w_hat]]) # 2x1, this is the actual weights, but the first element is always = 1 with wiener filtering
        #T = np.sqrt(2)/2 * np.array([[1, -1], [1, 1]]) # 2x2
        #w_vector_T = np.dot(T, w_vector) # 2x1 (DONT USE DOT IT CAN MEAN DIFFERENT THINGS, ITS A MATMUL HERE)
        #return w_vector_T
        return w_hat
        '''
        w_padded = np.zeros(100, dtype=complex) # first arg seems arbitrary
        w_padded[0] = w_vector_T[0][0]
        w_padded[1] = w_vector_T[1][0]
        w_fft = np.fft.fft(w_padded)
        w_shift = np.fft.fftshift(w_fft)
        w_db = 20*np.log10(np.abs(w_shift))
        plt.plot(w_db)
        plt.show()
        '''

    def dbfs(raw_data):
        # function to convert IQ samples to FFT plot, scaled in dBFS
        NumSamples = len(raw_data)
        win = np.hamming(NumSamples)
        y = raw_data * win
        s_fft = np.fft.fft(y) / np.sum(win)
        s_shift = np.fft.fftshift(s_fft)
        s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
        return s_dbfs
    
    # Measure the jammer signal
    w_hat = w_wiener(r_jammer)

    # Now "turn on the SOI", we will use r_both:
    Rx_0 = r_both[0,:]
    Rx_1 = r_both[1,:]
    y = Rx_0 - np.conj(w_hat) * Rx_1 # wiener filter equation for 2 elements, ITS AS IF THERE's a 1+0j INFRONT OF THE FIRST ELEMENT

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        delayed_sum = y + Rx_1 * np.exp(1j * theta_i) # Jons code 
        #delayed_sum = y + Rx_1 * np.exp(-1j * np.pi *  np.sin(theta_i)) # Me trying out the normal equation for exp()
        #delayed_sum_dbm = dbfs(delayed_sum)
        #results.append(np.max(delayed_sum_dbm))
        results.append(10*np.log10(np.var(delayed_sum))) # equivalent to 2 lines above

    print(theta_scan[np.argmax(results)] * 180 / np.pi / 3) # Angle at peak, in degrees NOTE THE ARBITRARY DIVIDE BY 3 NESSESARY TO GET IT TO WORK

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    plt.show()




