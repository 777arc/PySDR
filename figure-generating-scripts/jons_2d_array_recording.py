import numpy as np
import matplotlib.pyplot as plt

fc = 2.8e9 # ADQUADMXFE1EBZ has Rx range of 2.7-3.7GHz  (FE2 is DC to 1.8 GHz)
d = 0.051  # spacing between antennas in m (d=3e8/NCOFreq /2)
wavelength = 3e8 / fc
Nr = 15
rows = 3
cols = 5

r = np.load("/mnt/d/boresight.npy") 
r = np.delete(r, -1, axis=0)  # 16th element is not connected
# at this point its a 2D array of complex128, size (15, 16384)

# For now, because we have 1 recording, split into train and test
r_train = r[:,:8192]
r_test = r[:,8192:]

# Element positions, still as a list of x,y,z coordinates in meters
pos = np.zeros((Nr, 3))
for i in range(Nr):
    pos[i,0] = d * (i % cols)  # x position
    pos[i,1] = d * (i // cols) # y position
    pos[i,2] = 0               # z position

# Plot positions of elements
if False:
    plt.plot(pos[:,0], pos[:,1], 'o')
    for i in range(Nr):
        plt.text(pos[i,0], pos[i,1], str(i), fontsize=12)
    plt.xlim([-0.01, 0.25])
    plt.ylim([-0.01, 0.15])
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.grid()
    plt.show()
    exit()

###############
# Calibration #
###############

# amplitude and phase offsets are found from the eigenvalue decomposition of the covariance matrix R
r = r_train # TEMPORARY
R = r @ r.conj().T # Calc covariance matrix, it's Nr x Nr
w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]

# Plot eigenvalues
if False:
    w_dB = 10*np.log10(np.abs(w))
    w_dB -= np.max(w_dB) # normalize
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(w_dB, '.-')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue [dB]')
    plt.show()
    exit()

# Use max eigenvector to calibrate
# TODO: the accuracy of cal can be determined by taking crosscorr of all pairs of channels at 0 sample delay and some random sample delay and comparing the two
# TODO: There's also a technique of making a phase-frequency plot, to estimate fractional delay between samples, might be worth trying
v_max = v[:, np.argmax(np.abs(w))]
mags = np.mean(np.abs(r), axis=1)
mags = mags[0] / mags # normalize to first element
phases = np.angle(v_max)
phases = phases[0] - phases # normalize to first element
cal_table = mags * np.exp(1j * phases)
print("cal_table", cal_table)

# Plot Cal offsets
if False:
    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    ax1.plot(np.real(cal_table), np.imag(cal_table), 'r.')
    for i in range(Nr):
        ax1.text(np.real(cal_table[i]), np.imag(cal_table[i]), str(i), fontsize=10)
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imag')
    plt.show()
    exit()

# Apply cal offsets
r = r_test # TEMPORARY
for i in range(Nr):
    r[i, :] *= cal_table[i]


def steering_vector(pos, dir):
    return np.exp(-2j * np.pi * pos @ dir / wavelength) # outputs Nr x 1 (column vector)

def get_unit_vector(theta, phi):  # angles are in radians
    return np.asmatrix([np.sin(theta) * np.sin(phi), # x component
                        np.cos(theta) * np.sin(phi), # y component
                        0]                           # z component
                        ).T

# Crappy 3d plot of the array
if True:
    resolution = 100 # number of points in each direction
    theta_scan = np.linspace(-np.pi, np.pi, resolution) # azimuth angles
    phi_scan = np.linspace(0, np.pi, resolution) # elevation angles
    results = np.zeros((resolution, resolution)) # 2D array to store results
    for i, theta_i in enumerate(theta_scan):
        for j, phi_i in enumerate(phi_scan):
            dir_i = get_unit_vector(theta_i, phi_i)
            s = steering_vector(pos, dir_i) # 16 x 1
            #w = s # Conventional beamformer
            R = np.cov(r) # Covariance matrix, 16 x 16
            Rinv = np.linalg.pinv(R)
            w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation
            resp = w.conj().T @ r
            results[i, j] = 10*np.log10(np.abs(resp)[0,0]) # power in signal, in dB
    # plot_surface needs x,y,z form
    results[results < -10] = -10 # crop the z axis to -10 dB

    # 3D
    if True:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        surf = ax.plot_surface(np.sin(theta_scan[:,None]) * np.sin(phi_scan[None,:]), # x
                            np.cos(theta_scan[:,None]) * np.sin(phi_scan[None,:]), # y
                            results, cmap='viridis')
        # Plot a dot at the maximum point
        max_idx = np.unravel_index(np.argmax(results, axis=None), results.shape)
        ax.scatter(np.sin(theta_scan[max_idx[0]]) * np.sin(phi_scan[max_idx[1]]), # x
                np.cos(theta_scan[max_idx[0]]) * np.sin(phi_scan[max_idx[1]]), # y
                results[max_idx], color='red', s=100)
        ax.set_zlim(-10, results[max_idx])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Power [dB]')
        plt.show()
    
    # 2D
    else:
        x = np.sin(theta_scan[:,None]) * np.sin(phi_scan[None,:])
        y = np.cos(theta_scan[:,None]) * np.sin(phi_scan[None,:])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(x, y, results, levels=100, cmap='viridis')
        plt.show()




