import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
N = 10000 # number of samples to simulate
d = 0.5
Nr = 2
theta = np.deg2rad(20)

# Create a tone to act as the transmitted signal
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1,1) # steering vector
t = np.arange(N)/sample_rate
f_tone = 0.02e6
tx = np.exp(2j*np.pi*f_tone*t).reshape(1,-1) # 1 x 10000
tx = np.random.randn(1, N) + 1j*np.random.randn(1, N)
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = s @ tx + 0.05*n # Nr x 10000

# DOA loop
theta_scan = np.linspace(-1*np.pi, np.pi, 1000)
results_conventional = np.zeros(len(theta_scan))
results_mvdr = np.zeros(len(theta_scan))
for theta_i in range(len(theta_scan)):
    s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i]))

    # Conventional
    w_conventional = s
    r_weighted_conv = np.conj(w_conventional) @ r # apply our weights corresponding to the direction theta_i
    results_conventional[theta_i]  = 10*np.log10(np.mean(np.var(r_weighted_conv))) # energy detector

    # MVDR
    R = (r @ r.conj().T)/r.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    results_mvdr[theta_i] = 10*np.log10(1/(s.conj().T @ Rinv @ s).squeeze())

# Simple phase measurement using angle(x * x.conj)
phase_diff = np.mean(np.angle(r[0, :] * np.conj(r[1, :])))
theta_est = np.arcsin(phase_diff / (2 * np.pi * d))
print("Estimated angle of arrival: ", theta_est * 180/np.pi) # in degrees

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results_conventional)
ax.plot(theta_scan, results_mvdr)
ax.plot([theta_est]*2, [-20, 10], 'c')
ax.plot([theta]*2, [-20, 10], 'g--')
ax.legend(['Conventional', 'MVDR', 'np.angle(x x.conj)', 'Correct AoA'])
ax.set_theta_zero_location('N') # type: ignore # make 0 degrees point up
ax.set_theta_direction(-1) # type: ignore # increase clockwise
ax.set_thetamin(-90) # type: ignore # only show top half
ax.set_thetamax(90) # type: ignore
ax.set_ylim((-20, 10)) # because there's no noise, only go down 30 dB
plt.show()
exit()
