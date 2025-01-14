# Until the following is done, you'll get an error: Exception: Channel voltage2 not found
'''
fw_setenv attr_name compatible
fw_setenv attr_val ad9361
fw_setenv compatible ad9361
fw_setenv mode 2r2t
reboot
'''
# If that doesnt work, you may need to upgrade the plutos firmware or pyadi-iio or both
# Note- old revisions of the pluto may not support 2 channels, eg I had trouble with a rev B

import numpy as np
import adi

sample_rate = 10e6 # Hz
center_freq = 100e6 # Hz

sdr = adi.ad9361("ip:192.168.1.10") # "ip:192.168.2.1")
sdr.rx_enabled_channels = [1] # or [0] or [0,1]
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_lo = int(center_freq) # applies to both channels
sdr.rx_buffer_size = 100000
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = 50
sdr.rx_hardwaregain_chan1 = 50

samples = sdr.rx()
print(np.shape(samples)) # quick way to check how much signal power there is
