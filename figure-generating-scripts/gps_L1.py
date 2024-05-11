import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

'''
1. Acquisition - Detect the presence of satellites https://gnss-sdr.org/docs/sp-blocks/acquisition/
2. For each satellite determine frequency shift and delay
3. Tracking - Track carrier phase and code delay over time. see state machine here https://gnss-sdr.org/docs/sp-blocks/tracking/
4. Decode the navigation (aka telemetry) message https://gnss-sdr.org/docs/sp-blocks/telemetry-decoder/

https://github.com/JasonNg91/GNSS-SDR-Python/blob/master/acquire-gps-l1.py
https://github.com/psas/gps
'''

g1tap = [2,9]
g2tap = [1,2,5,7,8,9]
sats = [(1, 5), (2, 6), (3, 7), (4, 8), (0, 8), (1, 9), (0, 7), (1, 8), (2, 9), (1, 2),
        (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 3), (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 8), (0, 2), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (0, 5), (1, 6),
        (2, 7), (3, 8), (4, 9), (3, 9), (0, 6), (1, 7), (3, 9)]

# Test file from IQEngine, was taken on 2022-03-27T11:32:04.2147593125
samples_to_read = 10e6
samples_offset = 1e6
x = np.fromfile('/mnt/c/Users/marclichtman/Downloads/GPS-L1.sigmf-data', dtype=np.int16, count=int(samples_to_read), offset=int(samples_offset))
x = x.astype(np.complex64) / 32768.0
x = x[::2] + 1j*x[1::2]
sample_rate = 4e6
center_freq = 1575.42e6

# Resample to 4092 MHz because it's an integer multiple of the chip rate 1023M chips/sec
resampling_rate = 4.092e6/sample_rate
x = resample(x, int(resampling_rate*len(x)))
sample_rate = 4.092e6


def gold_code(prn): # Returns a list of bits that form the Gold Code PRN of the designated satellite
    g1 = np.ones(10)
    g2 = np.ones(10)
    g = np.empty(1023)
    for i in range(1023):
        val = (g1[9] + g2[prn[0]] + g2[prn[1]]) % 2
        g[i] = val

        # shift g1
        g1[9] = sum([g1[i] for i in g1tap]) % 2 
        g1 = np.roll(g1, 1)

        # shift g2
        g2[9] = sum([g2[i] for i in g2tap]) % 2 
        g2 = np.roll(g2, 1)

    # Convert to BPSK by changing 0 to -1
    g = g * 2 - 1
    return np.repeat(g, 4) # repeat each chip 4x to match our sample rate which is 4x the chip rate


def _GetSecondLargest(arr): # Returns the second largest value in an array.  It will also ignore any value that is close to the second largest value
    ScaledLargest = 0.95 * np.amax(arr) # Reduce value by a percent to prevent near-identical values from being selected
    SecondLargest = 0
    for ind, val in enumerate(arr):
        if val < ScaledLargest:
            if val > SecondLargest: # Ignore adjacent bins to Largest
                if np.abs(np.argmax(arr) - ind) > 100:
                    SecondLargest = val
    return SecondLargest


def findSat(samples, code, bins, block_size_ms=10, tracking = False):
    samples_slice = samples[0:(4092*block_size_ms)]
    NsamplesBlock = 4092 * block_size_ms

    peakToSecondList = np.zeros(len(bins))
    codePhaseList = np.zeros(len(bins))
    SNRList = np.zeros(len(bins))

    codefft = np.fft.fft(code, len(samples_slice))
    GCConj = np.conjugate(codefft)
    
    N = len(bins)
    freqInd = 0
    # Loop through all frequencies
    for n, curFreq in enumerate(bins):
        
        # Shift frequency to baseband using complex exponential
        t = np.arange(len(samples_slice))/sample_rate
        samples_slice = samples_slice * np.exp(-2j * np.pi * curFreq * t)

        # Mix code fft and take inverse, then square
        result = np.fft.ifft(GCConj * np.fft.fft(samples_slice))
        result_squared = np.real(result * np.conjugate(result)) # imag part will always be near zero

        #rmsPowerdB = 10*np.log10(np.mean(result_squared))
        #resultdB = 10*np.log10(result_squared)

        codePhaseInSamples = np.argmax(result_squared[0:4092])

        # Search for secondlargest value in 1 ms worth of data
        secondLargestValue = _GetSecondLargest(result_squared[0:int(sample_rate * 0.001)])

        # Pseudo SNR
        firstPeak = np.amax(result_squared[0:4092])
        peakToSecond = 10*np.log10(firstPeak / secondLargestValue)

        #if tracking is True:
        peakToSecondList[n] = peakToSecond
        codePhaseList[n] = codePhaseInSamples
        SNRList[n] = 10*np.log10(  firstPeak/np.mean(result_squared)  )

        # Don't print data when correlation is probably not happening
        SNR_THRESHOLD = 3.4
        if peakToSecond > SNR_THRESHOLD:
            print("Possible acquisition: Freq: %8.4f, Peak2Second: %8.4f, Code Phase (samples): %8.4f"
                  %(curFreq, peakToSecond, codePhaseInSamples))

        freqInd = freqInd + 1

        # Percentage Output
        print("%02d%%"%((n/N)*100), end="\r")
   
    #print(SNRList[np.argmax(peakToSecondList)])
    #print(bins[np.argmax(peakToSecondList)]) # doppler
    #print(codePhaseList[np.argmax(peakToSecondList)]) # CodePhaseSamples
    #print(1023 - (1.023e6) / (4.092e6) * codePhaseList[np.argmax(peakToSecondList)]) # CodePhaseChips

    if True:
        plt.ion()
        plt.plot(bin_list, peakToSecondList)
        plt.ylim((0, 20))
        plt.xlabel('Doppler Shift (Hz)')
        plt.ylabel('Peak-to-SecondLargest ratio (dB)')
        plt.title("Sat %d - PeakToSecondLargest"%curSat)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
    
    Acquired = np.amax(peakToSecondList) >= SNR_THRESHOLD # Check if Acquisition was successful for this satellite

    # Get fine-frequency (If acquired):
    if Acquired == True:
        # Already have a CA code that is at least 1 ms in length
        CACode = code[0:4092] # store first ms

        # Repeat entire array 5 times for 5 ms
        code5ms = np.tile(CACode, int(5))

        #GetFineFrequency(data,curSatInfo,code5ms)

    return


'''
def GetFineFrequency(data, SatInfo, code5ms): # now passed in data class
    # Performs fine-frequency estimation. In this case, data will be a slice
    # of data (probably same length of data that was used in the circular
    # cross-correlation)

    
    Ts = 1/sample_rate

    # Medium-frequency estimation data length (1ms in book, but may need to used
    # the data length from acquisition)
    numMSmf = 1 # num ms for medium-frequency estimation
    Nmf = int(np.ceil(numMSmf*0.001*sample_rate))  # num of samples to use for medium-frequency estimation (and DFT)

    dataMF = data.CData[0:(4092*numMSmf)]

    # Create list of the three frequencies to test for medium-frequency estimation.
    k = []
    k.append(SatInfo.DopplerHz - 400*10**3)
    k.append(SatInfo.DopplerHz)
    k.append(SatInfo.DopplerHz + 400*10**3)

    # Create sampled time array for DFT
    nTs = np.linspace(0,Ts*(Nmf + 1),Nmf,endpoint=False)

    # Perform DFT at each of the three frequencies.
    X = []
    X.append(np.abs(sum(dataMF*np.exp(-2*np.pi*1j*k[0]*nTs)))**2)
    X.append(np.abs(sum(dataMF*np.exp(-2*np.pi*1j*k[1]*nTs)))**2)
    X.append(np.abs(sum(dataMF*np.exp(-2*np.pi*1j*k[2]*nTs)))**2)

    # Store the frequency value that has the largest power
    kLargest = k[np.argmax(X)]
    print("Largest of three frequencies: %f"%kLargest) # Will remove. Temporarily for debugging purposes.

    # Get 5 ms of consecutive data, starting at beginning of CA Code
    CACodeBeginning = int(SatInfo.CodePhaseSamples)
    data5ms = data.CData[CACodeBeginning:int(5*4092) + CACodeBeginning]

    # Get 5 ms of CA Code, with no rotation performed.
    # passed in from function (code5ms)

    # Multiply data with ca code to get cw signal
    dataCW = data5ms*code5ms

    # Perform DFT on each of the ms of data (5 total), at kLargest frequency.
    # Uses variables from medium-frequency, so if they change, may need to re-create below.
    X = []
    PhaseAngle = []
    for i in range(0,5):
        X.append(sum(dataCW[i*4092:(i+1)*4092]*np.exp(-2*np.pi*1j*kLargest*nTs)))
        PhaseAngle.append(np.arctan(np.imag(X[i])/np.real(X[i])))
        print("Magnitude: %f" %X[i])
        print("Phase Angle: %f" %PhaseAngle[i])

    # Get difference angles
    PhaseDiff = []
    for i in range(1,5):
        PhaseDiff.append(PhaseAngle[i]-PhaseAngle[i-1])
        print("Phase difference %d, is: %f"%((i-1),PhaseDiff[i-1]))

    # Adjust phases so magnitude not greater than 2.3*pi/5
    # WIP
    PhaseThreshold = (2.3*np.pi)/5
    for (i,curPhaseDiff) in enumerate(PhaseDiff):
        if np.abs(curPhaseDiff) > PhaseThreshold:
            curPhaseDiff = PhaseDiff[i] - 2*np.pi
            if np.abs(curPhaseDiff) > PhaseThreshold:
                curPhaseDiff = PhaseDiff[i] + 2*np.pi
                if np.abs(curPhaseDiff) > (2.2*np.pi)/5:
                    curPhaseDiff = PhaseDiff[i] - np.pi
                    if np.abs(curPhaseDiff) > PhaseThreshold:
                        curPhaseDiff = PhaseDiff[i] - 3*np.pi
                        if np.abs(curPhaseDiff) > PhaseThreshold:
                            curPhaseDiff = PhaseDiff[i] + np.pi
        PhaseDiff[i] = curPhaseDiff
    fList = (np.array(PhaseDiff)/(2*np.pi*0.001))
    print(fList)
    print(np.mean(fList))

    FineFrequencyEst = 0 # Just a placeholder.
    return FineFrequencyEst
'''

bin_list=range(-10000,10000, 100)
sat_list=range(1, 33)
block_size_ms=10

# Create array to store max values, freq ranges, per satellite
satInd = 0
# Loop through selected satellites
for curSat in sat_list:
    print("Searching for SV " + str(curSat) + "...")
    CACode = gold_code(sats[curSat-1]) # Grab a CA Code
    CACodeSampled = np.tile(CACode, int(len(x)/sample_rate*1000)) # Repeat entire array for each ms of data sampled
    findSat(x, CACodeSampled, bin_list, block_size_ms)
    satInd += 1

