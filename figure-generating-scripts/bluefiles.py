
import numpy as np
import os
import matplotlib.pyplot as plt

filename = 'yourfile.blue' # or cdif

filesize = os.path.getsize(filename)
print('File size', filesize, 'bytes')
with open(filename, 'rb') as f:
    header = f.read(512)

# Decode the header
dtype = header[52:54].decode('utf-8') # eg 'CI'
endianness = header[8:12].decode('utf-8') # better be 'EEEI'! we'll assume it is from this point on
extended_header_start = int.from_bytes(header[24:28], byteorder='little') * 512 # in units of bytes
extended_header_size = int.from_bytes(header[28:32], byteorder='little')
if extended_header_size != filesize - extended_header_start:
    print('Warning: extended header size seems wrong')
time_interval = np.frombuffer(header[264:272], dtype=np.float64)[0]
sample_rate = 1/time_interval
print('Sample rate', sample_rate/1e6, 'MHz')

# Read in the IQ samples
if dtype == 'CI':
    samples = np.fromfile(filename, dtype=np.int16, offset=512, count=(filesize-extended_header_size))
    samples = samples[::2] + 1j*samples[1::2] # convert to IQIQIQ...

# Plot every 1000th sample to make sure there's no garbage
#print(len(samples))
#plt.plot(samples.real[::1000])
#plt.show()


# ...


# Read in the extended header at the end of the file
with open(filename, 'rb') as f:
    f.seek(filesize-extended_header_size)
    ext_header = f.read(extended_header_size)
    print("length of extended header", len(ext_header), '\n')

def parse_extended_header(idx):
    next_offset = np.frombuffer(ext_header[idx:idx+4], dtype=np.int32)[0]
    non_data_length = np.frombuffer(ext_header[idx+4:idx+6], dtype=np.int16)[0]
    name_length = ext_header[idx+6]
    dataStart = idx + 8
    dataLength = dataStart + next_offset - non_data_length
    midas_to_np = {'O' : np.uint8, 'B' : np.int8, 'I' : np.int16, 'L' : np.int32, 'X' : np.int64, 'F' : np.float32, 'D' : np.float64}
    format_code = chr(ext_header[idx+7])
    if format_code == 'A':
        val = ext_header[dataStart:dataLength].decode('latin_1')
    else:
        val = np.frombuffer(ext_header[dataStart:dataLength], dtype=midas_to_np[format_code])[0]
    key = ext_header[dataLength:dataLength+name_length].decode('latin_1')
    print(key, '  ', val)
    return idx + next_offset

next_idx = 0
while next_idx < extended_header_size:
    next_idx = parse_extended_header(next_idx)
