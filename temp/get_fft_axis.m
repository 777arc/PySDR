function fft_axis = get_fft_axis(N, Fs, shift)
% Retreives the proper fft axis in Hz (fftshifted version or not) given N fft points and
% sample rate Fs in Hz
% If shift == 1, neg. freqs are explicitly calculated and put on the left
% If shift == 0, neg. freqs. are not explicity calculated and kept on right

if shift == 1 % get fftshift form of axis (use explicit negative frequencies on left side of freq. axis)
    if round(N/2)==N/2 % N is even
        fft_axis=[-(N/2)/N:1/N:(N/2-1)/N]*Fs;
    elseif round(N/2)~=N/2 % N is odd
        fft_axis=[-((N-1)/2)/N:1/N:((N-1)/2)/N]*Fs;
    else
        error(['N needs to be an integer.'])
    end
else % else if shift = 0 (use implied negative frequencies on right side of freq. axis)
    fft_axis=[0:1/N:(N-1)/N]*Fs;
end

