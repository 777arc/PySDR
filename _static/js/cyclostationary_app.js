function cyclostationary_app() {
  const fs = 1; // sample rate in Hz
  const N = Math.pow(2, 16); // number of samples to simulate, needs to be power of 2

  const samples = generate_bspk(N, fs, 10, 0.2 * fs);

  // Plot PSD
  PSD = calc_PSD(samples);
  const f_vals = Array.from({ length: N }, (v, k) => k - N / 2).map((val) => (val * fs) / N);
  Plotly.newPlot("rectPlot", [{ x: f_vals, y: PSD }]);
}

function calc_PSD(samples) {
  const fftsize = samples.length / 2;
  const fft = new FFT(fftsize);
  const out = fft.createComplexArray();
  fft.transform(out, samples);

  // Calculate magnitude
  let PSD = new Array(fftsize);
  for (var i = 0; i < fftsize; i++) {
    PSD[i] = Math.sqrt(out[i * 2] * out[i * 2] + out[i * 2 + 1] * out[i * 2 + 1]);
  }

  // Square the signal
  for (var i = 0; i < fftsize; i++) {
    PSD[i] = PSD[i] * PSD[i];
  }

  PSD = fftshift(PSD);

  // Convert to dB and apply scaling factor
  for (var i = 0; i < fftsize; i++) {
    PSD[i] = 10 * Math.log10(PSD[i]) - 10; // scaling factor to make peak at 9 dB (8 linear)
  }
  return PSD;
}

function convolve(x, y) {
  let result = [];
  let lenX = x.length;
  let lenY = y.length;

  // Perform convolution, using "same" mode
  for (let i = 0; i < lenX + lenY - 1; i++) {
    let sum = 0;
    for (let j = Math.max(0, i - lenY + 1); j <= Math.min(i, lenX - 1); j++) {
      sum += x[j] * y[i - j];
    }
    result.push(sum);
  }
  return result;
}

function sinc(x) {
  return x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
}

// Standard Normal variate using Box-Muller transform.
function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random(); // Converting [0,1) to (0,1]
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  // Transform to the desired mean and standard deviation:
  return z * stdev + mean;
}

function generate_bspk(N, fs, sps, f_offset) {
  const bits = Array.from({ length: Math.ceil(N / sps) }, () => Math.floor(Math.random() * 2)); // Our data to be transmitted, 1's and 0's
  let bpsk = [];
  for (const bit of bits) {
    const pulse = new Array(sps).fill(0);
    pulse[0] = bit * 2 - 1; // set the first value to either a 1 or -1
    bpsk = bpsk.concat(pulse); // add the 8 samples to the signal
  }
  const num_taps = 101; // for our RRC filter
  let h;
  if (true) {
    // RC pulse shaping
    const beta = 0.249;
    const t = Array.from({ length: num_taps }, (_, i) => i - (num_taps - 1) / 2);
    h = t.map((val) => {
      return (sinc(val / sps) * Math.cos((Math.PI * beta * val) / sps)) / (1 - ((2 * beta * val) / sps) ** 2);
    });
  } else {
    // rect pulses
    h = new Array(sps).fill(1);
  }

  // Convolve bpsk and h
  bpsk = convolve(bpsk, h);
  bpsk = bpsk.slice(0, N); // clip off the extra samples

  // Freq shift, also is the start of it being complex, which is done with an interleaved 1d array that is twice the length
  const bpsk_complex = new Array(N * 2).fill(0);
  for (let i = 0; i < N; i++) {
    bpsk_complex[2 * i] = bpsk[i] * Math.cos((2 * Math.PI * f_offset * i) / fs);
    bpsk_complex[2 * i + 1] = bpsk[i] * Math.sin((2 * Math.PI * f_offset * i) / fs);
  }

  // Add noise
  for (let i = 0; i < N; i++) {
    bpsk_complex[2 * i] += gaussianRandom() * 0.1;
    bpsk_complex[2 * i + 1] += gaussianRandom() * 0.1;
  }

  return bpsk_complex;
}
