function cyclostationary_app() {
  // Params
  const fs = 1; // sample rate in Hz
  const N = Math.pow(2, 15); // number of samples to simulate, needs to be power of 2
  const Nw = 256; // window length
  //const Noverlap = Math.floor((2 / 3) * Nw); // block overlap
  const Noverlap = 0; // block overlap
  let alphas = [];
  for (let alpha = 0.05; alpha < 0.5; alpha += 0.002) {
    alphas.push(alpha);
  }
  console.log("Number of alphas: " + alphas.length);

  // Display the image using html
  let canvas = document.getElementById("scf_canvas");
  canvas.width = Nw;
  canvas.height = alphas.length;
  let ctx = canvas.getContext("2d");
  let imgData = ctx.createImageData(Nw, alphas.length); // width, height

  // Make slider change label
  document.getElementById("mainform").addEventListener("submit", (e) => {
    e.preventDefault();
    rect_checked = document.getElementById("rect").checked;
    document.getElementById("rolloff").disabled = rect_checked;
    let alphas = [];
    for (
      let alpha = parseFloat(document.getElementById("alpha_start").value);
      alpha < parseFloat(document.getElementById("alpha_stop").value);
      alpha += parseFloat(document.getElementById("alpha_step").value)
    ) {
      alphas.push(alpha);
    }
    canvas.height = alphas.length;
    let imgData = ctx.createImageData(Nw, alphas.length); // width, height
    console.log("Number of alphas: " + alphas.length);
    img = update_img(
      fs,
      N,
      Nw,
      Noverlap,
      alphas,
      parseInt(document.getElementById("sps").value),
      parseFloat(document.getElementById("freq").value) * fs,
      parseFloat(document.getElementById("rolloff").value),
      parseFloat(document.getElementById("noise").value),
      rect_checked
    );
    for (let i = 0; i < img.length; i++) {
      imgData.data[i * 4] = 0; // R
      imgData.data[i * 4 + 1] = img[i]; // G
      imgData.data[i * 4 + 2] = 0; // B
      imgData.data[i * 4 + 3] = 255; // alpha
    }
    ctx.putImageData(imgData, 0, 0); // data, dx, dy
    //Plotly.redraw("rectPlot");
  });

  // Run it once
  startTime = performance.now();
  img = update_img(fs, N, Nw, Noverlap, alphas, 10, 0.2 * fs, 0.5, 0.1, false);
  for (let i = 0; i < img.length; i++) {
    imgData.data[i * 4] = 0; // R
    imgData.data[i * 4 + 1] = img[i]; // G
    imgData.data[i * 4 + 2] = 0; // B
    imgData.data[i * 4 + 3] = 255; // alpha
  }
  ctx.putImageData(imgData, 0, 0);
  console.log(`first run took ${performance.now() - startTime} ms`);

  /*
  // Plot PSD
  PSD = calc_PSD(samples);
  const f_vals = Array.from({ length: N }, (v, k) => k - N / 2).map((val) => (val * fs) / N);
  Plotly.newPlot("rectPlot", [{ x: f_vals, y: PSD }]);
  */
}

function update_img(fs, N, Nw, Noverlap, alphas, sps, f_offset, rolloff, noise_level, rect_checked) {
  const samples = generate_bspk(N, fs, sps, f_offset, rolloff, noise_level, rect_checked);
  /*
    const samples2 = generate_bspk(N, fs, 8, 0.07 * fs);
    for (let i = 0; i < N; i++) {
      samples[2 * i] += samples2[2 * i];
      samples[2 * i + 1] += samples2[2 * i + 1];
    }
    */

  const SCF_mag = calc_SCF(samples, alphas, Nw, Noverlap);

  // Create an image out of SCF_mag (converts it from 2d to 1d). FIXME I could probably just have it 1D from the start
  let img = new Array(alphas.length * Nw).fill(0);
  for (let i = 0; i < alphas.length; i++) {
    for (let j = 0; j < Nw; j++) {
      img[i * Nw + j] = SCF_mag[i][j];
    }
  }

  // Print largest value of SCF_mag
  const max_val = Math.max(...img);
  //console.log("Max SCF_mag: " + max_val);
  //console.log("Min SCF_mag: " + Math.min(...img));

  // Normalize/scale
  for (let i = 0; i < img.length; i++) {
    img[i] = (img[i] / max_val) * 255;
  }
  return img;
}

function calc_SCF(samples, alphas, Nw, Noverlap) {
  N = samples.length / 2;

  // SCF
  const num_windows = Math.floor((N - Noverlap) / (Nw - Noverlap)); // Number of windows
  const window = new Array(Nw);
  for (let i = 0; i < Nw; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (Nw - 1)));
  }

  // outter array is for each alpha, inner array is for each window
  let SCF = Array.from({ length: alphas.length }, () => new Array(Nw * 2).fill(0));

  // Prep
  let neg = new Array(N * 2);
  let pos = new Array(N * 2);

  let fft_obj_pos = new FFT(Nw);
  let fft_obj_neg = new FFT(Nw);
  let neg_out = fft_obj_neg.createComplexArray();
  let pos_out = fft_obj_pos.createComplexArray();

  // loop through cyclic freq (alphas)
  for (let alpha_idx = 0; alpha_idx < alphas.length; alpha_idx++) {
    const alpha_times_pi = alphas[alpha_idx] * Math.PI;

    for (let i = 0; i < N; i++) {
      // remember (a + ib)(c + id) = (ac - bd) + i(ad + bc).
      neg[2 * i] = samples[2 * i] * Math.cos(-1 * alpha_times_pi * i) - samples[2 * i + 1] * Math.sin(-1 * alpha_times_pi * i);
      neg[2 * i + 1] = samples[2 * i] * Math.sin(-1 * alpha_times_pi * i) + samples[2 * i + 1] * Math.cos(-1 * alpha_times_pi * i);
      pos[2 * i] = samples[2 * i] * Math.cos(alpha_times_pi * i) - samples[2 * i + 1] * Math.sin(alpha_times_pi * i);
      pos[2 * i + 1] = samples[2 * i] * Math.sin(alpha_times_pi * i) + samples[2 * i + 1] * Math.cos(alpha_times_pi * i);
    }

    // Cross Cyclic Power Spectrum
    for (let i = 0; i < num_windows; i++) {
      let pos_slice = pos.slice(2 * i * (Nw - Noverlap), 2 * i * (Nw - Noverlap) + 2 * Nw); // 2* because of how we store complex
      let neg_slice = neg.slice(2 * i * (Nw - Noverlap), 2 * i * (Nw - Noverlap) + 2 * Nw);

      // Apply window
      //pos_slice = pos_slice.map((val, idx) => val * window[idx]);
      //neg_slice = neg_slice.map((val, idx) => val * window[idx]);

      // Take FFTs
      fft_obj_neg.transform(neg_out, neg_slice);
      fft_obj_pos.transform(pos_out, pos_slice);

      // Multiply neg_fft with complex conjugate of pos_fft
      for (let j = 0; j < Nw; j++) {
        SCF[alpha_idx][2 * j] += neg_out[2 * j] * pos_out[2 * j] + neg_out[2 * j + 1] * pos_out[2 * j + 1];
        SCF[alpha_idx][2 * j + 1] += neg_out[2 * j + 1] * pos_out[2 * j] - neg_out[2 * j] * pos_out[2 * j + 1]; // includes the conj
      }
    }
  }

  // Take magnitude of SCF
  let SCF_mag = Array.from({ length: alphas.length }, () => new Array(Nw));
  for (let i = 0; i < alphas.length; i++) {
    for (let j = 0; j < Nw; j++) {
      SCF_mag[i][j] = Math.sqrt(SCF[i][2 * j] * SCF[i][2 * j] + SCF[i][2 * j + 1] * SCF[i][2 * j + 1]);
    }
  }
  return SCF_mag;
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

function generate_bspk(N, fs, sps, f_offset, rolloff, noise_level, rect_checked) {
  const bits = Array.from({ length: Math.ceil(N / sps) }, () => Math.floor(Math.random() * 2)); // Our data to be transmitted, 1's and 0's
  //const bits = new Array(Math.ceil(N / sps)).fill(1);

  //startTime = performance.now();
  //let bpsk = [];
  let bpsk = new Array(N).fill(0);
  for (let i = 0; i < bits.length; i++) {
    bpsk[i * sps] = bits[i] * 2 - 1; // BPSK
  }
  //console.log(`making bpsk took ${performance.now() - startTime} ms`); // 0.5ms

  const num_taps = 101; // for our RRC filter
  let h;
  // easier than adding the math
  if (rolloff == 0.5) rolloff = 0.4999;
  if (rolloff == 0.25) rolloff = 0.2499;
  if (rolloff == 1) rolloff = 0.9999;
  if (rect_checked) {
    // rect pulses
    h = new Array(sps).fill(1);
  } else {
    // RC pulse shaping
    const t = Array.from({ length: num_taps }, (_, i) => i - (num_taps - 1) / 2);
    h = t.map((val) => {
      return (sinc(val / sps) * Math.cos((Math.PI * rolloff * val) / sps)) / (1 - ((2 * rolloff * val) / sps) ** 2);
    });
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

  // Add complex white gaussian noise
  for (let i = 0; i < N; i++) {
    bpsk_complex[2 * i] += gaussianRandom() * noise_level;
    bpsk_complex[2 * i + 1] += gaussianRandom() * noise_level;
  }

  return bpsk_complex;
}
