function cyclostationary_app() {
  // Params
  const fs = 1; // sample rate in Hz
  const Nw = 256; // window length

  // Display the image using html
  const canvas = document.getElementById("scf_canvas");
  const ctx = canvas.getContext("2d");

  document.getElementById("freq").addEventListener("input", function () {
    document.getElementById("freq_display").textContent = Math.round(this.value * 100) / 100;
    document.getElementById("submit_button").click();
  });

  document.getElementById("sps").addEventListener("input", function () {
    document.getElementById("sps_display").textContent = this.value;
    document.getElementById("submit_button").click();
  });

  // Reset button. MAKE SURE THESE ALWASY MATCH THE DEFAULTS IN THE HTML!
  document.getElementById("resetform").addEventListener("submit", (e) => {
    e.preventDefault();
    document.getElementById("N").value = "8192";
    document.getElementById("freq").value = "0.2";
    document.getElementById("freq_display").textContent = "0.2";
    document.getElementById("sps").value = "20";
    document.getElementById("sps_display").textContent = "20";
    document.getElementById("rolloff").value = "0.5";
    document.getElementById("rect").checked = true;
    document.getElementById("alpha_start").value = "0";
    document.getElementById("alpha_stop").value = "0.3";
    document.getElementById("alpha_step").value = "0.001";
    document.getElementById("noise").value = "0.001";

    document.getElementById("submit_button").click();
  });

  // Make slider change label
  document.getElementById("mainform").addEventListener("submit", (e) => {
    e.preventDefault();

    startTime = performance.now();

    const N = parseInt(document.getElementById("N").value);
    const sps = parseInt(document.getElementById("sps").value);
    const f_offset = parseFloat(document.getElementById("freq").value) * fs;
    const rolloff = parseFloat(document.getElementById("rolloff").value);
    const noise_level = parseFloat(document.getElementById("noise").value);
    const alpha_start = parseFloat(document.getElementById("alpha_start").value);
    const alpha_stop = parseFloat(document.getElementById("alpha_stop").value); // inclusive
    const alpha_step = parseFloat(document.getElementById("alpha_step").value);

    rect_checked = document.getElementById("rect").checked;
    document.getElementById("rolloff").disabled = rect_checked;
    let alphas = [];
    for (let alpha = alpha_start; alpha <= alpha_stop; alpha += alpha_step) {
      alphas.push(alpha);
    }
    if (alphas[0] == 0) {
      alphas[0] = alphas[1]; // avoid calc at alpha=0 or it throws off color scale
    }

    const samples = generate_bspk(N, fs, sps, f_offset, rolloff, noise_level, rect_checked);

    const SCF_mag = calc_SCF_time_smoothing(samples, alphas, Nw);
    const num_alphas = SCF_mag.length; // cyclic domain
    const num_freqs = SCF_mag[0].length; // RF freq domain

    scales_width = 35;
    scales_height = 25;
    canvas.width = num_freqs + scales_width + 10; // little at the end for text to use
    canvas.height = num_alphas + scales_height + 10; // little at the end for text to use
    canvas.style.width = String((num_freqs + scales_width + 10) * 2) + "px";
    canvas.style.height = String((num_alphas + scales_height + 10) * 2) + "px";
    let imgData = ctx.createImageData(num_freqs, num_alphas); // width, height
    console.log("Number of alphas: " + num_alphas);

    // Create an image out of SCF_mag (converts it from 2d to 1d). FIXME I could probably just have it 1D from the start
    let img = new Array(num_alphas * num_freqs).fill(0);
    let max_val = 0;
    for (let i = 0; i < num_alphas; i++) {
      for (let j = 0; j < num_freqs; j++) {
        img[i * num_freqs + j] = SCF_mag[i][j];
        max_val = Math.max(max_val, SCF_mag[i][j]);
      }
    }

    //console.log("Max SCF_mag: " + max_val);
    //console.log("Min SCF_mag: " + Math.min(...img));

    // Normalize/scale so that half the max is 255
    for (let i = 0; i < img.length; i++) {
      img[i] = Math.round((img[i] / max_val) * 2 * 255);
    }
    // Truncate to 0 to 255
    img = img.map((val) => Math.min(255, Math.max(0, val)));

    for (let i = 0; i < img.length; i++) {
      imgData.data[i * 4] = viridis[img[i]][0]; // R
      imgData.data[i * 4 + 1] = viridis[img[i]][1]; // G
      imgData.data[i * 4 + 2] = viridis[img[i]][2]; // B
      imgData.data[i * 4 + 3] = 255; // alpha
    }
    ctx.putImageData(imgData, scales_width, scales_height); // data, dx, dy

    // Add scales
    ctx.font = "12px serif";
    for (let i = 0; i <= 10; i++) {
      // horizontal scales
      ctx.beginPath();
      ctx.moveTo(Math.round(num_freqs * (i / 10) + scales_width) - 0.5, scales_height); // all the 0.5 are to make the lines not fuzzy, you have to straddle pixels
      ctx.lineTo(Math.round(num_freqs * (i / 10) + scales_width) - 0.5, Math.round(num_alphas * 0.02) + scales_height);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#ff6699";
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(Math.round((i / 10 - 0.5)*100)/100, Math.round(num_freqs * (i / 10) + scales_width) - 0.5, scales_height - 3);
    }

    for (let i = 0; i <= 10; i++) {
      // vertical scales
      ctx.beginPath();
      ctx.moveTo(scales_width, Math.round(num_alphas * (i / 10)) + scales_height - 0.5);
      ctx.lineTo(Math.round(num_freqs * 0.02 + scales_width), Math.round(num_alphas * (i / 10)) + scales_height - 0.5);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#ff6699";
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(
        Math.round(((i / 10) * (alpha_stop - alpha_start) + alpha_start) * 100) / 100,
        0.5 + scales_width - 2,
        num_alphas * (i / 10) + scales_height + 5 + 0.5
      );
    }

    ctx.textAlign = "center";
    ctx.fillText("Frequency [Hz]", num_freqs / 2 + scales_width, 9);

    // Make sure this is the last plotting, because it rotates the context
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Alpha [Hz]", num_alphas / -2 - scales_height, 9);

    console.log(`Processing took ${performance.now() - startTime} ms`); // 0.5ms
    //Plotly.redraw("rectPlot");
  });

  // Run it once
  document.getElementById("submit_button").click();

  /*
  // Plot PSD
  PSD = calc_PSD(samples);
  const f_vals = Array.from({ length: N }, (v, k) => k - N / 2).map((val) => (val * fs) / N);
  Plotly.newPlot("rectPlot", [{ x: f_vals, y: PSD }]);
  */
}

function arrayRotate(arr, count) {
  const len = arr.length;
  arr.push(...arr.splice(0, ((-count % len) + len) % len));
  return arr;
}

// SCF with time smoothing method (lots of FFTs)
function calc_SCF_time_smoothing(samples, alphas, Nw) {
  const N = samples.length / 2;

  // SCF
  const num_windows = Math.floor(N / Nw); // Number of windows

  //const window = new Array(Nw);
  //for (let i = 0; i < Nw; i++) {
  //  window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (Nw - 1)));
  //}

  // outter array is for each alpha, inner array is for each window
  const SCF = Array.from({ length: alphas.length }, () => new Array(Nw * 2).fill(0)); // need it to start at 0

  // Prep
  const neg = new Array(N * 2);
  const pos = new Array(N * 2);

  const fft_obj_pos = new FFT(Nw);
  const fft_obj_neg = new FFT(Nw);
  const neg_out = fft_obj_neg.createComplexArray();
  const pos_out = fft_obj_pos.createComplexArray();

  // loop through cyclic freq (alphas)
  for (let alpha_idx = 0; alpha_idx < alphas.length; alpha_idx++) {
    const alpha_times_pi = alphas[alpha_idx] * Math.PI;

    for (let i = 0; i < N; i++) {
      // below has been heavily optimized, see python for simpler version of whats going on
      const cos_term = Math.cos(alpha_times_pi * i);
      const sin_term = Math.sin(alpha_times_pi * i);
      const a = samples[2 * i] * cos_term;
      const b = samples[2 * i + 1] * sin_term;
      const c = samples[2 * i + 1] * cos_term;
      const d = samples[2 * i] * sin_term;
      neg[2 * i] = a + b;
      neg[2 * i + 1] = c - d;
      pos[2 * i] = a - b;
      pos[2 * i + 1] = d + c;
    }

    // Cross Cyclic Power Spectrum
    for (let i = 0; i < num_windows; i++) {
      const pos_slice = pos.slice(2 * i * Nw, 2 * i * Nw + 2 * Nw); // 2* because of how we store complex
      const neg_slice = neg.slice(2 * i * Nw, 2 * i * Nw + 2 * Nw);

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
      SCF_mag[i][j] = SCF[i][2 * j] * SCF[i][2 * j] + SCF[i][2 * j + 1] * SCF[i][2 * j + 1];
    }
    SCF_mag[i] = fftshift(SCF_mag[i]);
  }
  return SCF_mag;
}

// SCF using the freq smoothing method (1 FFT, lots of convolves)
function calc_SCF_freq_smoothing(samples, alphas, Nw) {
  const N = samples.length / 2;

  const window = Array.from({ length: Nw }, (_, i) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / (Nw - 1)))); // hanning window

  // FFT entire signal
  let fft_obj = new FFT(N);
  let X = fft_obj.createComplexArray(); // output of fft
  fft_obj.transform(X, samples);

  // separate into real and imag
  const X_real = new Array(N);
  const X_imag = new Array(N);
  const X_real_rev = new Array(N);
  const X_imag_rev = new Array(N);
  for (let i = 0; i < N; i++) {
    X_real[i] = X[i * 2];
    X_imag[i] = X[i * 2 + 1];
    X_real_rev[i] = X[i * 2];
    X_imag_rev[i] = X[i * 2 + 1];
  }

  const freq_decimation = Math.floor((N / Nw) * 8); // sort of arbitrary but if we dont decimate there will be thousands of pixels horizontally
  const skip = Math.floor(N / freq_decimation);

  // outter array is for each alpha, inner array will hold magnitudes
  let SCF_mag = Array.from({ length: alphas.length }, () => new Array(freq_decimation));

  // loop through cyclic freq (alphas)
  let prev_shift = 0;

  for (let alpha_idx = 0; alpha_idx < alphas.length; alpha_idx++) {
    const shift = Math.floor((alphas[alpha_idx] * N) / 2); // scalar, number of samples to shift by
    arrayRotate(X_real, shift - prev_shift);
    arrayRotate(X_imag, shift - prev_shift);
    arrayRotate(X_real_rev, -1 * (shift - prev_shift));
    arrayRotate(X_imag_rev, -1 * (shift - prev_shift));
    prev_shift = shift;

    let real_part = new Array(N);
    let imag_part = new Array(N);
    for (let i = 0; i < N; i++) {
      // TODO: based on the code Sam had, might not need to calc for all i
      // includes the conj of the non-rev part (X_imag), otherwise its just a complex multiply
      real_part[i] = X_real_rev[i] * X_real[i] + X_imag_rev[i] * X_imag[i];
      imag_part[i] = X_imag_rev[i] * X_real[i] - X_real_rev[i] * X_imag[i];
    }

    // Apply window (a critical part to the freq smoothing method!)
    real_part = convolve(real_part, window).slice(Nw / 2, N + Nw / 2); // slice makes it like "same" mode
    imag_part = convolve(imag_part, window).slice(Nw / 2, N + Nw / 2);

    // Take magnitude squared but also decimate by Nw
    for (let i = 0; i < freq_decimation; i++) {
      SCF_mag[alpha_idx][i] = real_part[i * skip] * real_part[i * skip] + imag_part[i * skip] * imag_part[i * skip];
    }
    // FFT Shift
    SCF_mag[alpha_idx] = fftshift(SCF_mag[alpha_idx]);
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

  // Perform convolution, using "full" mode
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

const viridis = [
  [68, 1, 84],
  [68, 2, 86],
  [69, 4, 87],
  [69, 5, 89],
  [70, 7, 90],
  [70, 8, 92],
  [70, 10, 93],
  [70, 11, 94],
  [71, 13, 96],
  [71, 14, 97],
  [71, 16, 99],
  [71, 17, 100],
  [71, 19, 101],
  [72, 20, 103],
  [72, 22, 104],
  [72, 23, 105],
  [72, 24, 106],
  [72, 26, 108],
  [72, 27, 109],
  [72, 28, 110],
  [72, 29, 111],
  [72, 31, 112],
  [72, 32, 113],
  [72, 33, 115],
  [72, 35, 116],
  [72, 36, 117],
  [72, 37, 118],
  [72, 38, 119],
  [72, 40, 120],
  [72, 41, 121],
  [71, 42, 122],
  [71, 44, 122],
  [71, 45, 123],
  [71, 46, 124],
  [71, 47, 125],
  [70, 48, 126],
  [70, 50, 126],
  [70, 51, 127],
  [70, 52, 128],
  [69, 53, 129],
  [69, 55, 129],
  [69, 56, 130],
  [68, 57, 131],
  [68, 58, 131],
  [68, 59, 132],
  [67, 61, 132],
  [67, 62, 133],
  [66, 63, 133],
  [66, 64, 134],
  [66, 65, 134],
  [65, 66, 135],
  [65, 68, 135],
  [64, 69, 136],
  [64, 70, 136],
  [63, 71, 136],
  [63, 72, 137],
  [62, 73, 137],
  [62, 74, 137],
  [62, 76, 138],
  [61, 77, 138],
  [61, 78, 138],
  [60, 79, 138],
  [60, 80, 139],
  [59, 81, 139],
  [59, 82, 139],
  [58, 83, 139],
  [58, 84, 140],
  [57, 85, 140],
  [57, 86, 140],
  [56, 88, 140],
  [56, 89, 140],
  [55, 90, 140],
  [55, 91, 141],
  [54, 92, 141],
  [54, 93, 141],
  [53, 94, 141],
  [53, 95, 141],
  [52, 96, 141],
  [52, 97, 141],
  [51, 98, 141],
  [51, 99, 141],
  [50, 100, 142],
  [50, 101, 142],
  [49, 102, 142],
  [49, 103, 142],
  [49, 104, 142],
  [48, 105, 142],
  [48, 106, 142],
  [47, 107, 142],
  [47, 108, 142],
  [46, 109, 142],
  [46, 110, 142],
  [46, 111, 142],
  [45, 112, 142],
  [45, 113, 142],
  [44, 113, 142],
  [44, 114, 142],
  [44, 115, 142],
  [43, 116, 142],
  [43, 117, 142],
  [42, 118, 142],
  [42, 119, 142],
  [42, 120, 142],
  [41, 121, 142],
  [41, 122, 142],
  [41, 123, 142],
  [40, 124, 142],
  [40, 125, 142],
  [39, 126, 142],
  [39, 127, 142],
  [39, 128, 142],
  [38, 129, 142],
  [38, 130, 142],
  [38, 130, 142],
  [37, 131, 142],
  [37, 132, 142],
  [37, 133, 142],
  [36, 134, 142],
  [36, 135, 142],
  [35, 136, 142],
  [35, 137, 142],
  [35, 138, 141],
  [34, 139, 141],
  [34, 140, 141],
  [34, 141, 141],
  [33, 142, 141],
  [33, 143, 141],
  [33, 144, 141],
  [33, 145, 140],
  [32, 146, 140],
  [32, 146, 140],
  [32, 147, 140],
  [31, 148, 140],
  [31, 149, 139],
  [31, 150, 139],
  [31, 151, 139],
  [31, 152, 139],
  [31, 153, 138],
  [31, 154, 138],
  [30, 155, 138],
  [30, 156, 137],
  [30, 157, 137],
  [31, 158, 137],
  [31, 159, 136],
  [31, 160, 136],
  [31, 161, 136],
  [31, 161, 135],
  [31, 162, 135],
  [32, 163, 134],
  [32, 164, 134],
  [33, 165, 133],
  [33, 166, 133],
  [34, 167, 133],
  [34, 168, 132],
  [35, 169, 131],
  [36, 170, 131],
  [37, 171, 130],
  [37, 172, 130],
  [38, 173, 129],
  [39, 173, 129],
  [40, 174, 128],
  [41, 175, 127],
  [42, 176, 127],
  [44, 177, 126],
  [45, 178, 125],
  [46, 179, 124],
  [47, 180, 124],
  [49, 181, 123],
  [50, 182, 122],
  [52, 182, 121],
  [53, 183, 121],
  [55, 184, 120],
  [56, 185, 119],
  [58, 186, 118],
  [59, 187, 117],
  [61, 188, 116],
  [63, 188, 115],
  [64, 189, 114],
  [66, 190, 113],
  [68, 191, 112],
  [70, 192, 111],
  [72, 193, 110],
  [74, 193, 109],
  [76, 194, 108],
  [78, 195, 107],
  [80, 196, 106],
  [82, 197, 105],
  [84, 197, 104],
  [86, 198, 103],
  [88, 199, 101],
  [90, 200, 100],
  [92, 200, 99],
  [94, 201, 98],
  [96, 202, 96],
  [99, 203, 95],
  [101, 203, 94],
  [103, 204, 92],
  [105, 205, 91],
  [108, 205, 90],
  [110, 206, 88],
  [112, 207, 87],
  [115, 208, 86],
  [117, 208, 84],
  [119, 209, 83],
  [122, 209, 81],
  [124, 210, 80],
  [127, 211, 78],
  [129, 211, 77],
  [132, 212, 75],
  [134, 213, 73],
  [137, 213, 72],
  [139, 214, 70],
  [142, 214, 69],
  [144, 215, 67],
  [147, 215, 65],
  [149, 216, 64],
  [152, 216, 62],
  [155, 217, 60],
  [157, 217, 59],
  [160, 218, 57],
  [162, 218, 55],
  [165, 219, 54],
  [168, 219, 52],
  [170, 220, 50],
  [173, 220, 48],
  [176, 221, 47],
  [178, 221, 45],
  [181, 222, 43],
  [184, 222, 41],
  [186, 222, 40],
  [189, 223, 38],
  [192, 223, 37],
  [194, 223, 35],
  [197, 224, 33],
  [200, 224, 32],
  [202, 225, 31],
  [205, 225, 29],
  [208, 225, 28],
  [210, 226, 27],
  [213, 226, 26],
  [216, 226, 25],
  [218, 227, 25],
  [221, 227, 24],
  [223, 227, 24],
  [226, 228, 24],
  [229, 228, 25],
  [231, 228, 25],
  [234, 229, 26],
  [236, 229, 27],
  [239, 229, 28],
  [241, 229, 29],
  [244, 230, 30],
  [246, 230, 32],
  [248, 230, 33],
  [251, 231, 35],
  [253, 231, 37]
];
