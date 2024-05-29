function cyclostationary_app() {
  // Params
  const fs = 1; // sample rate in Hz
  const N = Math.pow(2, 16); // number of samples to simulate, needs to be power of 2
  const Nw = 256; // window length
  const Noverlap = Math.floor((2 / 3) * Nw); // block overlap
  const alphas = [];
  for (let alpha = 0.05; alpha < 0.5; alpha += 0.005) {
    alphas.push(alpha);
  }

  parent = document.getElementById("sliders");

  // freq
  var span = document.createElement("span");
  span.className = "slider-span";
  span.append("Frequency [normalized Hz]");
  parent.appendChild(span);

  var input_element = document.createElement("input");
  input_element.type = "range";
  input_element.className = "slider";
  input_element.value = "0.2";
  input_element.min = "0";
  input_element.max = "0.5";
  input_element.step = "0.005";
  input_element.id = "freq";
  parent.appendChild(input_element);

  var span = document.createElement("span");
  span.className = "slider-span";
  span.id = "freq_label";
  span.textContent = "0.2";
  parent.appendChild(span);

  var br = document.createElement("br");
  parent.appendChild(br);

  // sps
  var span = document.createElement("span");
  span.className = "slider-span";
  span.append("Samples per Symbol [int]");
  parent.appendChild(span);

  var input_element = document.createElement("input");
  input_element.type = "range";
  input_element.className = "slider";
  input_element.value = "10";
  input_element.min = "2";
  input_element.max = "30";
  input_element.step = "1";
  input_element.id = "sps";
  parent.appendChild(input_element);

  var span = document.createElement("span");
  span.className = "slider-span";
  span.id = "sps_label";
  span.textContent = 10;
  parent.appendChild(span);

  // Display the image using html
  let canvas = document.createElement("canvas");
  canvas.height = Nw;
  canvas.width = alphas.length;
  canvas.style.width = "800px";
  let ctx = canvas.getContext("2d");
  let imgData = ctx.createImageData(Nw, alphas.length); // width, height
  document.getElementById("scf_img").appendChild(canvas);

  // Make slider change label
  document.getElementById("freq").addEventListener("input", function () {
    document.getElementById("freq_label").textContent = Math.round(this.value * 100) / 100;
    img = update_img(
      fs,
      N,
      Nw,
      Noverlap,
      alphas,
      parseInt(document.getElementById("sps").value),
      parseFloat(document.getElementById("freq").value) * fs
    );
    for (let i = 0; i < img.length; i++) {
      imgData.data[i * 4] = 0; // R
      imgData.data[i * 4 + 1] = img[i]; // G
      imgData.data[i * 4 + 2] = 0; // B
      imgData.data[i * 4 + 3] = 255; // alpha
      ctx.putImageData(imgData, 0, 0); // data, dx, dy
    }
  });

  // Make slider change label
  document.getElementById("sps").addEventListener("input", function () {
    document.getElementById("sps_label").textContent = Math.round(this.value);
    img = update_img(
      fs,
      N,
      Nw,
      Noverlap,
      alphas,
      parseInt(document.getElementById("sps").value),
      parseFloat(document.getElementById("freq").value) * fs
    );
    for (let i = 0; i < img.length; i++) {
      imgData.data[i * 4] = 0; // R
      imgData.data[i * 4 + 1] = img[i]; // G
      imgData.data[i * 4 + 2] = 0; // B
      imgData.data[i * 4 + 3] = 255; // alpha
      ctx.putImageData(imgData, 0, 0);
    }

    //Plotly.redraw("rectPlot");
  });

  // Run it once
  img = update_img(fs, N, Nw, Noverlap, alphas, 10, 0.2 * fs);
  for (let i = 0; i < img.length; i++) {
    imgData.data[i * 4] = 0; // R
    imgData.data[i * 4 + 1] = img[i]; // G
    imgData.data[i * 4 + 2] = 0; // B
    imgData.data[i * 4 + 3] = 255; // alpha
  }
  ctx.putImageData(imgData, 0, 0);

  /*
  // Plot PSD
  PSD = calc_PSD(samples);
  const f_vals = Array.from({ length: N }, (v, k) => k - N / 2).map((val) => (val * fs) / N);
  Plotly.newPlot("rectPlot", [{ x: f_vals, y: PSD }]);
  */
}

function update_img(fs, N, Nw, Noverlap, alphas, sps, f_offset) {
  var startTime = performance.now();
  const samples = generate_bspk(N, fs, sps, f_offset);
  /*
    const samples2 = generate_bspk(N, fs, 8, 0.07 * fs);
    for (let i = 0; i < N; i++) {
      samples[2 * i] += samples2[2 * i];
      samples[2 * i + 1] += samples2[2 * i + 1];
    }
    */
  console.log(`Calls to generate_bspk() took ${performance.now() - startTime} ms`); // 42ms

  startTime = performance.now();
  const SCF_mag = calc_SCF(samples, alphas, Nw, Noverlap);
  console.log(`Call to calc_SCF() took ${performance.now() - startTime} ms`); // 1500ms

  // Create an image out of SCF_mag
  let img = new Array(alphas.length * Nw).fill(0);
  for (let i = 0; i < alphas.length; i++) {
    for (let j = 0; j < Nw; j++) {
      img[i * Nw + j] = SCF_mag[i][j];
    }
  }

  // Print smallest value of SCF_mag
  console.log("Min SCF_mag: " + Math.min(...img));

  // Print largest value of SCF_mag
  console.log("Max SCF_mag: " + Math.max(...img));

  // Normalize/scale
  img = img.map((val) => val / Math.max(...img)); // Normalize img so max is 1
  img = img.map((val) => Math.max(val, 0)); // Truncate to 0
  img = img.map((val) => val * 255);

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
      if (alpha_idx === 0 && i === 1) {
        console.log(SCF[alpha_idx].slice(0, 20));
      }

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
  startTime = performance.now();
  let SCF_mag = Array.from({ length: alphas.length }, () => new Array(Nw));
  for (let i = 0; i < alphas.length; i++) {
    for (let j = 0; j < Nw; j++) {
      SCF_mag[i][j] = Math.sqrt(SCF[i][2 * j] * SCF[i][2 * j] + SCF[i][2 * j + 1] * SCF[i][2 * j + 1]);
    }
  }
  console.log(`calc magnitude took ${performance.now() - startTime} ms`); // 0.8ms
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

function generate_bspk(N, fs, sps, f_offset) {
  startTime = performance.now();
  const bits = Array.from({ length: Math.ceil(N / sps) }, () => Math.floor(Math.random() * 2)); // Our data to be transmitted, 1's and 0's
  //const bits = new Array(Math.ceil(N / sps)).fill(1);
  console.log(`making bits took ${performance.now() - startTime} ms`); // 0.5ms

  startTime = performance.now();
  //let bpsk = [];
  let bpsk = new Array(N).fill(0);
  for (let i = 0; i < bits.length; i++) {
    bpsk[i * sps] = bits[i] * 2 - 1; // BPSK
  }
  console.log(`making bpsk took ${performance.now() - startTime} ms`); // 0.5ms

  startTime = performance.now();
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

  console.log(`making h took ${performance.now() - startTime} ms`); // 0.2ms

  // Convolve bpsk and h
  startTime = performance.now();
  bpsk = convolve(bpsk, h);
  console.log(`convolve took ${performance.now() - startTime} ms`); // 9ms

  bpsk = bpsk.slice(0, N); // clip off the extra samples

  // Freq shift, also is the start of it being complex, which is done with an interleaved 1d array that is twice the length
  startTime = performance.now();
  const bpsk_complex = new Array(N * 2).fill(0);
  for (let i = 0; i < N; i++) {
    bpsk_complex[2 * i] = bpsk[i] * Math.cos((2 * Math.PI * f_offset * i) / fs);
    bpsk_complex[2 * i + 1] = bpsk[i] * Math.sin((2 * Math.PI * f_offset * i) / fs);
  }
  console.log(`freq shift took ${performance.now() - startTime} ms`); // 2ms

  // Add noise
  /*
  startTime = performance.now();
  for (let i = 0; i < N; i++) {
    bpsk_complex[2 * i] += gaussianRandom() * 0.1;
    bpsk_complex[2 * i + 1] += gaussianRandom() * 0.1;
  }
  console.log(`adding noise took ${performance.now() - startTime} ms`); // 6ms
*/

  return bpsk_complex;
}
