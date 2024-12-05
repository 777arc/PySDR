function homepage_app() {
  function createSignal(N) {
    const x = new Array(N * 2);

    // complex AWGN
    let noise_ampl_dB = document.getElementById("noise_ampl_dB").value;
    document.getElementById("noise_value").innerHTML = noise_ampl_dB - 30;
    let noise_ampl = Math.pow(10, noise_ampl_dB / 10);
    for (let i = 0; i < N; i++) {
      x[i * 2] = noise_ampl * Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
      x[i * 2 + 1] = noise_ampl * Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random());
    }

    // Tone
    let freq = document.getElementById("freq").value;
    document.getElementById("freq_value").innerHTML = freq;
    for (let i = 0; i < N; i++) {
      x[i * 2] += Math.cos(2 * Math.PI * i * freq);
      x[i * 2 + 1] += Math.sin(2 * Math.PI * i * freq);
    }

    return x;
  }

  const N = 1024;
  let update_period = 50; // in ms, gets doubled every time refersh is too slow to keep up

  function updatePlot() {
    const start_t = performance.now();
    const signal = createSignal(N);

    const fft_obj = new FFT(N);
    const signal_fft = fft_obj.createComplexArray();
    fft_obj.transform(signal_fft, signal);

    // Take magnitude of FFT
    const signal_fft_mag = new Array(N);
    for (let i = 0; i < N; i++) {
      signal_fft_mag[i] = signal_fft[2 * i] * signal_fft[2 * i] + signal_fft[2 * i + 1] * signal_fft[2 * i + 1];
    }
    let signal_fft_mag_shifted = fftshift(signal_fft_mag);

    // Convert to dB
    const signal_fft_mag_shifted_dB = new Array(N);
    for (let i = 0; i < N; i++) {
      signal_fft_mag_shifted_dB[i] = 10 * Math.log10(signal_fft_mag_shifted[i]);
    }

    // Plot freq
    const canvas = document.getElementById("freq_plot");
    const ctx = canvas.getContext("2d", { alpha: false }); // apparently turning off transparency makes it faster

    ctx.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx.fillStyle = "white";
    ctx.lineWidth = 1;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.translate(0, 400); // move half of canvas height so y=0 in middle
    ctx.beginPath();
    ctx.strokeStyle = "blue";
    ctx.moveTo(0, Math.floor(-7 * signal_fft_mag_shifted_dB[0] + 200));
    for (let i = 1; i < N; i++) {
      ctx.lineTo(i * 2, Math.floor(-7 * signal_fft_mag_shifted_dB[i] + 200)); // -1* to flip y-axis
    }
    ctx.stroke();

    // Freq x-axis ticks and labels
    ctx.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;
    ctx.font = "36px Arial";
    ctx.fillStyle = "black";
    // axis line
    ctx.moveTo(0, ctx.canvas.height - 80);
    ctx.lineTo(ctx.canvas.width, ctx.canvas.height - 80);
    // ticks
    for (let i = 0; i < 11; i++) {
      ctx.moveTo((ctx.canvas.width / 10) * i, ctx.canvas.height - 100);
      ctx.lineTo((ctx.canvas.width / 10) * i, ctx.canvas.height - 80);
      ctx.fillText(Math.round((i - 5) * 0.1 * 100) / 100, ((ctx.canvas.width / 10) * i - 0) * 0.975, ctx.canvas.height - 35);
    }
    ctx.fillText("Hz", ctx.canvas.width / 2 + 5, ctx.canvas.height - 35);
    ctx.fillText("Frequency", ctx.canvas.width / 2 - 70, ctx.canvas.height - 7);
    ctx.stroke();

    // Freq y-axis ticks and labels
    ctx.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;
    ctx.font = "36px Arial";
    ctx.fillStyle = "black";
    // axis line
    ctx.moveTo(0, 0);
    ctx.lineTo(0, ctx.canvas.height - 80);
    // ticks
    for (let i = 1; i < 6; i++) {
      ctx.moveTo(0, ((ctx.canvas.height - 80) / 6) * i);
      ctx.lineTo(20, ((ctx.canvas.height - 80) / 6) * i);
      ctx.fillText(i * -10, 30, ((ctx.canvas.height - 80) / 6) * i + 10);
    }
    ctx.fillText("dB", 20, 36);
    ctx.stroke();

    // Plot time
    const canvas_time = document.getElementById("time_plot");
    const ctx_time = canvas_time.getContext("2d", { alpha: false });

    ctx_time.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx_time.fillStyle = "white";
    ctx_time.lineWidth = 1;
    ctx_time.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx_time.translate(0, 400); // move half of canvas height so y=0 in middle
    ctx_time.beginPath();
    ctx_time.strokeStyle = "blue";
    ctx_time.moveTo(0, Math.floor(-20 * signal[0] - 40));
    for (let i = 1; i < N; i++) {
      ctx_time.lineTo(i * 2, Math.floor(-20 * signal[i * 2] - 40)); // -1* to flip y-axis
    }
    ctx_time.stroke();

    // Time x-axis ticks and labels
    ctx_time.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx_time.beginPath();
    ctx_time.strokeStyle = "black";
    ctx_time.lineWidth = 3;
    ctx_time.font = "36px Arial";
    ctx_time.fillStyle = "black";
    // axis line
    ctx_time.moveTo(0, ctx_time.canvas.height - 60);
    ctx_time.lineTo(ctx_time.canvas.width, ctx_time.canvas.height - 60);
    // ticks
    for (let i = 0; i < 11; i++) {
      ctx_time.moveTo((ctx_time.canvas.width / 10) * i, ctx_time.canvas.height - 80);
      ctx_time.lineTo((ctx_time.canvas.width / 10) * i, ctx_time.canvas.height - 60);
      //ctx_time.fillText(Math.round(i* 0.1 * 100) / 100, ((ctx_time.canvas.width / 10) * i - 0) * 0.98, ctx_time.canvas.height - 5);
    }
    ctx_time.fillText("Time", ctx_time.canvas.width / 2, ctx_time.canvas.height - 5);
    ctx_time.stroke();

    // Time y-axis ticks and labels
    ctx_time.setTransform(1, 0, 0, 1, 0, 0); // resets transform
    ctx_time.beginPath();
    ctx_time.strokeStyle = "black";
    ctx_time.lineWidth = 3;
    ctx_time.font = "36px Arial";
    ctx_time.fillStyle = "black";
    // axis line
    ctx_time.moveTo(0, 0);
    ctx_time.lineTo(0, ctx_time.canvas.height - 80);
    // ticks
    for (let i = 1; i < 6; i++) {
      ctx_time.moveTo(0, ((ctx_time.canvas.height - 80) / 6) * i);
      ctx_time.lineTo(20, ((ctx_time.canvas.height - 80) / 6) * i);
    }
    ctx_time.fillText("1", 20, 36);
    ctx_time.fillText("-1", 20, ctx_time.canvas.height - 80);
    ctx_time.fillText("0", 30, ctx_time.canvas.height / 2 - 40);
    ctx_time.stroke();
    // y=0 line
    ctx_time.strokeStyle = "grey";
    ctx_time.lineWidth = 1;
    ctx_time.moveTo(0, (ctx_time.canvas.height - 80) / 2);
    ctx_time.lineTo(ctx_time.canvas.width, (ctx_time.canvas.height - 80) / 2);
    ctx_time.stroke();

    //console.log("Time taken to update frame: " + (performance.now() - start_t) + " ms");
    if (performance.now() - start_t > update_period) {
      console.log("Warning: browser is not able to keep up, doubling update period");
      update_period = update_period * 2;
    }
  }

  setInterval(function () {
    updatePlot();
  }, update_period); // in ms
}
