function beamforming_slider_app() {
  // Init FFT
  const fftsize = 1024;
  const f = new FFT(fftsize);
  const input = new Array(fftsize * 2);
  const out = f.createComplexArray();

  // Map the FFT bins to angles in radians, in python its np.arcsin(np.linspace(-1, 1, fftsize))
  let bins = Array.from({ length: fftsize }, (v, k) => k + 1);
  bins = bins.map((x) => (x - fftsize / 2) / (fftsize / 2));
  bins = bins.map((x) => Math.asin(x));
  bins = bins.map((x) => (x * 180) / Math.PI); // convert from radians to degrees

  // Plotly
  var trace1 = {
    x: bins,
    y: Array.from({ length: fftsize }, (v, k) => k + 1),
    type: "scatter"
  };

  var data = [trace1];

  var layout = {
    autosize: false,
    width: 500,
    height: 300,
    margin: {
      l: 50,
      r: 0,
      b: 40,
      t: 0,
      pad: 0
    },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: {
      gridcolor: "black",
      dtick: 20,
      range: [-90, 90],
      title: {
        text: "Angle [Degrees]",
        font: {
          size: 14
        }
      }
    },
    yaxis: {
      gridcolor: "black",
      dtick: 5,
      range: [-40, 10],
      title: {
        text: "Beam Pattern [dB]",
        font: {
          size: 14
        }
      }
    }
  };

  Plotly.newPlot("rectPlot", data, layout);

  function calculate_beam_pattern() {
    input.fill(0);
    for (var i = 0; i < 8; i++) {
      input[i * 2] = document.getElementById("mag" + i).value * Math.cos(document.getElementById("phase" + i).value);
      input[i * 2 + 1] = document.getElementById("mag" + i).value * Math.sin(document.getElementById("phase" + i).value) * -1; // complex conjugate each value
    }

    // FFT
    f.transform(out, input);

    // Calculate magnitude
    let fftmag = new Array(fftsize);
    for (var i = 0; i < fftsize; i++) {
      fftmag[i] = Math.sqrt(out[i * 2] * out[i * 2] + out[i * 2 + 1] * out[i * 2 + 1]);
    }

    // Square the signal
    for (var i = 0; i < fftsize; i++) {
      fftmag[i] = fftmag[i] * fftmag[i];
    }

    fftmag = fftshift(fftmag);

    // Convert to dB and apply scaling factor
    for (var i = 0; i < fftsize; i++) {
      fftmag[i] = 10 * Math.log10(fftmag[i]) - 10; // scaling factor to make peak at 9 dB (8 linear)
    }
    return fftmag;
  }

  parent = document.getElementById("sliders");
  for (var i = 0; i < 8; i++) {
    var span = document.createElement("span");
    span.className = "slider-span";
    span.append(String(i));
    parent.appendChild(span);

    var mag = document.createElement("input");
    mag.type = "range";
    mag.className = "slider";
    mag.value = "1";
    mag.min = "0";
    mag.max = "1";
    mag.step = "0.01";
    mag.id = "mag" + i;
    parent.appendChild(mag);

    var span_mag = document.createElement("span");
    span_mag.className = "slider-span";
    span_mag.id = "mag" + i + "_label";
    span_mag.textContent = 1;
    parent.appendChild(span_mag);

    // Make slider change label
    document.getElementById("mag" + i).addEventListener("input", function () {
      document.getElementById(this.id + "_label").textContent = Math.round(this.value * 100) / 100;
      data[0]["y"] = calculate_beam_pattern();
      Plotly.redraw("rectPlot");
    });

    var phase = document.createElement("input");
    phase.type = "range";
    phase.className = "slider";
    phase.value = "0";
    phase.min = "-3.14159";
    phase.max = "3.14159";
    phase.step = "0.01";
    phase.id = "phase" + i;
    parent.appendChild(phase);

    var span = document.createElement("span");
    span.className = "slider-span";
    span.id = "phase" + i + "_label";
    span.textContent = 0;
    parent.appendChild(span);

    // Make slider change label
    document.getElementById("phase" + i).addEventListener("input", function () {
      document.getElementById(this.id + "_label").textContent = Math.round(this.value * 100) / 100;
      data[0]["y"] = calculate_beam_pattern();
      Plotly.redraw("rectPlot");
    });

    var br = document.createElement("br");
    parent.appendChild(br);
  }

  // Add reset button
  var reset = document.createElement("button");
  reset.textContent = "Reset";
  reset.style.marginTop = "10px";
  reset.onclick = function () {
    for (var i = 0; i < 8; i++) {
      document.getElementById("mag" + i).value = 1;
      document.getElementById("mag" + i + "_label").textContent = 1;
      document.getElementById("phase" + i).value = 0;
      document.getElementById("phase" + i + "_label").textContent = 0;
      data[0]["y"] = calculate_beam_pattern();
      Plotly.redraw("rectPlot");
    }
  };
  parent.appendChild(reset);

  // Run processing once
  data[0]["y"] = calculate_beam_pattern();
  Plotly.redraw("rectPlot");
}
