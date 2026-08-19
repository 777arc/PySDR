// Interactive I*cos() + Q*sin() demo for the Sampling (IQ) chapter.
// Browser port of figure-generating-scripts/sin_plus_cos.py, which used PyQt5 + pyqtgraph.
// Two sliders set the amplitudes I and Q; the plot shows I*cos() (red), Q*sin() (blue),
// and their sum (green, dotted), styled to match the original pyqtgraph app.
//
// Usage in a page:  <div id="sinPlusCosApp"></div> <script>sin_plus_cos_app("sinPlusCosApp")</script>

function sin_plus_cos_app(containerId) {
  const container = document.getElementById(containerId || "sinPlusCosApp") || document.body;

  // ---- inject scoped styles once (all rules are prefixed by .sin-plus-cos-app) ----
  if (!document.getElementById("sin-plus-cos-app-styles")) {
    const style = document.createElement("style");
    style.id = "sin-plus-cos-app-styles";
    style.textContent = `
.sin-plus-cos-app{max-width:800px;margin:10px auto 20px;background:#fff;
  padding:10px 12px 12px;color:#000;font-family:sans-serif;}
.sin-plus-cos-app *{box-sizing:border-box;}
.sin-plus-cos-app .slider-block{margin:6px 0 10px;}
.sin-plus-cos-app .slider-label{text-align:center;font-size:13px;color:#000;margin-bottom:4px;
  font-family:sans-serif;}
.sin-plus-cos-app .canvas-holder{line-height:0;}
.sin-plus-cos-app canvas{display:block;width:100%;height:auto;}
/* Qt-ish slider: thin sunken groove with a small rectangular handle */
.sin-plus-cos-app input[type=range]{-webkit-appearance:none;appearance:none;width:100%;
  background:transparent;margin:0;cursor:pointer;}
.sin-plus-cos-app input[type=range]:focus{outline:none;}
.sin-plus-cos-app input[type=range]::-webkit-slider-runnable-track{height:5px;background:#e8e8e8;
  border:1px solid #a0a0a0;border-radius:2px;}
.sin-plus-cos-app input[type=range]::-moz-range-track{height:5px;background:#e8e8e8;
  border:1px solid #a0a0a0;border-radius:2px;}
.sin-plus-cos-app input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
  width:12px;height:19px;margin-top:-8px;border:1px solid #6f6f6f;border-radius:2px;
  background:linear-gradient(#fdfdfd,#dcdcdc);}
.sin-plus-cos-app input[type=range]::-moz-range-thumb{width:12px;height:19px;border:1px solid #6f6f6f;
  border-radius:2px;background:linear-gradient(#fdfdfd,#dcdcdc);}
.sin-plus-cos-app input[type=range]:hover::-webkit-slider-thumb{background:linear-gradient(#fff,#e8e8e8);}
.sin-plus-cos-app input[type=range]:hover::-moz-range-thumb{background:linear-gradient(#fff,#e8e8e8);}
.sin-plus-cos-app .actions{text-align:center;margin-top:8px;}
.sin-plus-cos-app .actions button{font-family:sans-serif;font-size:13px;padding:5px 16px;
  border:1px solid #a0a0a0;border-radius:3px;background:linear-gradient(#fdfdfd,#e6e6e6);
  color:#000;cursor:pointer;}
.sin-plus-cos-app .actions button:hover{background:linear-gradient(#fff,#efefef);}`;
    document.head.appendChild(style);
  }

  // ---- build DOM inside the container ----
  const root = document.createElement("div");
  root.className = "sin-plus-cos-app";
  root.innerHTML = `
    <div class="slider-block">
      <div class="slider-label" id="spc-I-label">I = 1</div>
      <input type="range" id="spc-I" min="-2" max="2" step="0.01" value="1" aria-label="I, amplitude of the cosine">
    </div>
    <div class="slider-block">
      <div class="slider-label" id="spc-Q-label">Q = 0.5</div>
      <input type="range" id="spc-Q" min="-2" max="2" step="0.01" value="0.5" aria-label="Q, amplitude of the sine">
    </div>
    <div class="canvas-holder"><canvas id="spc-plot" width="780" height="420"></canvas></div>
    <div class="actions"><button id="spc-reset">Reset</button></div>`;
  container.appendChild(root);

  const $ = (sel) => root.querySelector(sel);
  const sliderI = $("#spc-I");
  const sliderQ = $("#spc-Q");
  const canvas = $("#spc-plot");
  const ctx = canvas.getContext("2d");

  // ---- plot config, mirroring the pyqtgraph version ----
  const N = 150;                       // np.linspace(0, 10, 150)
  const xData = Array.from({ length: N }, (_, k) => (k * 10) / (N - 1));
  const X_RANGE = [0, 150];            // plotted against sample index
  const Y_RANGE = [-3, 3];
  const CURVES = [
    { color: "#ff0000", dash: [], name: "I*cos()" },
    { color: "#0000ff", dash: [], name: "Q*sin()" },
    { color: "#00991c", dash: [5, 10], name: "I*cos() + Q*sin()" }  // Qt.DotLine at width 5
  ];
  const LINE_WIDTH = 5;
  const PAD = { left: 62, right: 12, top: 10, bottom: 46 };

  // Python's "{0:.2g}" formatting, e.g. 1 -> "1", 0.5 -> "0.5", -1.85 -> "-1.9"
  function fmt2g(v) {
    if (v === 0) return "0";
    return String(Number(v.toPrecision(2)));
  }

  // pyqtgraph-style tick spacing: 1/2/5 times a power of ten, aiming for a handful of ticks
  function tickStep(span, target) {
    const raw = span / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const mult = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
    return mult * mag;
  }

  function ticks(lo, hi, target) {
    const step = tickStep(hi - lo, target);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return { vals: out, step: step };
  }

  function tickText(v, step) {
    const decimals = Math.max(0, -Math.floor(Math.log10(step)));
    return v.toFixed(decimals);
  }

  function draw() {
    // Keep the backing store matched to the CSS size so lines stay crisp on any display
    const cssW = canvas.clientWidth || 780;
    const cssH = Math.round(cssW * 420 / 780);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.height = cssH + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, cssW, cssH);

    const plotL = PAD.left, plotR = cssW - PAD.right;
    const plotT = PAD.top, plotB = cssH - PAD.bottom;
    const sx = (x) => plotL + (x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (plotR - plotL);
    const sy = (y) => plotB - (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (plotB - plotT);

    const I = parseFloat(sliderI.value);
    const Q = parseFloat(sliderQ.value);
    const series = [
      xData.map((x) => I * Math.cos(x)),
      xData.map((x) => Q * Math.sin(x)),
      xData.map((x) => I * Math.cos(x) + Q * Math.sin(x))
    ];

    // ---- curves (clipped to the plot rect, like a pyqtgraph ViewBox) ----
    ctx.save();
    ctx.beginPath();
    ctx.rect(plotL, plotT, plotR - plotL, plotB - plotT);
    ctx.clip();
    ctx.lineJoin = "round";
    ctx.lineCap = "butt";
    CURVES.forEach((curve, ci) => {
      ctx.strokeStyle = curve.color;
      ctx.lineWidth = LINE_WIDTH;
      ctx.setLineDash(curve.dash);
      ctx.beginPath();
      series[ci].forEach((y, i) => (i === 0 ? ctx.moveTo(sx(i), sy(y)) : ctx.lineTo(sx(i), sy(y))));
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.restore();

    // ---- axes: black left and bottom lines with outward ticks ----
    const xt = ticks(X_RANGE[0], X_RANGE[1], 6);
    const yt = ticks(Y_RANGE[0], Y_RANGE[1], 6);
    ctx.strokeStyle = "#000";
    ctx.fillStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotL + 0.5, plotT);
    ctx.lineTo(plotL + 0.5, plotB + 0.5);
    ctx.lineTo(plotR, plotB + 0.5);
    ctx.stroke();

    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xt.vals.forEach((v) => {
      const x = Math.round(sx(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, plotB + 0.5);
      ctx.lineTo(x, plotB + 5.5);
      ctx.stroke();
      ctx.fillText(tickText(v, xt.step), x, plotB + 8);
    });

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yt.vals.forEach((v) => {
      const y = Math.round(sy(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(plotL + 0.5, y);
      ctx.lineTo(plotL - 4.5, y);
      ctx.stroke();
      ctx.fillText(tickText(v, yt.step), plotL - 8, y);
    });

    // ---- axis labels ----
    ctx.font = "18px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Time", (plotL + plotR) / 2, plotB + PAD.bottom - 2);
    ctx.save();
    ctx.translate(14, (plotT + plotB) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText("Amplitude", 0, 0);
    ctx.restore();

    // ---- legend, top-left inside the plot like addLegend(offset=5) ----
    ctx.font = "14px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const rowH = 22, sample = 26, gap = 8;
    const textW = Math.max(...CURVES.map((c) => ctx.measureText(c.name).width));
    const boxW = 10 + sample + gap + textW + 10;
    const boxH = 8 + CURVES.length * rowH;
    const boxX = plotL + 5, boxY = plotT + 5;
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(boxX + 0.5, boxY + 0.5, boxW, boxH);
    ctx.fill();
    ctx.stroke();
    CURVES.forEach((curve, i) => {
      const y = boxY + 4 + rowH * (i + 0.5);
      ctx.strokeStyle = curve.color;
      ctx.lineWidth = LINE_WIDTH;
      ctx.setLineDash(curve.dash);
      ctx.beginPath();
      ctx.moveTo(boxX + 10, y);
      ctx.lineTo(boxX + 10 + sample, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#000";
      ctx.fillText(curve.name, boxX + 10 + sample + gap, y);
    });
  }

  // ---- wiring ----
  function updateLabels() {
    $("#spc-I-label").textContent = "I = " + fmt2g(parseFloat(sliderI.value));
    $("#spc-Q-label").textContent = "Q = " + fmt2g(parseFloat(sliderQ.value));
  }
  function update() { updateLabels(); draw(); }

  sliderI.addEventListener("input", update);
  sliderQ.addEventListener("input", update);
  $("#spc-reset").addEventListener("click", () => {
    sliderI.value = 1;
    sliderQ.value = 0.5;
    update();
  });
  window.addEventListener("resize", draw);

  update();
}
