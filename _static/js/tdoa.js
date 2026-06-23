function tdoa_app(containerId) {
  // ----- configuration -------------------------------------------------------
  const c = 3e8; // propagation speed [m/s] (free space / RF)
  const W = 600; // canvas width  [px]
  const H = 480; // canvas height [px]
  const worldSpan = 1000; // world width represented across the canvas [m]
  const nodeRadius = 9; // hit/draw radius for draggable handles [px]
  const edgeMargin = 12; // keep handles at least this far inside the canvas [px]
  const pairColors = ["#e6550d", "#3182bd", "#31a354"]; // one per sensor pair

  // ----- DOM setup -----------------------------------------------------------
  const container = document.getElementById(containerId || "tdoaApp") || document.body;

  // canvas sits in a flex row next to the noise slider on its right
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "stretch";
  row.style.gap = "10px";
  container.appendChild(row);

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.style.border = "1px solid #888";
  canvas.style.touchAction = "none"; // let us handle touch-drag ourselves
  canvas.style.cursor = "grab";
  row.appendChild(canvas);

  // vertical noise slider: standard deviation of the Gaussian noise [m]
  const sliderBox = document.createElement("div");
  sliderBox.style.display = "flex";
  sliderBox.style.flexDirection = "column";
  sliderBox.style.alignItems = "center";
  sliderBox.style.fontFamily = "sans-serif";
  sliderBox.style.fontSize = "12px";
  row.appendChild(sliderBox);

  const sliderLabel = document.createElement("div");
  sliderLabel.style.textAlign = "center";
  sliderLabel.style.marginBottom = "6px";
  sliderBox.appendChild(sliderLabel);

  let noiseStd = 0; // std dev of Gaussian noise added to each range diff [m]
  function updateSliderLabel() {
    sliderLabel.innerHTML = `Noise<br>${noiseStd.toFixed(0)} m`;
  }
  updateSliderLabel();

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "100";
  slider.step = "1";
  slider.value = "0";
  // make the range input vertical (with a fallback for older browsers)
  slider.setAttribute("orient", "vertical");
  slider.style.writingMode = "vertical-lr";
  slider.style.direction = "rtl"; // 0 at the bottom, max at the top
  slider.style.height = H / 2 + "px"; // half the canvas height
  slider.style.width = "24px";
  sliderBox.appendChild(slider);

  slider.addEventListener("input", () => {
    noiseStd = parseFloat(slider.value);
    updateSliderLabel();
    render();
  });

  const readout = document.createElement("div");
  readout.style.fontFamily = "monospace";
  readout.style.fontSize = "13px";
  readout.style.marginTop = "6px";
  container.appendChild(readout);

  const ctx = canvas.getContext("2d");

  // ----- scene state (world coordinates, meters, origin at center) -----------
  // y points up in world coordinates (flipped when drawing to the canvas).
  const emitter = { x: 0, y: 120, label: "Emitter" };
  const sensors = [
    { x: -350, y: -200, label: "Sensor 0" },
    { x: 350, y: -200, label: "Sensor 1" },
    { x: 0, y: 300, label: "Sensor 2" }
  ];

  // ----- coordinate transforms ----------------------------------------------
  const scale = W / worldSpan; // px per meter
  function worldToPx(p) {
    return { x: W / 2 + p.x * scale, y: H / 2 - p.y * scale };
  }
  function pxToWorld(px) {
    return { x: (px.x - W / 2) / scale, y: (H / 2 - px.y) / scale };
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  // ----- TDOA simulation -----------------------------------------------------
  // Standard normal sample via the Box-Muller transform.
  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Returns the per-sensor TOA and, for each pair, the TDOA and range diff.
  function simulate() {
    const toa = sensors.map((s) => dist(emitter, s) / c); // seconds
    const pairs = [
      [0, 1],
      [0, 2],
      [1, 2]
    ];
    const measurements = pairs.map(([i, j]) => {
      // ideal TDOA, then add Gaussian noise (slider sets its std dev in meters,
      // so we convert that range error into the equivalent time error)
      const tdoa = toa[i] - toa[j] + (noiseStd * randn()) / c; // seconds
      return { i, j, tdoa, dr: c * tdoa }; // dr = r_i - r_j  [m]
    });
    return { toa, measurements };
  }

  // ----- hyperbola drawing ---------------------------------------------------
  // Locus of points u with |u - s_i| - |u - s_j| = dr is a hyperbola with foci
  // at the two sensors.  We parametrize it in a frame centered on the midpoint
  // of the foci, with the transverse axis along the baseline.
  function drawHyperbola(si, sj, dr, color) {
    const mid = { x: (si.x + sj.x) / 2, y: (si.y + sj.y) / 2 };
    const baseline = dist(si, sj);
    const cFoci = baseline / 2; // half the focal separation
    const a = dr / 2; // signed semi-transverse axis; sign picks the branch
    if (Math.abs(a) >= cFoci) return; // |dr| can't exceed the baseline
    const b = Math.sqrt(cFoci * cFoci - a * a);

    // Unit vector u from s_i toward s_j (axis), and perpendicular v.
    const ux = (sj.x - si.x) / baseline;
    const uy = (sj.y - si.y) / baseline;
    const vx = -uy;
    const vy = ux;

    // Sweep the parameter t; x = a*cosh(t) keeps us on the correct branch
    // because a carries the sign of dr.
    ctx.beginPath();
    let first = true;
    for (let t = -3; t <= 3.0001; t += 0.05) {
      const xl = a * Math.cosh(t);
      const yl = b * Math.sinh(t);
      const wx = mid.x + xl * ux + yl * vx;
      const wy = mid.y + xl * uy + yl * vy;
      const p = worldToPx({ x: wx, y: wy });
      if (first) {
        ctx.moveTo(p.x, p.y);
        first = false;
      } else {
        ctx.lineTo(p.x, p.y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ----- rendering -----------------------------------------------------------
  function drawGrid() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#eee";
    ctx.lineWidth = 1;
    const step = 100 * scale; // grid every 100 m
    for (let x = (W / 2) % step; x < W; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = (H / 2) % step; y < H; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    // axes
    ctx.strokeStyle = "#ccc";
    ctx.beginPath();
    ctx.moveTo(W / 2, 0);
    ctx.lineTo(W / 2, H);
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    // axis labels
    ctx.fillStyle = "#999";
    ctx.font = "13px sans-serif";
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "right";
    ctx.fillText("X", W - 6, H / 2 - 6);
    ctx.textAlign = "left";
    ctx.fillText("Y", W / 2 + 6, 14);
    ctx.textAlign = "left";
  }

  function drawSensor(s) {
    const p = worldToPx(s);
    ctx.fillStyle = "#222";
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - nodeRadius);
    ctx.lineTo(p.x + nodeRadius, p.y + nodeRadius);
    ctx.lineTo(p.x - nodeRadius, p.y + nodeRadius);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#222";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(s.label, p.x + nodeRadius + 2, p.y - 2);
  }

  function drawEmitter() {
    const p = worldToPx(emitter);
    ctx.fillStyle = "#d62728";
    ctx.beginPath();
    ctx.arc(p.x, p.y, nodeRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#d62728";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(emitter.label, p.x + nodeRadius + 2, p.y - 2);
  }

  function render() {
    const { toa, measurements } = simulate();

    drawGrid();
    measurements.forEach((m, k) => {
      drawHyperbola(sensors[m.i], sensors[m.j], m.dr, pairColors[k]);
    });
    sensors.forEach(drawSensor);
    drawEmitter();

    // text readout of the simulated quantities
    let html = "";
    toa.forEach((t, i) => {
      html += `TOA(${sensors[i].label}) = ${(t * 1e9).toFixed(1)} ns &nbsp; (range ${dist(emitter, sensors[i]).toFixed(0)} m)<br>`;
    });
    measurements.forEach((m, k) => {
      html += `<span style="color:${pairColors[k]}">TDOA(${sensors[m.i].label},${sensors[m.j].label}) = ${(m.tdoa * 1e9).toFixed(1)} ns &nbsp; Δr = ${m.dr.toFixed(0)} m</span><br>`;
    });
    readout.innerHTML = html;
  }

  // ----- dragging ------------------------------------------------------------
  let dragTarget = null;

  function eventPx(e) {
    const rect = canvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return { x: src.clientX - rect.left, y: src.clientY - rect.top };
  }

  function pickNode(px) {
    const all = [emitter, ...sensors];
    for (const node of all) {
      if (dist(worldToPx(node), px) <= nodeRadius + 4) return node;
    }
    return null;
  }

  function onDown(e) {
    const px = eventPx(e);
    dragTarget = pickNode(px);
    if (dragTarget) {
      canvas.style.cursor = "grabbing";
      e.preventDefault();
    }
  }

  function onMove(e) {
    const px = eventPx(e);
    if (!dragTarget) {
      canvas.style.cursor = pickNode(px) ? "grab" : "default";
      return;
    }
    // keep the handle inside the canvas (with a small margin) so it can't be
    // dragged off-screen and lost
    px.x = Math.max(edgeMargin, Math.min(W - edgeMargin, px.x));
    px.y = Math.max(edgeMargin, Math.min(H - edgeMargin, px.y));
    const w = pxToWorld(px);
    dragTarget.x = w.x;
    dragTarget.y = w.y;
    render();
    e.preventDefault();
  }

  function onUp() {
    dragTarget = null;
    canvas.style.cursor = "grab";
  }

  canvas.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  canvas.addEventListener("touchstart", onDown, { passive: false });
  canvas.addEventListener("touchmove", onMove, { passive: false });
  canvas.addEventListener("touchend", onUp);

  // ----- go ------------------------------------------------------------------
  render();
}
