function tdoa_app(containerId) {
  // ----- configuration -------------------------------------------------------
  const c = 3e8; // propagation speed [m/s] (free space / RF)
  const W = 600; // canvas width  [px]
  const H = 480; // canvas height [px]
  const worldSpan = 1000; // world width represented across the canvas [m]
  const nodeRadius = 9; // hit/draw radius for draggable handles [px]
  const edgeMargin = 12; // keep handles at least this far inside the canvas [px]
  const maxSensors = 10; // upper bound on how many sensors the user can add
  const palette = ["#e6550d", "#3182bd", "#31a354"]; // first few sensor-pair colors

  // heatmap: give each hyperbola some width and add them together so their
  // overlap lights up where the emitter actually is
  let heatHalfWidth = 150; // band half-width around each hyperbola [m]
  const heatRes = 2; // pixel block size of the heatmap grid (quality vs speed)
  const heatMaxAlpha = 0.55; // overlay opacity where every band coincides

  // Color for the k-th sensor pair: use the fixed palette first, then spread the
  // remaining hues around the color wheel so every pair stays distinguishable.
  function pairColor(k) {
    if (k < palette.length) return palette[k];
    return `hsl(${(k * 47) % 360}, 65%, 45%)`;
  }

  // ----- DOM setup -----------------------------------------------------------
  const container = document.getElementById(containerId || "tdoaApp") || document.body;

  // canvas sits in a flex row next to the noise slider on its right
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "stretch";
  row.style.gap = "10px";
  container.appendChild(row);

  // wrapper lets us overlay controls (the Add-sensor button) on top of the canvas
  const canvasWrap = document.createElement("div");
  canvasWrap.style.position = "relative";
  canvasWrap.style.lineHeight = "0"; // avoid extra space under the canvas
  row.appendChild(canvasWrap);

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.style.border = "1px solid #888";
  canvas.style.touchAction = "none"; // let us handle touch-drag ourselves
  canvas.style.cursor = "grab";
  canvasWrap.appendChild(canvas);

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

  // vertical dynamic-range slider: a gamma exponent applied to the heatmap
  // intensity.  Higher values darken the weak single bands and let only the
  // strong overlaps near the emitter stand out; lower values flatten it out.
  let heatGamma = 1.5; // exponent applied to the normalized heatmap intensity
  const drBox = document.createElement("div");
  drBox.style.display = "flex";
  drBox.style.flexDirection = "column";
  drBox.style.alignItems = "center";
  drBox.style.fontFamily = "sans-serif";
  drBox.style.fontSize = "12px";
  row.appendChild(drBox);

  const drLabel = document.createElement("div");
  drLabel.style.textAlign = "center";
  drLabel.style.marginBottom = "6px";
  drBox.appendChild(drLabel);

  function updateDrLabel() {
    drLabel.innerHTML = `Dyn.<br>range<br>${heatGamma.toFixed(1)}`;
  }
  updateDrLabel();

  const drSlider = document.createElement("input");
  drSlider.type = "range";
  drSlider.min = "0.5";
  drSlider.max = "5";
  drSlider.step = "0.1";
  drSlider.value = String(heatGamma);
  drSlider.setAttribute("orient", "vertical");
  drSlider.style.writingMode = "vertical-lr";
  drSlider.style.direction = "rtl"; // low at the bottom, high at the top
  drSlider.style.height = H / 2 + "px";
  drSlider.style.width = "24px";
  drBox.appendChild(drSlider);

  drSlider.addEventListener("input", () => {
    heatGamma = parseFloat(drSlider.value);
    updateDrLabel();
    render(false); // visual-only: keep the existing noise samples
  });

  // vertical width slider: half-width of the band drawn around each hyperbola.
  // Wider bands overlap more readily (good with lots of noise); narrow bands
  // pin the emitter down tightly.
  const widthBox = document.createElement("div");
  widthBox.style.display = "flex";
  widthBox.style.flexDirection = "column";
  widthBox.style.alignItems = "center";
  widthBox.style.fontFamily = "sans-serif";
  widthBox.style.fontSize = "12px";
  row.appendChild(widthBox);

  const widthLabel = document.createElement("div");
  widthLabel.style.textAlign = "center";
  widthLabel.style.marginBottom = "6px";
  widthBox.appendChild(widthLabel);

  function updateWidthLabel() {
    widthLabel.innerHTML = `Width<br>${heatHalfWidth.toFixed(0)} m`;
  }
  updateWidthLabel();

  const widthSlider = document.createElement("input");
  widthSlider.type = "range";
  widthSlider.min = "10";
  widthSlider.max = "200";
  widthSlider.step = "5";
  widthSlider.value = String(heatHalfWidth);
  widthSlider.setAttribute("orient", "vertical");
  widthSlider.style.writingMode = "vertical-lr";
  widthSlider.style.direction = "rtl"; // narrow at the bottom, wide at the top
  widthSlider.style.height = H / 2 + "px";
  widthSlider.style.width = "24px";
  widthBox.appendChild(widthSlider);

  widthSlider.addEventListener("input", () => {
    heatHalfWidth = parseFloat(widthSlider.value);
    updateWidthLabel();
    render(false); // visual-only: keep the existing noise samples
  });

  // buttons to add/remove a sensor, overlaid on the top-left corner of the canvas
  const addBtn = document.createElement("button");
  addBtn.style.position = "absolute";
  addBtn.style.top = "8px";
  addBtn.style.left = "8px";
  addBtn.style.fontFamily = "sans-serif";
  addBtn.style.fontSize = "13px";
  addBtn.style.lineHeight = "normal";
  addBtn.style.padding = "4px 10px";
  addBtn.style.cursor = "pointer";
  canvasWrap.appendChild(addBtn);

  const removeBtn = document.createElement("button");
  removeBtn.style.position = "absolute";
  removeBtn.style.top = "40px";
  removeBtn.style.left = "8px";
  removeBtn.style.fontFamily = "sans-serif";
  removeBtn.style.fontSize = "13px";
  removeBtn.style.lineHeight = "normal";
  removeBtn.style.padding = "4px 10px";
  removeBtn.style.cursor = "pointer";
  canvasWrap.appendChild(removeBtn);

  // checkbox to toggle the heatmap overlay, overlaid below the buttons
  let showHeatmap = true; // heatmap is on by default
  const heatToggle = document.createElement("label");
  heatToggle.style.position = "absolute";
  heatToggle.style.top = "72px";
  heatToggle.style.left = "8px";
  heatToggle.style.display = "flex";
  heatToggle.style.alignItems = "center";
  heatToggle.style.gap = "4px";
  heatToggle.style.fontFamily = "sans-serif";
  heatToggle.style.fontSize = "13px";
  heatToggle.style.color = "#222";
  heatToggle.style.cursor = "pointer";
  heatToggle.style.userSelect = "none";

  const heatCheckbox = document.createElement("input");
  heatCheckbox.type = "checkbox";
  heatCheckbox.checked = showHeatmap;
  heatCheckbox.style.cursor = "pointer";
  heatCheckbox.style.margin = "0";
  heatToggle.appendChild(heatCheckbox);
  heatToggle.appendChild(document.createTextNode("Heatmap"));
  canvasWrap.appendChild(heatToggle);

  heatCheckbox.addEventListener("change", () => {
    showHeatmap = heatCheckbox.checked;
    render(false); // visual-only: keep the existing noise samples
  });

  // collapsible details panel: a thick bar you click to reveal the text readout,
  // collapsed by default to keep the figure compact
  const details = document.createElement("details");
  details.style.marginTop = "8px";
  details.style.width = W + "px";
  details.style.border = "1px solid #ccc";
  details.style.borderRadius = "4px";
  details.style.overflow = "hidden";
  container.appendChild(details);

  const summary = document.createElement("summary");
  summary.textContent = "Show Debug Info";
  summary.style.cursor = "pointer";
  summary.style.userSelect = "none";
  summary.style.fontFamily = "sans-serif";
  summary.style.fontSize = "13px";
  summary.style.fontWeight = "bold";
  summary.style.padding = "10px 12px";
  summary.style.background = "#f0f0f0";
  summary.style.color = "#333";
  details.appendChild(summary);

  // swap the label between collapsed/expanded states
  details.addEventListener("toggle", () => {
    summary.textContent = details.open ? "Hide Debug Info" : "Show Debug Info";
  });

  const readout = document.createElement("div");
  readout.style.fontFamily = "monospace";
  readout.style.fontSize = "13px";
  readout.style.padding = "8px 12px";
  details.appendChild(readout);

  const ctx = canvas.getContext("2d");

  // ----- scene state (world coordinates, meters, origin at center) -----------
  // y points up in world coordinates (flipped when drawing to the canvas).
  const emitter = { x: -155, y: 226, label: "Emitter" };
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

  // Every unique sensor pair (i < j); order matches the original [0,1],[0,2],[1,2].
  function sensorPairs() {
    const pairs = [];
    for (let i = 0; i < sensors.length; i++) {
      for (let j = i + 1; j < sensors.length; j++) pairs.push([i, j]);
    }
    return pairs;
  }

  // Returns the per-sensor TOA and, for each pair, the TDOA and range diff.
  function simulate() {
    const toa = sensors.map((s) => dist(emitter, s) / c); // seconds
    const measurements = sensorPairs().map(([i, j]) => {
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

  // ----- heatmap -------------------------------------------------------------
  // Reuse the same hyperbolas, but instead of an infinitely thin curve give
  // each one a band of width 2*heatHalfWidth whose intensity follows a raised
  // sine (raised cosine): 1 right on the hyperbola, tapering to 0 at the band
  // edges.  Summing every band makes their common crossing — the emitter — the
  // brightest spot.  We compute on a coarse grid and let drawImage smooth it.
  const heatCanvas = document.createElement("canvas");
  const heatCtx = heatCanvas.getContext("2d");

  // warm colormap: yellow at low intensity ramping to red at high intensity
  function heatColor(t) {
    return [255, Math.round(220 * (1 - t)), Math.round(40 * (1 - t))];
  }

  function drawHeatmap(measurements) {
    if (measurements.length === 0) return;
    const gw = Math.ceil(W / heatRes);
    const gh = Math.ceil(H / heatRes);
    heatCanvas.width = gw;
    heatCanvas.height = gh;
    const img = heatCtx.createImageData(gw, gh);
    const data = img.data;
    const nPairs = measurements.length;

    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        // world coordinates at the center of this grid block
        const w = pxToWorld({
          x: gx * heatRes + heatRes / 2,
          y: gy * heatRes + heatRes / 2
        });

        let sum = 0;
        for (let k = 0; k < nPairs; k++) {
          const m = measurements[k];
          const si = sensors[m.i];
          const sj = sensors[m.j];
          const ri = Math.hypot(w.x - si.x, w.y - si.y);
          const rj = Math.hypot(w.x - sj.x, w.y - sj.y);
          // how far this point's range difference is from the measured one;
          // 0 means the point sits exactly on pair k's hyperbola
          const residual = ri - rj - m.dr;
          if (Math.abs(residual) < heatHalfWidth) {
            sum += 0.5 * (1 + Math.cos((Math.PI * residual) / heatHalfWidth));
          }
        }

        // normalized intensity (1 where all bands overlap, the emitter), then
        // a gamma to set the dynamic range: >1 suppresses the weak bands
        const norm = Math.pow(sum / nPairs, heatGamma);
        const idx = (gy * gw + gx) * 4;
        const rgb = heatColor(norm);
        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = Math.round(norm * heatMaxAlpha * 255);
      }
    }
    heatCtx.putImageData(img, 0, 0);

    // scale the coarse buffer up to full size; bilinear smoothing hides the grid
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heatCanvas, 0, 0, gw, gh, 0, 0, W, H);
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

  // small label near a node showing its world coordinates in meters
  function drawCoordTooltip(node) {
    const p = worldToPx(node);
    const text = `(${node.x.toFixed(0)}, ${node.y.toFixed(0)}) m`;
    ctx.font = "12px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    const padX = 5;
    const padY = 3;
    const tw = ctx.measureText(text).width;
    const boxW = tw + padX * 2;
    const boxH = 16 + padY * 2;
    // sit the box just below the node, nudged on-screen if it would clip
    let bx = p.x + nodeRadius + 2;
    let by = p.y + nodeRadius + 2;
    if (bx + boxW > W) bx = W - boxW;
    if (by + boxH > H) by = p.y - nodeRadius - boxH - 2;
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, bx + padX, by + boxH - padY - 3);
  }

  function drawEmitter() {
    const p = worldToPx(emitter);
    // black outline, no fill, so the heatmap underneath stays visible
    ctx.beginPath();
    ctx.arc(p.x, p.y, nodeRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(emitter.label, p.x + nodeRadius + 2, p.y - 2);
  }

  // cache the last simulation so purely-visual changes (heatmap width, dynamic
  // range, hover) can repaint without drawing fresh noise samples
  let lastSim = null;
  function render(resimulate = true) {
    if (resimulate || !lastSim) lastSim = simulate();
    const { toa, measurements } = lastSim;

    drawGrid();
    if (showHeatmap) drawHeatmap(measurements);
    measurements.forEach((m, k) => {
      drawHyperbola(sensors[m.i], sensors[m.j], m.dr, pairColor(k));
    });
    sensors.forEach(drawSensor);
    drawEmitter();

    // coordinate tooltip for the node under the cursor (or being dragged)
    const activeNode = dragTarget || hoverNode;
    if (activeNode) drawCoordTooltip(activeNode);

    // text readout of the simulated quantities
    let html = "";
    toa.forEach((t, i) => {
      html += `TOA(${sensors[i].label}) = ${(t * 1e9).toFixed(1)} ns &nbsp; (range ${dist(emitter, sensors[i]).toFixed(0)} m)<br>`;
    });
    measurements.forEach((m, k) => {
      html += `<span style="color:${pairColor(k)}">TDOA(${sensors[m.i].label},${sensors[m.j].label}) = ${(m.tdoa * 1e9).toFixed(1)} ns &nbsp; Δr = ${m.dr.toFixed(0)} m</span><br>`;
    });
    readout.innerHTML = html;
  }

  // ----- dragging ------------------------------------------------------------
  let dragTarget = null;
  let hoverNode = null; // node currently under the cursor (for the tooltip)

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
      const hit = pickNode(px);
      canvas.style.cursor = hit ? "grab" : "default";
      // only repaint when the hovered node actually changes, so the tooltip
      // appears/disappears without re-noising the scene on every mouse move
      if (hit !== hoverNode) {
        hoverNode = hit;
        render(false); // visual-only: keep the existing noise samples
      }
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

  canvas.addEventListener("mouseleave", () => {
    if (hoverNode) {
      hoverNode = null;
      render(false); // visual-only: keep the existing noise samples
    }
  });

  canvas.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  canvas.addEventListener("touchstart", onDown, { passive: false });
  canvas.addEventListener("touchmove", onMove, { passive: false });
  canvas.addEventListener("touchend", onUp);

  // ----- adding / removing sensors -------------------------------------------
  const minSensors = 2; // at least 2 sensors to form one TDOA pair

  function updateSensorButtons() {
    const atMax = sensors.length >= maxSensors;
    addBtn.disabled = atMax;
    addBtn.textContent = atMax
      ? `Max sensors reached (${maxSensors})`
      : `Add sensor`;

    const atMin = sensors.length <= minSensors;
    removeBtn.disabled = atMin;
    removeBtn.textContent = atMin
      ? `Min sensors reached (${minSensors})`
      : "Remove sensor";
  }

  function addSensor() {
    if (sensors.length >= maxSensors) return;
    const idx = sensors.length;
    // place the new sensor on a ring, stepping by the golden angle so successive
    // sensors spread out rather than landing on top of each other
    const ang = idx * 2.399963229728653;
    const r = 320;
    sensors.push({
      x: r * Math.cos(ang),
      y: r * Math.sin(ang),
      label: "Sensor " + idx
    });
    updateSensorButtons();
    render();
  }

  function removeSensor() {
    if (sensors.length <= minSensors) return;
    sensors.pop(); // drop the most recently added sensor
    updateSensorButtons();
    render();
  }

  addBtn.addEventListener("click", addSensor);
  removeBtn.addEventListener("click", removeSensor);
  updateSensorButtons();

  // ----- go ------------------------------------------------------------------
  render();
}
