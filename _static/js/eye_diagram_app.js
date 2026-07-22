// Interactive eye-diagram explorer for the Pulse Shaping chapter.
// Generates random real symbols (BPSK or 4-ASK), applies raised-cosine (or root-raised-cosine)
// pulse shaping, adds AWGN and timing jitter, then overlays short windows into a
// phosphor-style eye diagram with live eye-height/eye-width calipers.
//
// Usage in a page:  <div id="eyeApp"></div> <script>eye_diagram_app("eyeApp")</script>

function eye_diagram_app(containerId) {
  const container = document.getElementById(containerId || "eyeApp") || document.body;

  // ---- inject scoped styles once (all rules are prefixed by .eye-diagram-app) ----
  if (!document.getElementById("eye-diagram-app-styles")) {
    const style = document.createElement("style");
    style.id = "eye-diagram-app-styles";
    style.textContent = `
.eye-diagram-app{--accent:#e6550d;max-width:1000px;margin:8px auto 4px;color:#222;font-family:sans-serif;}
.eye-diagram-app *{box-sizing:border-box;}
.eye-diagram-app .lab{display:grid;grid-template-columns:1fr 300px;gap:16px;align-items:start;}
@media (max-width:820px){.eye-diagram-app .lab{grid-template-columns:1fr;}}
.eye-diagram-app .screen{background:#fff;border:1px solid #ccc;border-radius:4px;padding:12px;}
.eye-diagram-app .screen-label{display:flex;justify-content:space-between;align-items:center;
  font-size:12px;color:#555;margin:2px 2px 8px;}
.eye-diagram-app .screen-label .dot{width:8px;height:8px;border-radius:50%;background:#17c3b2;
  display:inline-block;margin-right:6px;vertical-align:middle;}
.eye-diagram-app .canvas-holder{position:relative;border:1px solid #888;border-radius:4px;overflow:hidden;background:#070c14;line-height:0;}
.eye-diagram-app canvas{display:block;width:100%;height:auto;}
.eye-diagram-app .meters{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px;}
.eye-diagram-app .meter{background:#f7f7f7;border:1px solid #ccc;border-radius:4px;padding:8px 10px;}
.eye-diagram-app .meter .k{font-size:11px;color:#666;}
.eye-diagram-app .meter .v{font-family:monospace;font-size:20px;font-weight:bold;margin-top:4px;color:#222;line-height:1;}
.eye-diagram-app .meter .v small{font-size:12px;color:#888;font-weight:normal;margin-left:2px;}
.eye-diagram-app .panel{background:#fafafa;border:1px solid #ccc;border-radius:4px;padding:14px;}
.eye-diagram-app .panel h2{font-size:13px;color:#333;margin:0 0 12px;font-weight:bold;}
.eye-diagram-app .seg{display:flex;gap:4px;margin-bottom:6px;}
.eye-diagram-app .seg button{flex:1;border:1px solid #bbb;background:#fff;color:#333;
  font-family:sans-serif;font-weight:normal;font-size:13px;padding:6px 4px;border-radius:4px;cursor:pointer;}
.eye-diagram-app .seg button[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:bold;}
.eye-diagram-app .seg button:hover:not([aria-pressed="true"]){background:#f0f0f0;}
.eye-diagram-app .seg-hint{font-size:12px;color:#666;line-height:1.4;margin:0 0 14px;}
.eye-diagram-app .ctrl{margin-bottom:14px;}
.eye-diagram-app .ctrl .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;}
.eye-diagram-app .ctrl label{font-size:13px;color:#333;}
.eye-diagram-app .ctrl .val{font-family:monospace;font-size:13px;color:var(--accent);font-weight:bold;}
.eye-diagram-app .ctrl .hint{font-size:11.5px;color:#777;margin-top:4px;line-height:1.4;}
.eye-diagram-app input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;margin:2px 0;}
.eye-diagram-app .actions{display:flex;gap:8px;margin-top:4px;}
.eye-diagram-app .actions button{flex:1;font-family:sans-serif;font-weight:normal;font-size:13px;padding:8px;
  border:1px solid #bbb;background:#fff;color:#333;border-radius:4px;cursor:pointer;}
.eye-diagram-app .actions button:hover{background:#f0f0f0;}`;
    document.head.appendChild(style);
  }

  // ---- build DOM inside the container ----
  const root = document.createElement("div");
  root.className = "eye-diagram-app";
  root.innerHTML = `
  <div class="lab">
    <section class="screen">
      <div class="screen-label"><span><span class="dot"></span>Transmitted baseband</span></div>
      <div class="canvas-holder"><canvas id="ed-strip" width="680" height="92"></canvas></div>

      <div class="screen-label" style="margin-top:12px;"><span><span class="dot" style="background:#31a354;"></span>Eye diagram · overlaid bits</span></div>
      <div class="canvas-holder"><canvas id="ed-eye" width="680" height="380"></canvas></div>

      <div class="meters">
        <div class="meter"><div class="k">Eye height</div><div class="v" id="ed-m-height">—<small>%</small></div></div>
        <div class="meter"><div class="k">Eye width</div><div class="v" id="ed-m-width">—<small>%UI</small></div></div>
      </div>
    </section>

    <aside class="panel">
      <h2>Channel + Controls</h2>

      <div class="seg" id="ed-levels" role="group" aria-label="Modulation">
        <button data-levels="2" aria-pressed="true">BPSK</button>
        <button data-levels="4" aria-pressed="false">4-ASK</button>
      </div>
      <p class="seg-hint">BPSK sends one bit per symbol (two amplitudes). 4-ASK packs two bits into four amplitudes → three stacked eyes, each about a third the height, so it needs more Eb/N0.</p>

      <div class="seg" id="ed-shape" role="group" aria-label="Pulse-shaping filter">
        <button data-shape="rc" aria-pressed="true">Full RC</button>
        <button data-shape="rrc" aria-pressed="false">Root RC</button>
      </div>
      <p class="seg-hint">Full raised cosine is ISI-free at the sample point. A single root-raised-cosine filter isn't — pair it with a matching receive filter in a real link.</p>

      <div class="seg" id="ed-view" role="group" aria-label="Eye plot style">
        <button data-view="lines" aria-pressed="true">Lines</button>
        <button data-view="heat" aria-pressed="false">Heatmap</button>
      </div>
      <p class="seg-hint">Draw each trace as a line, or color the eye by how often the traces pass through each point (a density heatmap) — hot spots mark the most common paths.</p>

      <div class="ctrl">
        <div class="row"><label for="ed-rolloff">Roll-off factor β</label><span class="val" id="ed-rolloff-v">0.35</span></div>
        <input type="range" id="ed-rolloff" min="0.1" max="1" step="0.01" value="0.35">
        <div class="hint">Sets occupied bandwidth = (1+β)·Rs⁄2. Small β is spectrum-thrifty but rings hard; large β is wider but gentle.</div>
      </div>

      <div class="ctrl">
        <div class="row"><label for="ed-ebn0">Eb/N0</label><span class="val" id="ed-ebn0-v">30.0 dB</span></div>
        <input type="range" id="ed-ebn0" min="2" max="30" step="0.5" value="30">
        <div class="hint">Signal-to-noise per bit. Lower it and Gaussian noise fattens the traces, closing the eye vertically.</div>
      </div>

      <div class="ctrl">
        <div class="row"><label for="ed-jitter">Timing jitter</label><span class="val" id="ed-jitter-v">0%</span></div>
        <input type="range" id="ed-jitter" min="0" max="22" step="0.5" value="0">
        <div class="hint">RMS clock wobble as a % of one bit period. Smears the crossings and pinches the eye horizontally.</div>
      </div>

      <div class="ctrl">
        <div class="row"><label for="ed-persist">Persistence</label><span class="val" id="ed-persist-v">Medium</span></div>
        <input type="range" id="ed-persist" min="0.70" max="0.965" step="0.005" value="0.9">
        <div class="hint">How long past traces glow before fading — like a phosphor scope.</div>
      </div>

      <div class="actions">
        <button id="ed-run" aria-pressed="true">Pause</button>
        <button id="ed-reset">Reset</button>
      </div>
    </aside>
  </div>`;
  container.appendChild(root);

  const $ = (sel) => root.querySelector(sel);

  // ================= app logic (scoped to this container) =================
  const SPS = 40, SPAN = 8, L = SPS * SPAN; // samples/bit, filter half-span (symbols), half-taps
  const VMIN = -2.25, VMAX = 2.25;
  const TRACES = 46;                        // fresh bits drawn per frame
  const STROKE = 'rgba(45,212,191,0.15)';   // additive teal; dense overlaps saturate to white
  const BG = [7, 12, 20];                   // phosphor background

  // heatmap color LUT (density → color): dark → blue → cyan → green → yellow → red
  const HEAT_LUT = (() => {
    const stops = [[0, [7, 12, 20]], [0.15, [26, 30, 120]], [0.38, [0, 150, 205]],
                   [0.58, [0, 200, 90]], [0.78, [245, 220, 40]], [1, [235, 60, 30]]];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255; let a = stops[0], b = stops[stops.length - 1];
      for (let s = 0; s < stops.length - 1; s++) { if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; } }
      const f = (t - a[0]) / ((b[0] - a[0]) || 1);
      lut[i * 3] = a[1][0] + (b[1][0] - a[1][0]) * f;
      lut[i * 3 + 1] = a[1][1] + (b[1][1] - a[1][1]) * f;
      lut[i * 3 + 2] = a[1][2] + (b[1][2] - a[1][2]) * f;
    }
    return lut;
  })();
  let heatMax = 60, heatImg = null;         // running density peak + reusable output buffer

  const eye = $('#ed-eye'), strip = $('#ed-strip');
  const ex = eye.getContext('2d'), sx = strip.getContext('2d');
  const EW = eye.width, EH = eye.height, SW = strip.width, SH = strip.height;

  // offscreen "phosphor" layer holds the persistent, additively-blended traces
  const phos = document.createElement('canvas'); phos.width = EW; phos.height = EH;
  const px = phos.getContext('2d');
  px.fillStyle = 'rgb(' + BG.join(',') + ')'; px.fillRect(0, 0, EW, EH);

  const params = { rolloff: 0.35, ebn0: 30, jitter: 0, persist: 0.9, shape: 'rc', levels: 2, heatmap: false };
  let running = true;
  let stripWave = null, stripPtr = 0, stripDirty = true;

  // ---------- math ----------
  function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
  function rcTap(x, b) { if (x === 0) return 1;
    const d = 1 - (2 * b * x) * (2 * b * x);
    if (Math.abs(d) < 1e-8) return (Math.PI / 4) * sinc(1 / (2 * b));
    return sinc(x) * Math.cos(Math.PI * b * x) / d; }
  function rrcTap(x, b) { if (x === 0) return 1 - b + 4 * b / Math.PI;
    if (Math.abs(Math.abs(x) - 1 / (4 * b)) < 1e-8) {
      const a = (1 + 2 / Math.PI) * Math.sin(Math.PI / (4 * b));
      const c = (1 - 2 / Math.PI) * Math.cos(Math.PI / (4 * b));
      return (b / Math.SQRT2) * (a + c); }
    const p = Math.PI * x, num = Math.sin(p * (1 - b)) + 4 * b * x * Math.cos(p * (1 + b));
    const den = p * (1 - (4 * b * x) * (4 * b * x)); return num / den; }
  const ebn0Lin = () => Math.pow(10, params.ebn0 / 10);
  const LEVELSETS = { 2: [-1, 1], 4: [-1, -1 / 3, 1 / 3, 1] };
  const levelArr = () => LEVELSETS[params.levels];
  function ebPerBit() { const lv = levelArr(); let es = 0; for (const a of lv) es += a * a; return (es / lv.length) / Math.log2(lv.length); }
  const sigma = () => Math.sqrt(ebPerBit() / (2 * ebn0Lin()));   // noise per real dimension, energy-per-bit accurate

  // ---------- pulse-shaping filter (cached) ----------
  let filt = null, filtDirty = true;
  function buildFilter() {
    const fn = params.shape === 'rrc' ? rrcTap : rcTap, b = params.rolloff;
    const h = new Float32Array(2 * L + 1), c = fn(0, b);
    for (let n = -L; n <= L; n++) h[n + L] = fn(n / SPS, b) / c;
    filt = h; filtDirty = false;
  }
  function makeWaveform(bits, addNoise) {
    if (filtDirty) buildFilter();
    const N = bits.length * SPS, out = new Float32Array(N);
    for (let k = 0; k < bits.length; k++) { const a = bits[k], base = k * SPS + (SPS >> 1);
      for (let n = -L; n <= L; n++) { const i = base + n; if (i >= 0 && i < N) out[i] += a * filt[n + L]; } }
    if (addNoise) { const s = sigma(); for (let i = 0; i < N; i++) out[i] += randn() * s; }
    return out;
  }
  function genSymbols(n) { const lv = levelArr(), m = lv.length, s = new Float32Array(n); for (let i = 0; i < n; i++) s[i] = lv[(Math.random() * m) | 0]; return s; }

  const mapX = t => ((t + 1) / 2) * (EW - 1);
  const mapY = v => (1 - (v - VMIN) / (VMAX - VMIN)) * (EH - 1);

  // ---------- draw fresh traces onto the phosphor layer (native anti-aliased lines) ----------
  function drawTraces() {
    const nT = params.levels === 4 ? 64 : TRACES;
    const bits = genSymbols(nT + 2 * SPAN);
    const wf = makeWaveform(bits, false);
    const s = sigma(), js = params.jitter * SPS;
    px.globalCompositeOperation = 'lighter';
    px.strokeStyle = STROKE; px.lineWidth = 1; px.lineJoin = 'round';
    for (let k = SPAN; k < bits.length - SPAN; k++) {
      const c = k * SPS + (SPS >> 1), jsh = Math.round(js * randn());
      px.beginPath();
      for (let o = 0; o <= 2 * SPS; o++) {
        let idx = c - SPS + o + jsh; if (idx < 0) idx = 0; else if (idx >= wf.length) idx = wf.length - 1;
        const v = wf[idx] + randn() * s, x = mapX((o - SPS) / SPS), y = mapY(v);
        if (o === 0) px.moveTo(x, y); else px.lineTo(x, y);
      }
      px.stroke();
    }
    px.globalCompositeOperation = 'source-over';
  }

  // ---------- measurement (reads the rendered phosphor, throttled) ----------
  const wLum = (r, g, b) => 0.25 * r + 0.6 * g + 0.15 * b;
  const BGL = wLum(BG[0], BG[1], BG[2]);
  function measure() {
    const d = px.getImageData(0, 0, EW, EH).data;
    const cx = Math.round(mapX(0)), cy = Math.round(mapY(0));
    let mx = 0; for (let i = 0; i < d.length; i += 4) { const l = wLum(d[i], d[i + 1], d[i + 2]) - BGL; if (l > mx) mx = l; }
    heatMax = mx;                           // reuse the peak density to normalize the heatmap
    if (mx < 2) return { heightPct: 0, widthPct: 0, hPx: 0, wPx: 0, cx, cy };
    const thr = mx * 0.10;
    const sig = (r, c) => { const i = (r * EW + c) * 4; return wLum(d[i], d[i + 1], d[i + 2]) - BGL; };
    const avgC = r => { let s = 0; for (let c = cx - 2; c <= cx + 2; c++) s += sig(r, c); return s * 0.2; };
    const avgR = c => { let s = 0; for (let r = cy - 2; r <= cy + 2; r++) s += sig(r, c); return s * 0.2; };
    let hPx = 0, wPx = 0;
    if (avgC(cy) < thr) { let top = cy, bot = cy; while (top > 0 && avgC(top - 1) < thr) top--; while (bot < EH - 1 && avgC(bot + 1) < thr) bot++; hPx = bot - top; }
    if (avgR(cx) < thr) { let l = cx, r = cx; while (l > 0 && avgR(l - 1) < thr) l--; while (r < EW - 1 && avgR(r + 1) < thr) r++; wPx = r - l; }
    const hFrac = hPx / EH * (VMAX - VMIN);
    return { heightPct: Math.max(0, hFrac / 2 * 100), widthPct: Math.max(0, wPx / EW * 2 * 100), hPx, wPx, cx, cy };
  }

  // ---------- overlays (grid, sampling line, calipers) ----------
  function cap(x, y, dir) { ex.beginPath();
    if (dir === 'h') { ex.moveTo(x - 6, y); ex.lineTo(x + 6, y); } else { ex.moveTo(x, y - 6); ex.lineTo(x, y + 6); } ex.stroke(); }
  function drawOverlays(m) {
    ex.save();
    ex.strokeStyle = 'rgba(120,140,175,0.14)'; ex.lineWidth = 1;
    for (const t of [-1, -0.5, 0.5, 1]) { const x = mapX(t) | 0; ex.beginPath(); ex.moveTo(x + .5, 0); ex.lineTo(x + .5, EH); ex.stroke(); }
    for (const v of [-1, 0, 1]) { const y = mapY(v) | 0; ex.beginPath(); ex.moveTo(0, y + .5); ex.lineTo(EW, y + .5); ex.stroke(); }

    ex.strokeStyle = 'rgba(230,85,13,0.65)'; ex.setLineDash([5, 5]); ex.lineWidth = 1.5;
    ex.beginPath(); ex.moveTo(m.cx + .5, 0); ex.lineTo(m.cx + .5, EH); ex.stroke(); ex.setLineDash([]);

    if (m.hPx > 4 || m.wPx > 4) {
      ex.strokeStyle = '#e6550d'; ex.lineWidth = 2;
      const top = m.cy - m.hPx / 2, bot = m.cy + m.hPx / 2, l = m.cx - m.wPx / 2, r = m.cx + m.wPx / 2;
      if (m.hPx > 4) { ex.beginPath(); ex.moveTo(m.cx, top); ex.lineTo(m.cx, bot); ex.stroke(); cap(m.cx, top, 'h'); cap(m.cx, bot, 'h'); }
      if (m.wPx > 4) { ex.beginPath(); ex.moveTo(l, m.cy); ex.lineTo(r, m.cy); ex.stroke(); cap(l, m.cy, 'v'); cap(r, m.cy, 'v'); }
    }
    ex.fillStyle = 'rgba(150,166,196,0.8)'; ex.font = '11px "IBM Plex Mono",ui-monospace,monospace';
    ex.textAlign = 'center'; ex.fillText('sample here', m.cx, EH - 8);
    ex.textAlign = 'left'; ex.fillText('−1 UI', 6, EH - 8);
    ex.textAlign = 'right'; ex.fillText('+1 UI', EW - 6, EH - 8);
    ex.restore();
  }

  // ---------- heatmap display (remap the phosphor density through a color LUT) ----------
  function applyHeatmap() {
    const src = px.getImageData(0, 0, EW, EH).data;
    if (!heatImg) heatImg = ex.createImageData(EW, EH);
    const out = heatImg.data, norm = 1 / Math.max(heatMax, 8);
    for (let i = 0; i < src.length; i += 4) {
      let l = (wLum(src[i], src[i + 1], src[i + 2]) - BGL) * norm;
      if (l < 0) l = 0; else if (l > 1) l = 1;
      l = Math.sqrt(l);                                        // gamma lift so faint traces are visible
      const j = ((l * 255) | 0) * 3;
      out[i] = HEAT_LUT[j]; out[i + 1] = HEAT_LUT[j + 1]; out[i + 2] = HEAT_LUT[j + 2]; out[i + 3] = 255;
    }
    ex.putImageData(heatImg, 0, 0);
  }

  // ---------- render ----------
  let frameCount = 0, meas = { heightPct: 0, widthPct: 0, hPx: 0, wPx: 0, cx: Math.round(mapX(0)), cy: Math.round(mapY(0)) };
  function renderEye() {
    px.globalCompositeOperation = 'source-over';               // fade previous traces (persistence)
    px.fillStyle = 'rgba(' + BG[0] + ',' + BG[1] + ',' + BG[2] + ',' + (1 - params.persist).toFixed(3) + ')';
    px.fillRect(0, 0, EW, EH);
    drawTraces();
    if ((frameCount++ % 6) === 0) meas = measure();
    ex.clearRect(0, 0, EW, EH);
    if (params.heatmap) applyHeatmap(); else ex.drawImage(phos, 0, 0);
    drawOverlays(meas);
    updateMeters(meas);
  }
  function updateMeters(m) {
    $('#ed-m-height').innerHTML = m.heightPct.toFixed(0) + '<small>%</small>';
    $('#ed-m-width').innerHTML = m.widthPct.toFixed(0) + '<small>%UI</small>';
  }
  function clearPhosphor() { px.globalCompositeOperation = 'source-over'; px.fillStyle = 'rgb(' + BG.join(',') + ')'; px.fillRect(0, 0, EW, EH); }

  // ---------- transmitted-signal strip ----------
  function rebuildStrip() { stripWave = makeWaveform(genSymbols(200), true); stripPtr = 0; stripDirty = false; }
  function renderStrip() {
    if (stripDirty || !stripWave) rebuildStrip();
    const visible = 12 * SPS;
    if (running) { stripPtr += Math.round(SPS / 6); if (stripPtr + visible >= stripWave.length) rebuildStrip(); }
    sx.clearRect(0, 0, SW, SH);
    sx.strokeStyle = 'rgba(120,140,175,0.10)'; sx.lineWidth = 1;
    for (const v of [-1, 0, 1]) { const y = (1 - (v + 2.25) / 4.5) * SH | 0; sx.beginPath(); sx.moveTo(0, y + .5); sx.lineTo(SW, y + .5); sx.stroke(); }
    sx.beginPath(); sx.lineWidth = 2; sx.strokeStyle = '#17c3b2'; sx.shadowBlur = 8; sx.shadowColor = 'rgba(23,195,178,0.6)';
    for (let i = 0; i < visible; i++) { const s = stripWave[stripPtr + i], x = i / visible * SW, y = (1 - (s + 2.25) / 4.5) * SH;
      if (i === 0) sx.moveTo(x, y); else sx.lineTo(x, y); }
    sx.stroke(); sx.shadowBlur = 0;
  }

  function frame() { if (running) renderEye(); renderStrip(); requestAnimationFrame(frame); }

  // ---------- controls ----------
  function fill(el) { const min = +el.min, max = +el.max, v = +el.value; el.style.setProperty('--fill', ((v - min) / (max - min) * 100) + '%'); }
  function bindRange(id, fmt, apply) { const el = $('#' + id), out = $('#' + id + '-v');
    const upd = () => { apply(+el.value); out.textContent = fmt(+el.value); fill(el); }; el.addEventListener('input', upd); upd(); }
  bindRange('ed-rolloff', v => v.toFixed(2), v => { params.rolloff = v; filtDirty = true; stripDirty = true; });
  bindRange('ed-ebn0', v => v.toFixed(1) + ' dB', v => { params.ebn0 = v; stripDirty = true; });
  bindRange('ed-jitter', v => v.toFixed(0) + '%', v => { params.jitter = v / 100; });
  bindRange('ed-persist', v => v < 0.8 ? 'Short' : v < 0.9 ? 'Medium' : v < 0.94 ? 'Long' : 'Very long', v => { params.persist = v; });

  $('#ed-levels').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    params.levels = +b.dataset.levels; stripDirty = true; clearPhosphor();
    [...e.currentTarget.children].forEach(x => x.setAttribute('aria-pressed', x === b));
  });

  $('#ed-shape').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    params.shape = b.dataset.shape; filtDirty = true; stripDirty = true; clearPhosphor();
    [...e.currentTarget.children].forEach(x => x.setAttribute('aria-pressed', x === b));
  });

  $('#ed-view').addEventListener('click', e => {              // Lines vs Heatmap: display-only, no phosphor reset needed
    const b = e.target.closest('button'); if (!b) return;
    params.heatmap = b.dataset.view === 'heat';
    [...e.currentTarget.children].forEach(x => x.setAttribute('aria-pressed', x === b));
  });

  const runBtn = $('#ed-run');
  runBtn.addEventListener('click', () => { running = !running; runBtn.textContent = running ? 'Pause' : 'Run';
    runBtn.classList.toggle('paused', !running); runBtn.setAttribute('aria-pressed', running); });
  $('#ed-reset').addEventListener('click', () => { clearPhosphor();
    const set = (id, val) => { const el = $('#' + id); el.value = val; el.dispatchEvent(new Event('input')); };
    set('ed-rolloff', 0.35); set('ed-ebn0', 30); set('ed-jitter', 0); set('ed-persist', 0.9);
    root.querySelector('[data-shape="rc"]').click();
    root.querySelector('[data-levels="2"]').click();
    root.querySelector('[data-view="lines"]').click(); });

  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    for (let i = 0; i < 26; i++) renderEye();
    running = false; runBtn.textContent = 'Run'; runBtn.classList.add('paused'); runBtn.setAttribute('aria-pressed', false);
  }
  requestAnimationFrame(frame);
}
