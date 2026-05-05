/* ============================================================
 * Entrocraft Interactive Demo
 *  - Drag entropy control points
 *  - Watch the AIME-25 accuracy curve react in real time
 * ============================================================ */

const SVG_NS = "http://www.w3.org/2000/svg";

/* ---------- Chart geometry (shared by both charts) ---------- */
const W = 720;
const H = 380;
const M = { left: 58, right: 22, top: 22, bottom: 46 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

const X_MAX = 400;          // training samples (in thousands)
const NUM_SIM = 220;        // number of simulation steps
const NUM_CTRL = 9;         // number of draggable entropy control points

const ENTROPY_MIN = 0;
const ENTROPY_MAX = 0.9;
const ACC_MIN = 4;
const ACC_MAX = 16;

/* Recommended entropy bounds (matches the paper's annealing scheme) */
function recommendedUpper(t) {
  return 0.65 - 0.30 * (t / X_MAX);
}
function recommendedLower(t) {
  return 0.45 - 0.30 * (t / X_MAX);
}

/* ---------- Coordinate helpers ---------- */
const xScaleEntropy = (t) => M.left + (t / X_MAX) * PW;
const yScaleEntropy = (h) =>
  M.top + (1 - (h - ENTROPY_MIN) / (ENTROPY_MAX - ENTROPY_MIN)) * PH;
const xScaleAcc = (t) => M.left + (t / X_MAX) * PW;
const yScaleAcc = (a) =>
  M.top + (1 - (a - ACC_MIN) / (ACC_MAX - ACC_MIN)) * PH;
const yToEntropy = (y) =>
  ENTROPY_MIN + (1 - (y - M.top) / PH) * (ENTROPY_MAX - ENTROPY_MIN);

/* ---------- Tiny utilities ---------- */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function el(name, attrs = {}, parent = null) {
  const e = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

/* ---------- Smooth path generation (Catmull-Rom -> Bezier) ---------- */
function smoothPath(points) {
  if (points.length === 0) return "";
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  let d = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i - 1] || points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }
  return d;
}

/* ---------- Linear interpolation for entropy at simulation steps ---------- */
function interpEntropyAt(controlPoints, t) {
  const pts = controlPoints;
  if (t <= pts[0].t) return pts[0].h;
  if (t >= pts[pts.length - 1].t) return pts[pts.length - 1].h;
  for (let i = 0; i < pts.length - 1; i++) {
    if (t >= pts[i].t && t <= pts[i + 1].t) {
      const span = pts[i + 1].t - pts[i].t;
      const r = span === 0 ? 0 : (t - pts[i].t) / span;
      // Use a smoothstep blend for visual smoothness
      const s = r * r * (3 - 2 * r);
      return pts[i].h + s * (pts[i + 1].h - pts[i].h);
    }
  }
  return pts[pts.length - 1].h;
}

/* ============================================================
 *  ACCURACY SIMULATION
 *
 *  Intuition (paper-aligned):
 *    - Low entropy  -> high learning rate, but low ceiling
 *                      (model commits to narrow solutions)
 *    - High entropy -> low learning rate, high noise
 *                      (model wanders, slow + jittery)
 *    - Annealed mid -> steady gain, high stable ceiling
 * ============================================================ */
function simulateAccuracy(entropySamples, seed = 7) {
  const rng = mulberry32(seed);
  const n = entropySamples.length;
  const acc = new Array(n);

  acc[0] = 4 + gauss(rng) * 0.12;

  // The "achievable ceiling" grows monotonically with sustained entropy.
  let ceiling = 4.6;
  let smoothNoise = 0;

  // Cumulative "instability damage" from any time the entropy goes above
  // the recommended upper bound. This is the key Entrocraft insight:
  // ANY excursion above the safe band leaves a lasting scar on the policy
  // that can't be undone by later good behavior.
  let cumDamage = 0;

  for (let t = 1; t < n; t++) {
    const h = clamp(entropySamples[t], 0, 1);
    const trainingStep = (t / (n - 1)) * X_MAX;

    // Ceiling growth ∝ h^1.8  (low entropy contributes very little)
    const dCeil = 0.085 * Math.pow(h, 1.8) * Math.max(0, ACC_MAX - ceiling);
    ceiling += dCeil;

    // Going above the recommended upper bound accumulates permanent damage
    const ub = recommendedUpper(trainingStep);
    if (h > ub) {
      cumDamage += Math.pow(h - ub, 1.4) * 1.6;
    }
    // Saturating penalty (one bad spike cannot drop you to zero)
    const damagePenalty = 4.0 * Math.tanh(cumDamage / 5.0);

    // Effective ceiling: damaged by out-of-band exposure plus an
    // instantaneous high-entropy noise penalty.
    const effectiveCeiling =
      ceiling - damagePenalty - Math.pow(h, 2) * 1.0;

    // Learning rate: bell curve peaking at h ≈ 0.4 (the "sweet spot").
    //   - very low h  -> slow because there is no exploration signal
    //   - moderate h  -> fast, balanced policy gradient
    //   - very high h -> slow because rollouts are mostly noise
    const lr = 0.04 + 0.20 * Math.exp(-Math.pow((h - 0.40) / 0.25, 2));

    let delta = lr * (effectiveCeiling - acc[t - 1]);

    // Soft saturation when close to the effective ceiling
    const ratio =
      (acc[t - 1] - ACC_MIN) / Math.max(0.5, effectiveCeiling - ACC_MIN);
    if (ratio > 0.85 && delta > 0) delta *= 0.4;

    // Jitter grows super-linearly with entropy -> visible spikes at high h
    const sigma = 0.04 + Math.pow(h, 1.4) * 1.55;
    smoothNoise = smoothNoise * 0.55 + gauss(rng) * sigma * 0.45;

    let next = acc[t - 1] + delta + smoothNoise * 0.6;

    // Instability can cut accuracy roughly in half. Inside the recommended
    // range, this is only a tiny background risk; it mainly appears when
    // entropy is meaningfully above the recommended upper bound, and the
    // same excess is riskier later in training.
    const trainProgress = t / (n - 1);
    const progressRisk = Math.pow(trainProgress, 1.6);
    const backgroundChance = 0.00005 * Math.pow(h, 2) * (0.2 + 0.8 * progressRisk);
    const safeUpper = Math.min(ENTROPY_MAX, ub + 0.06);
    const excessAboveSafe = Math.max(0, h - safeUpper);
    const normalizedExcess = excessAboveSafe / Math.max(0.01, ENTROPY_MAX - safeUpper);
    const entropyRisk = Math.pow(normalizedExcess, 2.4);
    const excessChance = 0.08 * entropyRisk * (0.10 + 2.40 * progressRisk);
    const halfCutChance = clamp(backgroundChance + excessChance, 0, 0.20);

    if (rng() < halfCutChance) {
      const keepRatio = 0.60 - rng() * (0.10 + 0.05 * trainProgress);
      next *= keepRatio;
    }

    next = clamp(next, ACC_MIN - 0.3, ACC_MAX + 0.3);
    acc[t] = next;
  }
  return acc;
}

/* ============================================================
 *  CHART INFRASTRUCTURE
 * ============================================================ */
function buildAxes(svg, opts) {
  // Background bands
  if (opts.bands) {
    for (const band of opts.bands) {
      el("rect", {
        x: M.left,
        y: M.top + (1 - band.yMaxNorm) * PH,
        width: PW,
        height: (band.yMaxNorm - band.yMinNorm) * PH,
        class: band.cls,
      }, svg);
    }
  }
  if (opts.background) opts.background(svg);

  // Plot border
  el("rect", {
    x: M.left,
    y: M.top,
    width: PW,
    height: PH,
    fill: "none",
    stroke: "#dadcd6",
    "stroke-width": 1,
  }, svg);

  // Y grid + ticks
  const gridG = el("g", { class: "grid" }, svg);
  const axisG = el("g", { class: "axis" }, svg);
  for (const tick of opts.yTicks) {
    const yPx = opts.yScale(tick);
    el("line", {
      x1: M.left, x2: M.left + PW,
      y1: yPx, y2: yPx,
    }, gridG);
    el("text", {
      x: M.left - 10,
      y: yPx + 3.5,
      "text-anchor": "end",
    }, axisG).textContent = opts.yFormat ? opts.yFormat(tick) : String(tick);
  }

  // X ticks
  for (const tick of opts.xTicks) {
    const xPx = opts.xScale(tick);
    el("line", {
      x1: xPx, x2: xPx,
      y1: M.top + PH, y2: M.top + PH + 5,
    }, axisG);
    el("text", {
      x: xPx,
      y: M.top + PH + 18,
      "text-anchor": "middle",
    }, axisG).textContent = opts.xFormat ? opts.xFormat(tick) : String(tick);
  }

  // Axis labels
  el("text", {
    x: M.left + PW / 2,
    y: H - 8,
    "text-anchor": "middle",
    class: "axis-label",
  }, svg).textContent = opts.xLabel || "Training Samples";

  el("text", {
    x: 14,
    y: M.top + PH / 2,
    "text-anchor": "middle",
    transform: `rotate(-90 14 ${M.top + PH / 2})`,
    class: "axis-label",
  }, svg).textContent = opts.yLabel || "";
}

/* ============================================================
 *  STATE
 * ============================================================ */
function buildLinearControl() {
  const pts = [];
  for (let i = 0; i < NUM_CTRL; i++) {
    const t = (i / (NUM_CTRL - 1)) * X_MAX;
    const h = 0.6 - 0.4 * (i / (NUM_CTRL - 1)); // 0.6 -> 0.2
    pts.push({ t, h });
  }
  return pts;
}

let entropyControls = buildLinearControl();

/* Pre-compute the GRPO baseline accuracy curve once.
 * It uses a fast exponentially collapsing entropy schedule. */
function buildBaselineEntropySamples() {
  const arr = new Array(NUM_SIM);
  const start = 0.8;
  const end = 0.0001;
  for (let i = 0; i < NUM_SIM; i++) {
    const r = i / (NUM_SIM - 1);
    arr[i] = start * Math.pow(end / start, r);
  }
  return arr;
}
const baselineEntropy = buildBaselineEntropySamples();
const baselineAccuracy = simulateAccuracy(baselineEntropy, 11);

/* ============================================================
 *  RENDERING
 * ============================================================ */
const entropySvg = document.getElementById("entropyChart");
const accSvg = document.getElementById("accuracyChart");
const summary = document.getElementById("summary");

function straightPath(points) {
  if (points.length === 0) return "";
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(" ");
}

function entropyBandPath(topPoints, bottomPoints) {
  return `${straightPath(topPoints)} ${straightPath([...bottomPoints].reverse()).replace("M", "L")} Z`;
}

/* --- Static layers (axes, bands, bounds) for entropy chart --- */
function initEntropyChartStatic() {
  entropySvg.innerHTML = "";

  buildAxes(entropySvg, {
    background: (svg) => {
      const topPts = [];
      const upperPts = [];
      const lowerPts = [];
      const bottomPts = [];

      for (let t = 0; t <= X_MAX; t += 20) {
        const x = xScaleEntropy(t);
        topPts.push({ x, y: yScaleEntropy(ENTROPY_MAX) });
        upperPts.push({ x, y: yScaleEntropy(recommendedUpper(t)) });
        lowerPts.push({ x, y: yScaleEntropy(recommendedLower(t)) });
        bottomPts.push({ x, y: yScaleEntropy(ENTROPY_MIN) });
      }

      el("path", {
        d: entropyBandPath(topPts, upperPts),
        class: "bg-band-bad",
      }, svg);
      el("path", {
        d: entropyBandPath(upperPts, lowerPts),
        class: "bg-band-good",
      }, svg);
      el("path", {
        d: entropyBandPath(lowerPts, bottomPts),
        class: "bg-band-bad",
      }, svg);
    },
    xTicks: [0, 100, 200, 300, 400],
    yTicks: [0.0, 0.2, 0.4, 0.6, 0.8],
    xScale: xScaleEntropy,
    yScale: yScaleEntropy,
    xFormat: (v) => (v === 0 ? "0" : `${v}k`),
    yFormat: (v) => v.toFixed(1),
    xLabel: "Training Samples",
    yLabel: "Entropy 𝓗",
  });

  // Recommended bound dashed lines (annealed)
  const upperPts = [];
  const lowerPts = [];
  for (let t = 0; t <= X_MAX; t += 20) {
    upperPts.push({ x: xScaleEntropy(t), y: yScaleEntropy(recommendedUpper(t)) });
    lowerPts.push({ x: xScaleEntropy(t), y: yScaleEntropy(recommendedLower(t)) });
  }
  el("path", {
    d: smoothPath(upperPts),
    class: "bound",
  }, entropySvg);
  el("path", {
    d: smoothPath(lowerPts),
    class: "bound",
  }, entropySvg);

  const grpoPts = baselineEntropy.map((h, i) => ({
    x: xScaleEntropy((i / (NUM_SIM - 1)) * X_MAX),
    y: yScaleEntropy(h),
  }));
  el("path", {
    d: smoothPath(grpoPts),
    class: "entropy-baseline",
  }, entropySvg);

  el("text", {
    x: xScaleEntropy(38),
    y: yScaleEntropy(recommendedUpper(38)) - 6,
    fill: "var(--green-strong)",
    "font-size": "11.5",
    "font-style": "italic",
    opacity: "0.85",
  }, entropySvg).textContent = "entropy upper bound";

  el("text", {
    x: xScaleEntropy(220),
    y: yScaleEntropy(recommendedLower(220)) + 16,
    fill: "var(--green-strong)",
    "font-size": "11.5",
    "font-style": "italic",
    opacity: "0.85",
  }, entropySvg).textContent = "entropy lower bound";

  // Dynamic layer (curve + handles) appended below
  const dynLayer = el("g", { id: "entropy-dyn" }, entropySvg);
  return dynLayer;
}

let entropyDynLayer = initEntropyChartStatic();

/* --- Static layers for accuracy chart --- */
function initAccChartStatic() {
  accSvg.innerHTML = "";

  buildAxes(accSvg, {
    xTicks: [0, 100, 200, 300, 400],
    yTicks: [4, 6, 8, 10, 12, 14, 16],
    xScale: xScaleAcc,
    yScale: yScaleAcc,
    xFormat: (v) => (v === 0 ? "0" : `${v}k`),
    xLabel: "Training Samples",
    yLabel: "AIME-25 mean@32",
  });

  // Baseline (GRPO) curve
  const basePts = baselineAccuracy.map((a, i) => ({
    x: xScaleAcc((i / (NUM_SIM - 1)) * X_MAX),
    y: yScaleAcc(a),
  }));
  el("path", {
    d: smoothPath(basePts),
    class: "acc-baseline",
  }, accSvg);

  el("text", {
    x: xScaleAcc(330),
    y: yScaleAcc(baselineAccuracy[Math.floor(NUM_SIM * 0.85)]) + 18,
    fill: "var(--gray)",
    "font-size": "12.5",
    "font-weight": "600",
    "text-anchor": "middle",
  }, accSvg).textContent = "GRPO baseline";

  const dynLayer = el("g", { id: "acc-dyn" }, accSvg);
  return dynLayer;
}

let accDynLayer = initAccChartStatic();

/* --- Render dynamic layers --- */
function renderEntropyDynamic() {
  entropyDynLayer.innerHTML = "";

  const renderPts = entropyControls.map((p) => ({
    x: xScaleEntropy(p.t),
    y: yScaleEntropy(p.h),
  }));

  // Filled area under curve
  const pathD = smoothPath(renderPts);
  const areaD =
    pathD +
    ` L ${xScaleEntropy(X_MAX)} ${yScaleEntropy(0)}` +
    ` L ${xScaleEntropy(0)} ${yScaleEntropy(0)} Z`;
  el("path", {
    d: areaD,
    class: "entropy-area",
  }, entropyDynLayer);

  el("path", {
    d: pathD,
    class: "entropy-line",
  }, entropyDynLayer);

  // Draggable handles
  entropyControls.forEach((p, i) => {
    const handle = el("circle", {
      cx: xScaleEntropy(p.t),
      cy: yScaleEntropy(p.h),
      r: 7,
      class: "handle",
      "data-idx": i,
    }, entropyDynLayer);
    attachDrag(handle, i);
  });
}

function renderAccuracyDynamic(entropySamples) {
  accDynLayer.innerHTML = "";
  const acc = simulateAccuracy(entropySamples, 7);
  const pts = acc.map((a, i) => ({
    x: xScaleAcc((i / (NUM_SIM - 1)) * X_MAX),
    y: yScaleAcc(a),
  }));
  el("path", {
    d: smoothPath(pts),
    class: "acc-line",
  }, accDynLayer);

  // Final value label
  const last = acc[acc.length - 1];
  el("text", {
    x: xScaleAcc(X_MAX) - 6,
    y: yScaleAcc(last) - 10,
    "text-anchor": "end",
    fill: "var(--red)",
    "font-size": "12",
    "font-weight": "700",
  }, accDynLayer).textContent = `final ≈ ${last.toFixed(1)}`;

  // Update summary
  updateSummary(entropySamples, acc);
}

/* --- Sample entropy at every simulation step --- */
function sampleEntropy() {
  const arr = new Array(NUM_SIM);
  for (let i = 0; i < NUM_SIM; i++) {
    const t = (i / (NUM_SIM - 1)) * X_MAX;
    arr[i] = clamp(interpEntropyAt(entropyControls, t), 0, ENTROPY_MAX);
  }
  return arr;
}

function rerender() {
  renderEntropyDynamic();
  const samples = sampleEntropy();
  renderAccuracyDynamic(samples);
}

/* --- Summary text below the accuracy chart --- */
function updateSummary(entropySamples, accSamples) {
  const avgH = entropySamples.reduce((s, v) => s + v, 0) / entropySamples.length;
  const startH = entropySamples[0];
  const endH = entropySamples[entropySamples.length - 1];
  const final = accSamples[accSamples.length - 1];
  const baseFinal = baselineAccuracy[baselineAccuracy.length - 1];

  let label;
  if (avgH < 0.18) {
    label = "Entropy too low — model commits early and the accuracy curve plateaus quickly.";
  } else if (avgH > 0.7) {
    label = "Entropy too high — heavy exploration causes a noisy, slow-rising accuracy curve.";
  } else if (startH - endH > 0.25 && startH > 0.45) {
    label = `Entrocraft annealing — steady, stable improvement (final ≈ ${final.toFixed(1)} vs. GRPO ${baseFinal.toFixed(1)}).`;
  } else if (Math.abs(startH - endH) < 0.1) {
    label = "Constant entropy — works, but doesn’t fully exploit the late training phase.";
  } else {
    label = `Custom schedule — final accuracy ≈ ${final.toFixed(1)} vs. GRPO ${baseFinal.toFixed(1)}.`;
  }
  summary.textContent = label;
}

/* ============================================================
 *  DRAG INTERACTION
 * ============================================================ */
function clientToSvgPoint(svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

let activeIdx = -1;

function attachDrag(handle, idx) {
  handle.addEventListener("mousedown", (e) => startDrag(e, idx, handle));
  handle.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) {
      e.preventDefault();
      startDrag(e.touches[0], idx, handle);
    }
  }, { passive: false });
}

function startDrag(evt, idx, handle) {
  activeIdx = idx;
  handle.classList.add("dragging");

  const onMove = (clientX, clientY) => {
    const p = clientToSvgPoint(entropySvg, clientX, clientY);
    const newH = clamp(yToEntropy(p.y), ENTROPY_MIN, ENTROPY_MAX);
    entropyControls[idx].h = newH;
    rerender();
  };

  const mouseMove = (e) => onMove(e.clientX, e.clientY);
  const touchMove = (e) => {
    if (e.touches.length === 1) {
      e.preventDefault();
      onMove(e.touches[0].clientX, e.touches[0].clientY);
    }
  };
  const cleanup = () => {
    activeIdx = -1;
    handle.classList.remove("dragging");
    window.removeEventListener("mousemove", mouseMove);
    window.removeEventListener("mouseup", cleanup);
    window.removeEventListener("touchmove", touchMove);
    window.removeEventListener("touchend", cleanup);
  };

  window.addEventListener("mousemove", mouseMove);
  window.addEventListener("mouseup", cleanup);
  window.addEventListener("touchmove", touchMove, { passive: false });
  window.addEventListener("touchend", cleanup);
}

/* ============================================================
 *  RESET BUTTON
 * ============================================================ */
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    entropyControls = buildLinearControl();
    rerender();
  });
}

/* ============================================================
 *  INITIAL RENDER
 * ============================================================ */
rerender();

/* Re-render on resize for crisp rendering */
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    entropyDynLayer = initEntropyChartStatic();
    accDynLayer = initAccChartStatic();
    rerender();
  }, 120);
});
