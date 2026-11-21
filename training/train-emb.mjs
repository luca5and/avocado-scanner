// Softmax classifier on pretrained-network embeddings (optionally + colour features).
// Evaluation: 5-fold CV grouped by avocado. Usage:
//   node train-emb.mjs cv   <tag>[+<tag>...] [+color]      e.g. dinov2-small-q8+color
//   node train-emb.mjs save <tags> <l2>                    -> ../docs/head.json
import fs from 'node:fs';
const [mode, spec, l2arg] = process.argv.slice(2);
const K = 5;
const { names: colorNames, rows } = JSON.parse(fs.readFileSync('data/features.json'));

// ---- assemble feature matrix ----
const parts = spec.split('+');
const blocks = [];
for (const p of parts) {
  if (p === 'color') { blocks.push({ name: 'color', dim: colorNames.length, get: i => rows[i].x }); continue; }
  const meta = JSON.parse(fs.readFileSync(`data/emb-${p}.json`));
  const raw = fs.readFileSync(`data/emb-${p}.bin`);
  const emb = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const pos = new Map(meta.files.map((f, i) => [f, i]));
  blocks.push({ name: p, dim: meta.dim, modelId: meta.modelId, dtype: meta.dtype,
    get: i => emb.subarray(pos.get(rows[i].file) * meta.dim, (pos.get(rows[i].file) + 1) * meta.dim) });
}
const D = blocks.reduce((s, b) => s + b.dim, 0), N = rows.length;
const X = new Float32Array(N * D);
for (let i = 0; i < N; i++) { let o = 0; for (const b of blocks) { X.set(b.get(i), i * D + o); o += b.dim; } }
const Y = Int32Array.from(rows, r => r.stage - 1);

let seed = 7; const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

function fit(idx, l2, epochs = 30) {
  // standardise on training rows
  const mu = new Float64Array(D), sd = new Float64Array(D);
  for (const i of idx) for (let j = 0; j < D; j++) mu[j] += X[i * D + j];
  for (let j = 0; j < D; j++) mu[j] /= idx.length;
  for (const i of idx) for (let j = 0; j < D; j++) sd[j] += (X[i * D + j] - mu[j]) ** 2;
  for (let j = 0; j < D; j++) sd[j] = Math.sqrt(sd[j] / idx.length) || 1;
  const W = new Float32Array(K * D), b = new Float32Array(K);
  const mW = new Float32Array(K * D), vW = new Float32Array(K * D), mb = new Float32Array(K), vb = new Float32Array(K);
  const gW = new Float32Array(K * D), gb = new Float32Array(K), x = new Float32Array(D), z = new Float32Array(K);
  const order = Int32Array.from(idx); let t = 0; const lr = 0.002, bs = 128;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let s = 0; s < order.length; s += bs) {
      gW.fill(0); gb.fill(0); const e = Math.min(order.length, s + bs);
      for (let q = s; q < e; q++) {
        const i = order[q];
        for (let j = 0; j < D; j++) x[j] = (X[i * D + j] - mu[j]) / sd[j];
        let mx = -Infinity;
        for (let k = 0; k < K; k++) { let v = b[k]; for (let j = 0; j < D; j++) v += W[k * D + j] * x[j]; z[k] = v; if (v > mx) mx = v; }
        let se = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); se += z[k]; }
        for (let k = 0; k < K; k++) { const d = z[k] / se - (k === Y[i] ? 1 : 0); gb[k] += d; for (let j = 0; j < D; j++) gW[k * D + j] += d * x[j]; }
      }
      t++; const n = e - s, c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let p = 0; p < K * D; p++) {
        const g = gW[p] / n + l2 * W[p];
        mW[p] = 0.9 * mW[p] + 0.1 * g; vW[p] = 0.999 * vW[p] + 0.001 * g * g;
        W[p] -= lr * (mW[p] / c1) / (Math.sqrt(vW[p] / c2) + 1e-8);
      }
      for (let k = 0; k < K; k++) {
        const g = gb[k] / n; mb[k] = 0.9 * mb[k] + 0.1 * g; vb[k] = 0.999 * vb[k] + 0.001 * g * g;
        b[k] -= lr * (mb[k] / c1) / (Math.sqrt(vb[k] / c2) + 1e-8);
      }
    }
  }
  return { mu, sd, W, b };
}
function probs(m, i) {
  const z = new Float64Array(K); let mx = -Infinity;
  for (let k = 0; k < K; k++) { let v = m.b[k]; for (let j = 0; j < D; j++) v += m.W[k * D + j] * (X[i * D + j] - m.mu[j]) / m.sd[j]; z[k] = v; if (v > mx) mx = v; }
  let se = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); se += z[k]; }
  return Array.from(z, v => v / se);
}

const samples = [...new Set(rows.map(r => r.sample))];
for (let i = samples.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [samples[i], samples[j]] = [samples[j], samples[i]]; }
const fold = new Map(samples.map((s, i) => [s, i % 5]));

function cv(l2) {
  const conf = Array.from({ length: K }, () => new Array(K).fill(0)); let nll = 0;
  for (let f = 0; f < 5; f++) {
    const tr = [], te = [];
    rows.forEach((r, i) => (fold.get(r.sample) === f ? te : tr).push(i));
    const m = fit(tr, l2);
    for (const i of te) { const p = probs(m, i); const k = p.indexOf(Math.max(...p)); conf[Y[i]][k]++; nll -= Math.log(p[Y[i]] + 1e-9); }
  }
  let ok = 0, near = 0;
  conf.forEach((row, t) => row.forEach((c, p) => { if (t === p) ok += c; if (Math.abs(t - p) <= 1) near += c; }));
  const recall = conf.map((row, t) => row[t] / row.reduce((a, b) => a + b));
  return { exact: ok / N, within1: near / N, nll: nll / N, recall, conf };
}

if (mode === 'cv') {
  for (const l2 of (l2arg ? [+l2arg] : [1e-3, 1e-2, 3e-2])) {
    const r = cv(l2);
    console.log(`${spec} D=${D} l2=${l2}: exact ${(100 * r.exact).toFixed(1)}%  ±1 ${(100 * r.within1).toFixed(1)}%  logloss ${r.nll.toFixed(3)}  recall/stage ${r.recall.map(v => (100 * v).toFixed(0)).join(' ')}`);
    if (l2arg) r.conf.forEach((c, i) => console.log('   ', i + 1, c.join('\t')));
  }
} else if (mode === 'save') {
  const l2 = +l2arg, r = cv(l2);
  const m = fit(rows.map((_, i) => i), l2);
  const r5 = a => Array.from(a, v => +v.toFixed(5));
  fs.writeFileSync('../docs/head.json', JSON.stringify({
    blocks: blocks.map(b => ({ name: b.name, dim: b.dim, modelId: b.modelId, dtype: b.dtype })),
    colorFeatures: colorNames, mu: r5(m.mu), sd: r5(m.sd), W: r5(m.W), b: r5(m.b),
    cv: { exact: +r.exact.toFixed(3), within1: +r.within1.toFixed(3), recall: r.recall.map(v => +v.toFixed(3)), confusion: r.conf, split: '5-fold, grouped by avocado' },
  }));
  console.log('saved', spec, 'exact', r.exact.toFixed(3), '±1', r.within1.toFixed(3), 'recall', r.recall.map(v => v.toFixed(2)).join(' '));
}
