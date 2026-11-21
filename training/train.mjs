// Train a small classifier on skin features. Evaluation splits by avocado (sample),
// never by photo, so the same fruit is never in both train and test.
import fs from 'node:fs';
const { names, rows } = JSON.parse(fs.readFileSync('data/features.json'));
const K = 5, D = names.length;

function rng(seed) { return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32); }
const rand = rng(42);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

function fitScaler(X) {
  const mu = new Array(D).fill(0), sd = new Array(D).fill(0);
  for (const x of X) x.forEach((v, j) => (mu[j] += v / X.length));
  for (const x of X) x.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / X.length));
  return { mu, sd: sd.map(s => Math.sqrt(s) || 1) };
}
const scale = (x, s) => x.map((v, j) => (v - s.mu[j]) / s.sd[j]);

// MLP with one hidden layer (H=0 -> softmax regression)
function init(H) {
  const w = (r, c, s) => Array.from({ length: r }, () => Array.from({ length: c }, () => gauss() * s));
  return H ? { W1: w(H, D, Math.sqrt(2 / D)), b1: new Array(H).fill(0), W2: w(K, H, Math.sqrt(1 / H)), b2: new Array(K).fill(0) }
           : { W2: w(K, D, 0.01), b2: new Array(K).fill(0) };
}
export function forward(m, x) {
  let h = x, z1;
  if (m.W1) { z1 = m.W1.map((row, i) => row.reduce((s, w, j) => s + w * x[j], m.b1[i])); h = z1.map(v => Math.max(0, v)); }
  const z = m.W2.map((row, k) => row.reduce((s, w, j) => s + w * h[j], m.b2[k]));
  const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), se = e.reduce((a, b) => a + b);
  return { p: e.map(v => v / se), h, z1 };
}
function train(X, Y, H, { epochs = 60, lr = 0.003, l2 = 1e-4, bs = 64 } = {}) {
  const m = init(H);
  const params = Object.keys(m);
  const mom = {}, vel = {};
  const zeros = a => Array.isArray(a[0]) ? a.map(r => r.map(() => 0)) : a.map(() => 0);
  for (const k of params) { mom[k] = zeros(m[k]); vel[k] = zeros(m[k]); }
  let t = 0; const idx = X.map((_, i) => i);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let s = 0; s < idx.length; s += bs) {
      const g = {}; for (const k of params) g[k] = zeros(m[k]);
      const batch = idx.slice(s, s + bs);
      for (const i of batch) {
        const x = X[i], { p, h, z1 } = forward(m, x);
        const dz = p.map((v, k) => v - (k === Y[i] ? 1 : 0));
        for (let k = 0; k < K; k++) { g.b2[k] += dz[k]; for (let j = 0; j < h.length; j++) g.W2[k][j] += dz[k] * h[j]; }
        if (m.W1) for (let a = 0; a < m.W1.length; a++) {
          if (z1[a] <= 0) continue;
          let dh = 0; for (let k = 0; k < K; k++) dh += dz[k] * m.W2[k][a];
          g.b1[a] += dh; for (let j = 0; j < D; j++) g.W1[a][j] += dh * x[j];
        }
      }
      t++;
      const upd = (P, G, M, V, decay) => {
        for (let i = 0; i < P.length; i++) {
          if (Array.isArray(P[i])) { upd(P[i], G[i], M[i], V[i], decay); continue; }
          const gr = G[i] / batch.length + (decay ? l2 * P[i] : 0);
          M[i] = 0.9 * M[i] + 0.1 * gr; V[i] = 0.999 * V[i] + 0.001 * gr * gr;
          P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (Math.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8);
        }
      };
      for (const k of params) upd(m[k], g[k], mom[k], vel[k], k.startsWith('W'));
    }
  }
  return m;
}

// 5-fold CV grouped by sample
const samples = [...new Set(rows.map(r => r.sample))];
for (let i = samples.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [samples[i], samples[j]] = [samples[j], samples[i]]; }
const fold = new Map(samples.map((s, i) => [s, i % 5]));
function evaluate(H, opts) {
  const conf = Array.from({ length: K }, () => new Array(K).fill(0));
  let ok = 0, near = 0, nll = 0;
  for (let f = 0; f < 5; f++) {
    const tr = rows.filter(r => fold.get(r.sample) !== f), te = rows.filter(r => fold.get(r.sample) === f);
    const sc = fitScaler(tr.map(r => r.x));
    const m = train(tr.map(r => scale(r.x, sc)), tr.map(r => r.stage - 1), H, opts);
    for (const r of te) {
      const { p } = forward(m, scale(r.x, sc)); const pred = p.indexOf(Math.max(...p));
      conf[r.stage - 1][pred]++; if (pred === r.stage - 1) ok++; if (Math.abs(pred - (r.stage - 1)) <= 1) near++;
      nll -= Math.log(p[r.stage - 1] + 1e-9);
    }
  }
  return { acc: ok / rows.length, within1: near / rows.length, nll: nll / rows.length, conf };
}
const mode = process.argv[2] ?? 'cv';
if (mode === 'cv') {
  for (const [H, ep] of [[0, 40], [24, 40], [48, 60]]) {
    const r = evaluate(H, { epochs: ep });
    console.log(`hidden=${H}: exact ${(100 * r.acc).toFixed(1)}%  within±1 ${(100 * r.within1).toFixed(1)}%  logloss ${r.nll.toFixed(3)}`);
    console.log('  confusion (rows=true stage 1..5, cols=predicted):'); r.conf.forEach((c, i) => console.log('   ', i + 1, c.join('\t')));
  }
} else {
  const H = +mode;
  const sc = fitScaler(rows.map(r => r.x));
  const m = train(rows.map(r => scale(r.x, sc)), rows.map(r => r.stage - 1), H, { epochs: 60 });
  const round = a => Array.isArray(a) ? a.map(round) : +a.toFixed(5);
  const r = evaluate(H, { epochs: 60 });
  fs.writeFileSync('../docs/model.json', JSON.stringify({
    features: names, scaler: { mu: round(sc.mu), sd: round(sc.sd) },
    ...Object.fromEntries(Object.entries(m).map(([k, v]) => [k, round(v)])),
    cv: { exact: +r.acc.toFixed(3), within1: +r.within1.toFixed(3), confusion: r.conf, split: '5-fold, grouped by avocado' },
  }));
  console.log('saved model hidden=', H, 'cv exact', r.acc.toFixed(3), 'within1', r.within1.toFixed(3));
}
