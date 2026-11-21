// Colour/texture features of an avocado's skin.
// Shared by the browser app and the Node training script, so both see the same numbers.
//
// Input: RGBA pixels of a SIZE x SIZE image, and a "region" mask (1 = may be fruit).
// The light background (paper / lightbox) is found with Otsu's threshold, dropped from the region,
// and used to correct white balance and exposure.

export const SIZE = 128;

function srgbToLinear(c) {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
const LIN = new Float32Array(256).map((_, i) => srgbToLinear(i));

function labFromLinear(r, g, b) {
  // linear sRGB -> XYZ (D65) -> Lab
  let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
  let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  x = f(x); y = f(y); z = f(z);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

/** Ellipse region used by the app's camera guide (fraction of the square frame). */
export function ellipseRegion(size = SIZE, rx = 0.36, ry = 0.44) {
  const m = new Uint8Array(size * size);
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const dx = (x - c) / (rx * size), dy = (y - c) / (ry * size);
      m[y * size + x] = dx * dx + dy * dy <= 1 ? 1 : 0;
    }
  return m;
}

export const FEATURE_NAMES = [];

/**
 * @param {Uint8Array|Uint8ClampedArray} rgba  SIZE*SIZE*4 pixels
 * @param {Uint8Array|null} region  SIZE*SIZE, 1 where fruit may be (null = whole image)
 * @returns {{features:number[], fruitPixels:number, balanced:boolean, mask:Uint8Array}|null}
 */
export function extractFeatures(rgba, region = null) {
  const n = SIZE * SIZE;
  // 1) split light background (paper / lightbox) from the darker fruit with Otsu's threshold
  //    on perceptual lightness. Works under any tint or exposure, unlike a fixed "white" test.
  const Y = new Float32Array(n);
  const BINS = 64, hist = new Float64Array(BINS);
  for (let i = 0; i < n; i++) {
    const y = Math.sqrt(0.2126 * LIN[rgba[4 * i]] + 0.7152 * LIN[rgba[4 * i + 1]] + 0.0722 * LIN[rgba[4 * i + 2]]);
    Y[i] = y; hist[Math.min(BINS - 1, Math.floor(y * BINS))]++;
  }
  let sumAll = 0; for (let k = 0; k < BINS; k++) sumAll += k * hist[k];
  let w0 = 0, s0 = 0, bestVar = -1, tBin = 0;
  for (let k = 0; k < BINS; k++) {
    w0 += hist[k]; s0 += k * hist[k];
    const w1 = n - w0; if (!w0 || !w1) continue;
    const v = w0 * w1 * (s0 / w0 - (sumAll - s0) / w1) ** 2;
    if (v > bestVar) { bestVar = v; tBin = k; }
  }
  const thr = (tBin + 1) / BINS;
  let nBright = 0, outside = 0, yBright = 0, yDark = 0, nDark = 0;
  for (let i = 0; i < n; i++) {
    if (Y[i] >= thr) { nBright++; yBright += Y[i]; if (region && !region[i]) outside++; }
    else { nDark++; yDark += Y[i]; }
  }
  // Accept the light class as background only if it is big, clearly lighter than the rest,
  // and (in the app) mostly outside the oval, i.e. it is the paper and not the fruit.
  const hasBg = nBright > n * 0.08 && nDark > n * 0.04 &&
    yBright / nBright > 1.5 * (yDark / nDark) && (!region || outside > nBright * 0.5);

  // 2) white balance + exposure: scale so the background averages a neutral light grey (~ the lightbox)
  let kr = 1, kg = 1, kb = 1, balanced = false;
  if (hasBg) {
    let br = 0, bg = 0, bb = 0, nb = 0;
    for (let i = 0; i < n; i++) {
      if (Y[i] < thr) continue;
      const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
      if (Math.max(r, g, b) >= 250) continue; // clipped
      br += LIN[r]; bg += LIN[g]; bb += LIN[b]; nb++;
    }
    if (nb > n * 0.03) {
      const target = 0.75;
      kr = target / (br / nb); kg = target / (bg / nb); kb = target / (bb / nb);
      balanced = true;
    }
  }

  // 3) collect fruit pixels in Lab
  const L = [], A = [], B = [];
  const fruit = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (region && !region[i]) continue;
    if (hasBg && Y[i] >= thr) continue; // background
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    const rl = Math.min(1, LIN[r] * kr), gl = Math.min(1, LIN[g] * kg), bl = Math.min(1, LIN[b] * kb);
    const [l, a, bb2] = labFromLinear(rl, gl, bl);
    if (l > 85) continue; // specular highlight
    L.push(l); A.push(a); B.push(bb2); fruit[i] = 1;
  }
  const m = L.length;
  if (m < n * 0.04) return null; // not enough fruit visible

  const mean = v => v.reduce((s, x) => s + x, 0) / v.length;
  const std = (v, mu) => Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / v.length);
  const quant = (v, ps) => { const s = Float32Array.from(v).sort(); return ps.map(p => s[Math.floor(p * (s.length - 1))]); };

  const mL = mean(L), mA = mean(A), mB = mean(B);
  const C = A.map((a, i) => Math.hypot(a, B[i]));
  const H = A.map((a, i) => Math.atan2(B[i], a)); // hue angle, radians

  // hue histogram (8 bins) weighted by chroma, normalised
  const hb = new Array(8).fill(0); let hw = 0;
  for (let i = 0; i < m; i++) { const k = Math.floor(((H[i] + Math.PI) / (2 * Math.PI)) * 8) % 8; hb[k] += C[i]; hw += C[i]; }
  for (let k = 0; k < 8; k++) hb[k] /= hw || 1;

  const frac = pred => { let c = 0; for (let i = 0; i < m; i++) if (pred(i)) c++; return c / m; };

  // texture: mean |dL| between neighbouring fruit pixels (spots / wrinkles)
  let tex = 0, tn = 0;
  const Lmap = new Float32Array(n); { let j = 0; for (let i = 0; i < n; i++) if (fruit[i]) Lmap[i] = L[j++]; }
  for (let y = 0; y < SIZE - 1; y++)
    for (let x = 0; x < SIZE - 1; x++) {
      const i = y * SIZE + x;
      if (fruit[i] && fruit[i + 1] && fruit[i + SIZE]) { tex += Math.abs(Lmap[i] - Lmap[i + 1]) + Math.abs(Lmap[i] - Lmap[i + SIZE]); tn++; }
    }
  tex = tn ? tex / tn : 0;

  const [l10, l50, l90] = quant(L, [0.1, 0.5, 0.9]);
  const [a10, a50, a90] = quant(A, [0.1, 0.5, 0.9]);
  const [b10, b50, b90] = quant(B, [0.1, 0.5, 0.9]);
  const mC = mean(C);

  const f = {
    mL, mA, mB, sL: std(L, mL), sA: std(A, mA), sB: std(B, mB),
    l10, l50, l90, a10, a50, a90, b10, b50, b90,
    mC, sC: std(C, mC),
    green: frac(i => A[i] < -4 && B[i] > 5),
    purple: frac(i => A[i] > 2),
    dark: frac(i => L[i] < 25),
    veryDark: frac(i => L[i] < 15),
    brown: frac(i => A[i] > 0 && B[i] > 8),
    tex,
    ...Object.fromEntries(hb.map((v, k) => [`h${k}`, v])),
  };
  if (!FEATURE_NAMES.length) FEATURE_NAMES.push(...Object.keys(f));
  return { features: Object.values(f), fruitPixels: m, balanced, mask: fruit };
}
