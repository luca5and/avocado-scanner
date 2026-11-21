// Runs the app's exact pipeline (features.js + model.js, oval region) on dataset photos,
// under normal and simulated tinted lighting. Uses every 7th avocado to keep it quick.
// Note: the model was trained on all avocados, so absolute numbers here are optimistic;
// the point is to compare conditions.
import sharp from 'sharp';
import fs from 'node:fs';
import { extractFeatures, ellipseRegion, SIZE } from '../docs/features.js';
import { predict } from '../docs/model.js';
const dir = process.argv[2];
const model = JSON.parse(fs.readFileSync('docs/model.json'));
const { rows } = JSON.parse(fs.readFileSync('training/data/features.json'));
const pick = rows.filter(r => +r.sample.split('-')[1] % 7 === 0);
const oval = ellipseRegion(SIZE, 0.30, 0.42);
const conds = {
  'whole image (training setup)': [null, [1, 1, 1], 1],
  'app oval': [oval, [1, 1, 1], 1],
  'app oval, warm light': [oval, [1.12, 1.0, 0.78], 1],
  'app oval, cool light': [oval, [0.88, 1.0, 1.15], 1],
  'app oval, 25% darker': [oval, [1, 1, 1], 0.75],
};
const res = Object.fromEntries(Object.keys(conds).map(k => [k, { ok: 0, near: 0, n: 0, fail: 0 }]));
for (const r of pick) {
  const { data } = await sharp(`${dir}/${r.file}.jpg`).resize(SIZE, SIZE, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (const [k, [reg, tint, gain]] of Object.entries(conds)) {
    const px = Uint8ClampedArray.from(data);
    for (let i = 0; i < px.length; i += 4) for (let c = 0; c < 3; c++) px[i + c] = px[i + c] * tint[c] * gain;
    const f = extractFeatures(px, reg); const s = res[k]; s.n++;
    if (!f) { s.fail++; continue; }
    const p = predict(model, f.features).stage;
    if (p === r.stage) s.ok++; if (Math.abs(p - r.stage) <= 1) s.near++;
  }
}
console.log('photos', pick.length);
for (const [k, s] of Object.entries(res)) console.log(k.padEnd(30), `exact ${(100 * s.ok / s.n).toFixed(1)}%  within±1 ${(100 * s.near / s.n).toFixed(1)}%  not found ${s.fail}`);
