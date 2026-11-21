// App pipeline (locateFruit crop + DINOv2 + colour head) on dataset photos under different conditions.
// Model was trained on all photos, so absolute numbers are optimistic; compare conditions.
import { pipeline, RawImage } from '@huggingface/transformers';
import sharp from 'sharp'; import fs from 'node:fs';
import { extractFeatures, locateFruit, backgroundGains, applyGains, SIZE } from '../docs/features.js';
import { predictHead, embeddingFromTokens } from '../docs/model.js';
const dir = process.argv[2];
const head = JSON.parse(fs.readFileSync('../docs/head.json'));
const ex = await pipeline('image-feature-extraction', head.blocks[0].modelId, { dtype: head.blocks[0].dtype });
const { rows } = JSON.parse(fs.readFileSync('data/features.json'));
const pick = rows.filter(r => +r.sample.split('-')[1] % 14 === 0);
const conds = { 'full frame (as trained)': [false, [1, 1, 1], false], 'auto-crop + WB': [true, [1, 1, 1], true], 'auto-crop + warm light + WB': [true, [1.12, 1, 0.78], true], 'auto-crop + cool light + WB': [true, [0.88, 1, 1.15], true] };
const res = Object.fromEntries(Object.keys(conds).map(k => [k, { ok: 0, near: 0, n: 0, s5: 0, n5: 0, crop: 0 }]));
for (const r of pick) {
  const base = await sharp(`${dir}/${r.file}.jpg`).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (const [k, [auto, tint, wb]] of Object.entries(conds)) {
    const W = base.info.width, H = base.info.height, px = Uint8ClampedArray.from(base.data);
    for (let i = 0; i < px.length; i += 4) for (let c = 0; c < 3; c++) px[i + c] = px[i + c] * tint[c];
    let x = 0, y = 0, side = Math.min(W, H);
    if (auto) {
      const s = 256 / Math.max(W, H);
      const sm = await sharp(Buffer.from(px.buffer), { raw: { width: W, height: H, channels: 4 } }).resize(Math.round(W * s), Math.round(H * s)).raw().toBuffer();
      const b = locateFruit(sm, Math.round(W * s), Math.round(H * s));
      if (b) { x = b.x / s; y = b.y / s; side = b.side / s; res[k].crop++; }
    }
    const img = sharp(Buffer.from(px.buffer), { raw: { width: W, height: H, channels: 4 } })
      .extract({ left: Math.round(x), top: Math.round(y), width: Math.round(side), height: Math.round(side) });
    const c448 = await img.clone().resize(448, 448).raw().toBuffer();
    if (wb) { const g = backgroundGains(c448, 448 * 448); if (g) applyGains(c448, g); }
    const c128 = await img.clone().resize(SIZE, SIZE).raw().toBuffer();
    const out = await ex(new RawImage(new Uint8ClampedArray(c448), 448, 448, 4).rgb());
    const f = extractFeatures(c128, null);
    const p = predictHead(head, embeddingFromTokens(out.data, out.dims), f.features).stage;
    const s = res[k]; s.n++; if (p === r.stage) s.ok++; if (Math.abs(p - r.stage) <= 1) s.near++;
    if (r.stage === 5) { s.n5++; if (p === 5) s.s5++; }
  }
}
console.log('photos', pick.length);
for (const [k, s] of Object.entries(res)) console.log(k.padEnd(26), `exact ${(100 * s.ok / s.n).toFixed(1)}%  ±1 ${(100 * s.near / s.n).toFixed(1)}%  stage5 recall ${(100 * s.s5 / s.n5).toFixed(0)}%  fruit found ${s.crop}`);
