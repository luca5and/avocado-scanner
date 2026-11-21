// Compute image embeddings for dataset photos with a pretrained network (same library the browser uses).
// usage: node embed.mjs <modelId> <dtype> <imageDir> [limit]
import { pipeline, RawImage } from '@huggingface/transformers';
import fs from 'node:fs';
const [modelId, dtype, dir, limit] = process.argv.slice(2);
const { rows } = JSON.parse(fs.readFileSync('data/features.json'));
const list = limit ? rows.slice(0, +limit) : rows;
const extractor = await pipeline('image-feature-extraction', modelId, { dtype });
const tag = modelId.split('/')[1].split('.')[0] + '-' + dtype;
let dim = 0, buf, t0 = Date.now();
for (let i = 0; i < list.length; i++) {
  const img = await RawImage.read(`${dir}/${list[i].file}.jpg`);
  const out = await extractor(img);
  let v;
  if (out.dims.length === 3) { // ViT: [1, tokens, d] -> CLS + mean of patch tokens
    const [, T, d] = out.dims, a = out.data; v = new Float32Array(2 * d);
    for (let j = 0; j < d; j++) { v[j] = a[j]; let s = 0; for (let t = 1; t < T; t++) s += a[t * d + j]; v[d + j] = s / (T - 1); }
  } else v = Float32Array.from(out.data);
  if (!i) { dim = v.length; buf = new Float32Array(list.length * dim); console.log(tag, 'dims', out.dims, '->', dim); }
  buf.set(v, i * dim);
  if ((i + 1) % 1000 === 0 || i === list.length - 1) console.log(i + 1, ((Date.now() - t0) / (i + 1)).toFixed(0) + 'ms/img');
}
fs.writeFileSync(`data/emb-${tag}${limit ? '-test' : ''}.bin`, Buffer.from(buf.buffer));
fs.writeFileSync(`data/emb-${tag}${limit ? '-test' : ''}.json`, JSON.stringify({ modelId, dtype, dim, files: list.map(r => r.file) }));
