// Node reference prediction for docs/samples/*.jpg using the full square image (compare with ?raw in the browser).
import { pipeline, RawImage } from '@huggingface/transformers';
import sharp from 'sharp'; import fs from 'node:fs';
import { extractFeatures, SIZE } from '../docs/features.js';
import { predictHead, embeddingFromTokens } from '../docs/model.js';
const head = JSON.parse(fs.readFileSync('../docs/head.json'));
const ex = await pipeline('image-feature-extraction', head.blocks[0].modelId, { dtype: head.blocks[0].dtype });
for (const f of ['T20_d01_001_a_1', 'T20_d05_001_a_3', 'T20_d07_001_a_4']) {
  const p = `../docs/samples/${f}.jpg`;
  const out = await ex(await RawImage.read(p));
  const { data } = await sharp(p).resize(SIZE, SIZE, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const r = predictHead(head, embeddingFromTokens(out.data, out.dims), extractFeatures(data, null).features);
  console.log(f, 'node', r.stage, r.probs.map(v => v.toFixed(2)).join(' '));
}
