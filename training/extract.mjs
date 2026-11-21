// Extract skin features for every dataset photo -> data/features.json
import sharp from 'sharp';
import ExcelJS from 'exceljs';
import fs from 'node:fs';
import { extractFeatures, SIZE, FEATURE_NAMES } from '../docs/features.js';
const imgDir = process.argv[2];
const wb = new ExcelJS.Workbook();
await wb.xlsx.readFile('data/ripening.xlsx');
const rows = [];
wb.getWorksheet('DATABASE').eachRow((r, i) => {
  if (i === 1) return;
  const [, file, , group, sample, day, stage] = r.values;
  if (file) rows.push({ file: String(file), group, sample: `${group}-${sample}`, day: +day, stage: +stage });
});
const out = []; let skipped = 0, done = 0;
const t0 = Date.now();
async function work(r) {
  const p = `${imgDir}/${r.file}.jpg`;
  if (!fs.existsSync(p)) { skipped++; return; }
  const { data } = await sharp(p).resize(SIZE, SIZE, { fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const f = extractFeatures(data, null);
  if (!f) { skipped++; return; }
  out.push({ ...r, x: f.features.map(v => +v.toFixed(5)) });
  if (++done % 2000 === 0) console.log(done, ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
const queue = [...rows];
await Promise.all(Array.from({ length: 8 }, async () => { while (queue.length) await work(queue.shift()); }));
fs.writeFileSync('data/features.json', JSON.stringify({ names: FEATURE_NAMES, rows: out }));
console.log('done', out.length, 'skipped', skipped);
