// Builds ../docs/shelf-life.json from the dataset spreadsheet.
// For every photo of a sample that was followed until it reached a target stage,
// "days left" = first day the sample reached that stage minus the photo's day.
import ExcelJS from 'exceljs';
import fs from 'node:fs';
const wb = new ExcelJS.Workbook();
await wb.xlsx.readFile('data/ripening.xlsx');
const ws = wb.getWorksheet('DATABASE');
const rows = [];
ws.eachRow((row, i) => {
  if (i === 1) return;
  const [, file, , group, sample, day, stage] = row.values;
  if (!file || !group) return;
  rows.push({ file: String(file), group, sample: `${group}-${sample}`, day: +day, stage: +stage });
});
console.log('rows', rows.length);
const bySample = new Map();
for (const r of rows) (bySample.get(r.sample) ?? bySample.set(r.sample, []).get(r.sample)).push(r);
const firstDay = (list, s) => Math.min(...list.filter(r => r.stage >= s).map(r => r.day));
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * p; const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };
const out = { source: "Xavier, Rodrigues & Silva (2024), 'Hass' Avocado Ripening Photographic Dataset, Mendeley Data, doi:10.17632/3xd9n945v8.1, CC BY 4.0", groups: {} };
for (const group of ['T10', 'T20', 'Tam']) {
  const samples = [...bySample.values()].filter(l => l[0].group === group);
  const g = { samples: samples.length, stages: {} };
  for (let stage = 1; stage <= 5; stage++) {
    const res = {};
    for (const [key, target] of [['toRipe', 3], ['toOverripe', 5]]) {
      if (stage >= target) continue;
      const d = [];
      for (const l of samples) {
        const f = firstDay(l, target);
        if (!isFinite(f)) continue; // never reached target while observed (censored)
        // one value per sample per stage: use the first day the sample was seen at this stage
        const seen = l.filter(r => r.stage === stage && r.day < f);
        if (seen.length) d.push(f - Math.min(...seen.map(r => r.day)));
      }
      if (d.length >= 5) res[key] = { n: d.length, median: q(d, .5), p25: q(d, .25), p75: q(d, .75) };
    }
    const n = samples.filter(l => l.some(r => r.stage === stage)).length;
    g.stages[stage] = { samplesSeen: n, ...res };
  }
  out.groups[group] = g;
}
fs.mkdirSync('../site', { recursive: true });
fs.writeFileSync('../docs/shelf-life.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.groups, null, 0).replace(/\},"/g, '},\n"'));
// stage distribution + days range per group
for (const group of ['T10','T20','Tam']) { const c=[0,0,0,0,0]; rows.filter(r=>r.group===group).forEach(r=>c[r.stage-1]++); console.log(group, 'photos per stage', c, 'max day', Math.max(...rows.filter(r=>r.group===group).map(r=>r.day))); }
