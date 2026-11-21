// Checks the model is only downloaded on scan intent, and loaded from cache on later visits.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const base = process.argv[2] ?? 'http://localhost:8123/';
const prof = process.argv[3];
fs.rmSync(prof, { recursive: true, force: true });
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, userDataDir: prof });
const page = await browser.newPage();
let onnx = 0, onnxBytes = 0;
page.on('response', r => { const len = +(r.headers()['content-length'] ?? 0); if (len > 1e6 && !r.fromCache()) { onnx++; onnxBytes += len; } });
page.on('pageerror', e => console.log('  [pageerror]', e.message));
const state = () => page.evaluate(() => ({
  note: !document.getElementById('first-download').hidden,
  status: [...document.querySelectorAll('.net-status')].map(e => e.textContent).find(Boolean) ?? '',
  result: window.__lastResult ? { stage: window.__lastResult.stage, net: window.__lastResult.usedNet } : null,
}));
await page.goto(base); await new Promise(r => setTimeout(r, 6000));
console.log('1) first visit, idle 6s:', JSON.stringify(await state()), 'big downloads', onnx, (onnxBytes / 1e6).toFixed(1) + 'MB');
await page.click('#sample');
const t0 = Date.now(); let seen = '';
while (!(await state()).result && Date.now() - t0 < 300000) { const s = await state(); if (s.status && s.status !== seen) seen = s.status; await new Promise(r => setTimeout(r, 300)); }
console.log('2) after tapping sample:', JSON.stringify(await state()), 'big downloads', onnx, (onnxBytes / 1e6).toFixed(1) + 'MB', `(${((Date.now() - t0) / 1000).toFixed(1)}s)`, 'progress seen:', JSON.stringify(seen));
onnx = 0;
await page.goto(base); await new Promise(r => setTimeout(r, 6000));
console.log('3) second visit, idle 6s:', JSON.stringify(await state()), 'big downloads', onnx);
const t1 = Date.now(); await page.click('#sample');
while (!(await state()).result && Date.now() - t1 < 60000) await new Promise(r => setTimeout(r, 100));
console.log('4) scan on second visit:', JSON.stringify(await state()), `(${((Date.now() - t1) / 1000).toFixed(1)}s)`);
await browser.close();
