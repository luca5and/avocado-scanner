// Drives the real site in Chrome, runs the 3 sample photos, prints predictions + timing.
// usage: node browser-test.mjs [baseUrl]
import puppeteer from 'puppeteer-core';
const base = process.argv[2] ?? 'http://localhost:8123/';
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 390, height: 1400, deviceScaleFactor: 1 });
page.on('console', m => { if (['error', 'warn'].includes(m.type())) console.log('  [console]', m.text().slice(0, 200)); });
page.on('pageerror', e => console.log('  [pageerror]', e.message));
for (const i of [0, 1, 2]) {
  const t0 = Date.now();
  await page.goto(`${base}?sample=${i}${process.env.RAW ? '&raw' : ''}`);
  await page.waitForFunction(() => window.__lastResult || !document.getElementById('error').hidden, { timeout: 300000 });
  const r = await page.evaluate(() => window.__lastResult);
  console.log(`sample ${i}: stage ${r?.stage} probs ${r?.probs.map(p => p.toFixed(2)).join(' ')} net=${r?.usedNet} autoCrop=${r?.autoCrop} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  if (i === 2 && process.env.SHOT) await page.screenshot({ path: process.env.SHOT, fullPage: true });
}
await browser.close();
