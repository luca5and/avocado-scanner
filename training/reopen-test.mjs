// Close the browser completely, reopen with the same profile: is the model downloaded again? What does the status line say?
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const base = process.argv[2] ?? 'http://localhost:8123/';
const prof = process.argv[3];
fs.rmSync(prof, { recursive: true, force: true });
async function visit(label, tap) {
  const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, userDataDir: prof });
  const page = await browser.newPage();
  let big = 0;
  page.on('response', r => { const len = +(r.headers()['content-length'] ?? 0); if (len > 1e6 && !r.fromCache()) big += len; });
  await page.evaluateOnNewDocument(() => {
    window.__statuses = [];
    new MutationObserver(() => document.querySelectorAll('.net-status').forEach(e => { const t = e.textContent; if (t && window.__statuses.at(-1) !== t) window.__statuses.push(t); }))
      .observe(document, { subtree: true, childList: true, characterData: true });
  });
  await page.goto(base);
  if (tap) await page.click('#sample');
  const t0 = Date.now();
  while (!(await page.evaluate(() => window.__lastResult?.usedNet !== undefined || false)) && Date.now() - t0 < 120000) await new Promise(r => setTimeout(r, 200));
  const st = await page.evaluate(() => [...new Set(window.__statuses.map(s => s.replace(/\d+ of/, 'N of')))]);
  const used = await page.evaluate(() => window.__lastResult?.usedNet);
  console.log(label, 'downloaded', (big / 1e6).toFixed(1), 'MB; usedNet', used, '; status texts:', JSON.stringify(st));
  await browser.close();
}
await visit('1) first visit + scan:', true);
await visit('2) browser reopened + scan:', true);
