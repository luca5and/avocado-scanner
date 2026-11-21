// Privacy audit: list every network request during a scan of a user photo, and check scanning works offline.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const [base = 'http://localhost:8123/', prof, photo] = process.argv.slice(2);
fs.rmSync(prof, { recursive: true, force: true });
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, userDataDir: prof });
const page = await browser.newPage();
const reqs = [];
page.on('request', r => reqs.push({ m: r.method(), url: r.url(), body: r.postData()?.length ?? 0, phase: phase }));
let phase = 'load';
const waitResult = async () => { const t = Date.now(); while (!(await page.evaluate(() => window.__lastResult)) && Date.now() - t < 120000) await new Promise(r => setTimeout(r, 200)); };
await page.goto(base);
phase = 'scan1';
await (await page.$('#file-gallery')).uploadFile(photo);
await waitResult();
console.log('scan 1 result', await page.evaluate(() => ({ stage: __lastResult.stage, net: __lastResult.usedNet })));
// second scan with the network cut ("airplane mode")
await page.evaluate(() => { window.__lastResult = null; });
await page.setOfflineMode(true);
phase = 'offline';
await page.evaluate(() => document.getElementById('again').click());
await (await page.$('#file-gallery')).uploadFile(photo);
await waitResult();
console.log('offline scan result', await page.evaluate(() => ({ stage: __lastResult?.stage, net: __lastResult?.usedNet })));
const hosts = {};
for (const r of reqs) { const k = `${r.phase} ${r.m} ${new URL(r.url).protocol === 'data:' || r.url.startsWith('blob:') ? r.url.slice(0, 5) : new URL(r.url).host}`; hosts[k] = (hosts[k] ?? 0) + 1; }
console.log(hosts);
console.log('non-GET requests:', reqs.filter(r => r.m !== 'GET').map(r => `${r.m} ${r.url} body=${r.body}`));
console.log('scan-phase URLs:', [...new Set(reqs.filter(r => r.phase !== 'load').map(r => r.url.slice(0, 110)))]);
await browser.close();
