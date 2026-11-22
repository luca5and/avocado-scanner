// Opening the site with "#privacy" in the address (as Safari reopens it) must start at the top,
// and tapping the privacy badge must scroll there without changing the address.
import puppeteer from 'puppeteer-core';
const base = process.argv[2] ?? 'http://localhost:8123/';
const b = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const p = await b.newPage(); await p.setViewport({ width: 390, height: 844 });
await p.goto(base + '#privacy'); await new Promise(r => setTimeout(r, 1000));
console.log('reopen with #privacy -> scrollY', await p.evaluate(() => scrollY), 'url', p.url());
await p.click('.lock'); await new Promise(r => setTimeout(r, 1200));
console.log('tap badge -> scrollY', await p.evaluate(() => scrollY), 'url', p.url(),
  'privacy heading top', await p.evaluate(() => Math.round(document.getElementById('privacy').getBoundingClientRect().top)));
await p.reload(); await new Promise(r => setTimeout(r, 1000));
console.log('reload -> scrollY', await p.evaluate(() => scrollY));
await b.close();
