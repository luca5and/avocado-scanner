import { extractFeatures, ellipseRegion, locateFruit, backgroundGains, applyGains, SIZE } from './features.js';
import { predict, predictHead, embeddingFromTokens, outlook, STAGES } from './model.js';

// The oval guide on screen, as a fraction of the square frame. Must match the SVG in index.html.
// Sized so the fruit fills the frame about as much as in the dataset photos.
const OVAL = { rx: 0.19, ry: 0.29 };
const region = ellipseRegion(SIZE, OVAL.rx, OVAL.ry);
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';

const $ = id => document.getElementById(id);
const views = ['home', 'camera', 'result'];
const show = v => views.forEach(n => ($(n).hidden = n !== v));

let colorModel, head, shelf, stream, lastProbs;
const ready = Promise.all([
  fetch('model.json').then(r => r.json()),
  fetch('head.json').then(r => r.json()),
  fetch('shelf-life.json').then(r => r.json()),
]).then(([m, h, s]) => {
  colorModel = m; head = h; shelf = s;
  $('accuracy').textContent = `${Math.round(h.cv.exact * 100)}%`;
  $('accuracy1').textContent = `${Math.round(h.cv.within1 * 100)}%`;
  $('accuracy5').textContent = `${Math.round(h.cv.recall[4] * 100)}%`;
});

// ---------- image network (DINOv2), downloaded once and cached by the browser ----------
// Downloaded only when someone starts a scan (or straight away if it is already cached),
// so visitors who just look at the page don't spend 25 MB of data.
const netStatus = t => document.querySelectorAll('.net-status').forEach(el => { el.textContent = t; el.hidden = !t; });
let netPromise = null;
function loadNet() {
  netPromise ??= ready.then(async () => {
    $('first-download').hidden = true;
    // Transformers.js reports "progress" while reading from the cache too, so check first
    // which message is true.
    const cached = await netIsCached();
    const { pipeline } = await import(TRANSFORMERS);
    const files = {};
    const extractor = await pipeline('image-feature-extraction', head.blocks[0].modelId, {
      dtype: head.blocks[0].dtype,
      device: 'wasm',
      progress_callback: e => {
        if (e.status !== 'progress' || !e.total) return;
        files[e.file] = [e.loaded, e.total];
        const [l, t] = Object.values(files).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
        if (t > 1e6 && l < t) netStatus(cached
          ? 'Loading the image model from this phone…'
          : `Downloading the image model (first time only)… ${Math.round(l / 1e6)} of ${Math.round(t / 1e6)} MB`);
      },
    });
    netStatus('');
    // Ask the browser not to clear the saved model when space is low (granted silently or ignored).
    navigator.storage?.persist?.().catch(() => {});
    return extractor;
  }).catch(e => {
    console.error('image model failed to load', e);
    netStatus('Image model unavailable – using the simpler colour model.');
    return null;
  });
  return netPromise;
}

/** Is the model file already in the browser's cache from an earlier visit? */
async function netIsCached() {
  try {
    await ready;
    const cache = await caches.open('transformers-cache');
    const keys = await cache.keys();
    return keys.some(k => k.url.includes(head.blocks[0].modelId) && k.url.endsWith('.onnx'));
  } catch { return false; }
}

// ---------- camera ----------
async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return $('file').click();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1920 } },
      audio: false,
    });
  } catch (e) {
    console.warn('camera unavailable, falling back to photo picker', e);
    return $('file').click();
  }
  const v = $('video');
  v.srcObject = stream;
  await v.play();
  show('camera');
}
function closeCamera() {
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
}

function capture() {
  const v = $('video');
  // freeze the frame before stopping the camera
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  closeCamera();
  analyse(c, c.width, c.height);
}

async function fromUrl(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  await analyse(img, img.naturalWidth, img.naturalHeight);
}
async function fromFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try { await fromUrl(url); } finally { URL.revokeObjectURL(url); }
}

// Sample photos from the dataset, for trying the app without an avocado at hand.
const SAMPLES = ['T20_d01_001_a_1', 'T20_d05_001_a_3', 'T20_d07_001_a_4'];
let sampleIdx = 0;
function trySample(i = sampleIdx++ % SAMPLES.length) {
  fromUrl(`samples/${SAMPLES[i]}.jpg`);
}

// ---------- analysis ----------
function crop(source, sx, sy, side, size) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  return { canvas: c, data: ctx.getImageData(0, 0, size, size).data };
}

async function analyse(source, w, h) {
  await ready;
  show('result');
  window.scrollTo(0, 0);
  $('busy').hidden = false; $('verdict').hidden = true; $('error').hidden = true;

  // 1) find the avocado on a light background and frame it like the dataset photos;
  //    otherwise fall back to the centre square + oval guide
  const k = 256 / Math.max(w, h), sw = Math.round(w * k), sh = Math.round(h * k);
  const small = document.createElement('canvas'); small.width = sw; small.height = sh;
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(source, 0, 0, sw, sh);
  // ?raw (testing only): use the whole square photo, exactly like training
  const box = RAW ? { x: 0, y: 0, side: Math.min(sw, sh) } : locateFruit(sctx.getImageData(0, 0, sw, sh).data, sw, sh);
  let sx, sy, side, reg;
  if (box) { sx = box.x / k; sy = box.y / k; side = box.side / k; reg = null; }
  else { side = Math.min(w, h); sx = (w - side) / 2; sy = (h - side) / 2; reg = region; }

  // 2) picture for the user
  const big = $('shot'), D = 480;
  big.width = big.height = D;
  const bctx = big.getContext('2d');
  bctx.drawImage(source, sx, sy, side, side, 0, 0, D, D);

  // 3) colour features
  const f = extractFeatures(crop(source, sx, sy, side, SIZE).data, reg);
  drawOverlay(bctx, D, f?.mask, !reg);
  if (!f) { $('busy').hidden = true; $('error').hidden = false; return; }

  // 4) image network + combined classifier (falls back to colour-only)
  let r, usedNet = false;
  const extractor = await loadNet();
  if (extractor) {
    try {
      const { RawImage } = await import(TRANSFORMERS);
      const { data } = crop(source, sx, sy, side, 448);
      // make the paper look like the dataset's lightbox, so the network sees familiar colours
      const gains = backgroundGains(data, 448 * 448);
      if (gains) applyGains(data, gains);
      const out = await extractor(new RawImage(new Uint8ClampedArray(data), 448, 448, 4).rgb());
      r = predictHead(head, embeddingFromTokens(out.data, out.dims), f.features);
      usedNet = true;
    } catch (e) { console.error('image model failed', e); }
  }
  if (!r) r = predict(colorModel, f.features);
  window.__lastResult = { ...r, usedNet, autoCrop: !reg }; // for testing

  $('busy').hidden = true; $('verdict').hidden = false;
  lastProbs = r.probs;
  const s = STAGES[r.stage - 1];
  $('stage-num').textContent = r.stage;
  $('stage-name').textContent = s.name;
  $('stage-advice').textContent = s.advice;
  $('stage-detail').textContent = s.detail;
  $('result').dataset.stage = r.stage;
  $('confidence').textContent = r.confidence >= 0.6
    ? `Fairly confident (${pct(r.confidence)})`
    : `Unsure – between stage ${Math.min(r.stage, r.runnerUp)} and ${Math.max(r.stage, r.runnerUp)} (${pct(r.confidence)} for stage ${r.stage})`;
  // stages 4 and 5 often look the same from outside: suggest a hands-on check
  $('check45').hidden = !(r.probs[3] + r.probs[4] > 0.5 && Math.min(r.probs[3], r.probs[4]) > 0.15);
  $('wb-note').hidden = f.balanced;
  $('colour-only').hidden = usedNet;

  const bars = $('bars');
  bars.innerHTML = '';
  r.probs.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'bar' + (i === r.stage - 1 ? ' top' : '');
    row.innerHTML = `<span class="lbl">${i + 1} · ${STAGES[i].short}</span>
      <span class="track"><span class="fill s${i + 1}" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="val">${pct(p)}</span>`;
    bars.append(row);
  });
  renderOutlook();
}

function drawOverlay(ctx, D, mask, autoCrop) {
  // dim everything the colour analysis ignored; show the oval when it was used
  if (mask) {
    const m = document.createElement('canvas');
    m.width = m.height = SIZE;
    const mctx = m.getContext('2d'), img = mctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) img.data[4 * i + 3] = mask[i] ? 0 : 130;
    mctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(m, 0, 0, D, D);
  }
  if (autoCrop) return;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 8]);
  ctx.beginPath();
  ctx.ellipse(D / 2, D / 2, OVAL.rx * D, OVAL.ry * D, 0, 0, Math.PI * 2);
  ctx.stroke();
}

const pct = p => `${Math.round(p * 100)}%`;
const days = d => {
  const r = Math.round(d.median);
  if (d.high < 0.5) return 'now';
  return r <= 0 ? 'under a day' : `${r} day${r === 1 ? '' : 's'}`;
};
const range = d => {
  if (d.high < 0.5) return '';
  const lo = Math.round(d.low), hi = Math.round(d.high);
  return lo === hi ? `usually ${lo} day${lo === 1 ? '' : 's'}` : `typically ${lo}–${hi} days`;
};

function renderOutlook() {
  if (!lastProbs) return;
  const group = document.querySelector('input[name=storage]:checked').value;
  const o = outlook(shelf, lastProbs, group);
  const readyNow = o.toRipe.high < 0.5;
  $('to-ripe').textContent = readyNow ? 'Ready now' : days(o.toRipe);
  $('to-ripe-range').textContent = readyNow ? 'It has reached the ripe stage' : range(o.toRipe);
  $('to-over').textContent = o.toOverripe.high < 0.5 ? 'Already past' : days(o.toOverripe);
  $('to-over-range').textContent = range(o.toOverripe);
}

// ---------- wiring ----------
// start the model download as soon as someone shows they want to scan; it continues while they frame the shot
for (const id of ['scan', 'upload', 'gallery', 'sample']) $(id).addEventListener('click', loadNet);
$('scan').addEventListener('click', openCamera);
$('again').addEventListener('click', openCamera);
$('shutter').addEventListener('click', capture);
$('cancel').addEventListener('click', () => { closeCamera(); show('home'); });
// Never leave the camera running in the background: switch it off when the page is hidden.
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) { closeCamera(); show('home'); } });
$('upload').addEventListener('click', () => $('file').click());
$('upload2').addEventListener('click', () => { closeCamera(); $('file').click(); });
for (const id of ['file', 'file-gallery']) $(id).addEventListener('change', e => { fromFile(e.target.files[0]); e.target.value = ''; });
$('gallery').addEventListener('click', () => $('file-gallery').click());
$('sample').addEventListener('click', () => trySample());
document.querySelectorAll('input[name=storage]').forEach(r => r.addEventListener('change', renderOutlook));
show('home');
const qs = new URLSearchParams(location.search), RAW = qs.has('raw');
if (qs.has('sample')) trySample(+qs.get('sample') || 0);
// already downloaded on an earlier visit? then get it ready now (no data used); otherwise explain the first download
netIsCached().then(cached => { if (cached) loadNet(); else if (!netPromise) $('first-download').hidden = false; });
