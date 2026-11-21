import { extractFeatures, ellipseRegion, SIZE } from './features.js';
import { predict, outlook, STAGES } from './model.js';

// The oval guide on screen, as a fraction of the square frame. Must match the SVG in index.html.
const OVAL = { rx: 0.30, ry: 0.42 };
const region = ellipseRegion(SIZE, OVAL.rx, OVAL.ry);

const $ = id => document.getElementById(id);
const views = ['home', 'camera', 'result'];
const show = v => views.forEach(n => ($(n).hidden = n !== v));

let model, shelf, stream, lastProbs;
const ready = Promise.all([
  fetch('model.json').then(r => r.json()),
  fetch('shelf-life.json').then(r => r.json()),
]).then(([m, s]) => {
  model = m; shelf = s;
  $('accuracy').textContent = `${Math.round(m.cv.exact * 100)}%`;
  $('accuracy1').textContent = `${Math.round(m.cv.within1 * 100)}%`;
});

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
  const w = v.videoWidth, h = v.videoHeight, side = Math.min(w, h);
  analyse(v, (w - side) / 2, (h - side) / 2, side);
  closeCamera();
}

async function fromUrl(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  const w = img.naturalWidth, h = img.naturalHeight, side = Math.min(w, h);
  await analyse(img, (w - side) / 2, (h - side) / 2, side);
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
async function analyse(source, sx, sy, side) {
  await ready;
  // small copy for the model
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, sx, sy, side, side, 0, 0, SIZE, SIZE);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;

  // large copy for the user to see
  const big = $('shot'), D = 480;
  big.width = big.height = D;
  const bctx = big.getContext('2d');
  bctx.drawImage(source, sx, sy, side, side, 0, 0, D, D);

  const f = extractFeatures(px, region);
  show('result');
  window.scrollTo(0, 0);
  if (!f) {
    $('verdict').hidden = true;
    $('error').hidden = false;
    drawOverlay(bctx, D, null);
    return;
  }
  $('error').hidden = true;
  $('verdict').hidden = false;
  drawOverlay(bctx, D, f.mask);

  const r = predict(model, f.features);
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
  $('wb-note').hidden = f.balanced;

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

function drawOverlay(ctx, D, mask) {
  // dim everything the model ignored, outline the oval
  if (mask) {
    const m = document.createElement('canvas');
    m.width = m.height = SIZE;
    const mctx = m.getContext('2d'), img = mctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) img.data[4 * i + 3] = mask[i] ? 0 : 150;
    mctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(m, 0, 0, D, D);
  }
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
const range = d => (d.high < 0.5 ? '' : `typically ${Math.round(d.low)}–${Math.round(d.high)} days`);

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
$('scan').addEventListener('click', openCamera);
$('again').addEventListener('click', openCamera);
$('shutter').addEventListener('click', capture);
$('cancel').addEventListener('click', () => { closeCamera(); show('home'); });
$('upload').addEventListener('click', () => $('file').click());
$('upload2').addEventListener('click', () => { closeCamera(); $('file').click(); });
for (const id of ['file', 'file-gallery']) $(id).addEventListener('change', e => { fromFile(e.target.files[0]); e.target.value = ''; });
$('gallery').addEventListener('click', () => $('file-gallery').click());
$('sample').addEventListener('click', () => trySample());
document.querySelectorAll('input[name=storage]').forEach(r => r.addEventListener('change', renderOutlook));
show('home');
const qs = new URLSearchParams(location.search);
if (qs.has('sample')) trySample(+qs.get('sample') || 0);
