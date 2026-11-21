# Avocado Scanner 🥑

Point your phone at a Hass avocado and see its ripening stage (1–5) and roughly how many days it has
until it is ripe and until it is past its prime.

**Live site:** https://luca5and.github.io/avocado-scanner/

Everything runs in the browser; photos are never uploaded.

## How it works

1. **Find the avocado.** [`docs/features.js`](docs/features.js) separates the light background (white paper)
   from the fruit with an Otsu threshold and crops a square around the fruit, framed like the dataset photos
   (fruit ≈ 51% of the frame). Without a clear light background it falls back to the centre square + oval guide.
2. **Image network.** A pretrained [DINOv2-small](https://huggingface.co/onnx-community/dinov2-small) (8-bit, ~25 MB,
   run in the browser with [Transformers.js](https://github.com/huggingface/transformers.js)) turns the crop into
   768 numbers (CLS token + mean of patch tokens). Before that, the image is colour-corrected so the paper has the
   same tint as the dataset's lightbox.
3. **Colour features.** 31 skin measurements (CIELab statistics, hue histogram, dark/purple/green fractions, texture),
   white-balanced against the paper.
4. **Classifier.** A softmax classifier on both ([`docs/head.json`](docs/head.json), 47 KB) gives the stage
   probabilities, and [`docs/shelf-life.json`](docs/shelf-life.json) turns them into "days left".
   If the network can't load, the older colour-only model ([`docs/model.json`](docs/model.json)) is used.

### Accuracy

5-fold cross-validation, split **by avocado** (no fruit appears in both train and test):

| Model | Exact stage | Within ±1 | Overripe (stage 5) found |
|---|---|---|---|
| Colour features only (v1) | 67% | 98% | 56% |
| MobileNetV4-small logits | 53% | 92% | 48% |
| DINOv2-small | 72% | 99% | 74% |
| **DINOv2-small + colour (live)** | **73%** | **99%** | **71%** |

Missed overripe avocados are almost always called stage 4 (ripe, soft): overripeness (mould, flesh separating from
the skin) often doesn't show on the outside, so the site suggests a stem-cap / press check when it is torn between 4 and 5.

Robustness of the full app pipeline (`training/test-app2.mjs`, ~1,000 photos; model trained on all photos, so compare rows,
not absolute numbers): full frame 75%; auto-crop + colour correction 74%; same under simulated warm light 74%;
cool light 67% (the simulation clips the white background, which defeats the correction).

These are lab photos. Phone photos in kitchen light will do worse; how much is not yet measured.

### Days left

From the spreadsheet: for each avocado and stage, the number of days from when it was first seen at that
stage until it first reached stage 3 (ripe) and stage 5 (overripe). Median and inter-quartile range per
storage group: ambient (`Tam`) and 10 °C (`T10`).

## Data

Xavier, P., Rodrigues, P. & Silva, C. L. M. (2024). *'Hass' Avocado Ripening Photographic Dataset*.
Mendeley Data, V1. https://doi.org/10.17632/3xd9n945v8.1 — CC BY 4.0.
14,710 photos of 478 avocados, photographed daily in a lightbox. The spreadsheet and three sample photos
are included here under that licence.

## Retraining

```sh
cd training
npm install
# download + unzip the dataset from Mendeley (418 MB), then:
node stats.mjs                              # -> docs/shelf-life.json
node extract.mjs "<path to 'Avocado Ripening Dataset' image folder>"   # -> data/features.json
node train.mjs cv                           # cross-validation report
node train.mjs 0                            # -> docs/model.json (colour-only fallback)
node embed.mjs onnx-community/dinov2-small q8 "<image folder>"   # ~20 min -> data/emb-dinov2-small-q8.*
node train-emb.mjs cv dinov2-small-q8+color # cross-validation
node train-emb.mjs save dinov2-small-q8+color 0.01   # -> docs/head.json
node test-app2.mjs "<image folder>"         # app pipeline: auto-crop, colour correction, lighting
node browser-test.mjs                       # runs the samples in real Chrome (needs serve.mjs running)
cd .. && node training/test-app.mjs "<image folder>"   # app pipeline under simulated lighting
node training/serve.mjs                     # local preview at http://localhost:8123
```

## Ideas for next steps

- Collect real phone photos (with known stage) to measure and fine-tune for kitchen conditions.
- Scan both sides of the fruit (as in the dataset) and average.
- Fine-tune the network itself (needs a GPU / Python) rather than only training a classifier on its output.
- Train with colour/crop augmentation so phone photos look less unfamiliar.
