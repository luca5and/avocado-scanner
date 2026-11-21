# Avocado Scanner 🥑

Point your phone at a Hass avocado and see its ripening stage (1–5) and roughly how many days it has
until it is ripe and until it is past its prime.

**Live site:** https://luca5and.github.io/avocado-scanner/

Everything runs in the browser; photos are never uploaded.

## How it works

1. The camera view shows an oval guide. The photo is cropped to a square and shrunk to 128×128.
2. [`docs/features.js`](docs/features.js) separates the light background from the fruit (Otsu threshold),
   uses the background to correct white balance and exposure, and measures 31 colour/texture features
   of the skin inside the oval (CIELab statistics, hue histogram, dark/purple/green fractions, texture).
3. [`docs/model.js`](docs/model.js) runs a softmax classifier ([`docs/model.json`](docs/model.json), 2 KB)
   and turns the stage probabilities into "days left" using [`docs/shelf-life.json`](docs/shelf-life.json).

### Accuracy

5-fold cross-validation, split **by avocado** (no fruit appears in both train and test):

| | |
|---|---|
| Exact stage | 67% |
| Within ±1 stage | 98% |

Most errors are between stage 4 (ripe, soft) and 5 (overripe), which often look alike from the outside.
These numbers are for lab photos; phone photos in kitchen light will do somewhat worse. Simulated warm/dim
lighting costs only 1–2 points thanks to the white-paper correction; cool light costs ~6.

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
node train.mjs 0                            # -> docs/model.json (0 = no hidden layer)
cd .. && node training/test-app.mjs "<image folder>"   # app pipeline under simulated lighting
node training/serve.mjs                     # local preview at http://localhost:8123
```

## Ideas for next steps

- Collect real phone photos (with known stage) to measure and fine-tune for kitchen conditions.
- Scan both sides of the fruit (as in the dataset) and average.
- A small CNN (e.g. MobileNet via ONNX/TF.js) may separate stages 4 and 5 better.
