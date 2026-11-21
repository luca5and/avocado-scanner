// Prediction + ripening outlook. Pure functions, used by the app and by training/test-app.mjs.

export const STAGES = [
  { name: 'Underripe', short: 'Not ready', advice: 'Firm and yellowish-green. Leave it on the counter to ripen.',
    detail: 'Very firm, skin yellowish-green.' },
  { name: 'Breaking', short: 'Ripening', advice: 'Starting to soften. A few more days on the counter.',
    detail: 'Skin darkening to greyish olive-green with brown hues; gives slightly under pressure.' },
  { name: 'Ripe (early)', short: 'Ripe – firm', advice: 'Ready to eat. Good for slicing; still holds its shape.',
    detail: 'Purple spots appearing, flesh softened. Slices well but resists mashing.' },
  { name: 'Ripe (late)', short: 'Ripe – soft', advice: 'Eat now. Perfect for guacamole. Move to the fridge if you need more time.',
    detail: 'Uniformly purple, soft flesh, dry light-brown stem. End of optimal shelf life.' },
  { name: 'Overripe', short: 'Past its best', advice: 'Likely past its prime. Check the flesh for brown or grey areas before eating.',
    detail: 'Signs of senescence: mould, flesh separating from the skin.' },
];

export function predict(model, x) {
  const z = model.W2.map((row, k) => {
    let h = x.map((v, j) => (v - model.scaler.mu[j]) / model.scaler.sd[j]);
    if (model.W1) h = model.W1.map((r, i) => Math.max(0, r.reduce((s, w, j) => s + w * h[j], model.b1[i])));
    return row.reduce((s, w, j) => s + w * h[j], model.b2[k]);
  });
  const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), se = e.reduce((a, b) => a + b);
  const probs = e.map(v => v / se);
  const best = probs.indexOf(Math.max(...probs));
  // runner-up, for "between X and Y" messages
  const second = probs.map((p, i) => [p, i]).filter(([, i]) => i !== best).sort((a, b) => b[0] - a[0])[0][1];
  return { probs, stage: best + 1, confidence: probs[best], runnerUp: second + 1 };
}

/**
 * Days until ripe / until past its prime, weighted by the stage probabilities.
 * group: 'Tam' (room temperature) or 'T10' (cool storage, 10 °C).
 */
export function outlook(shelf, probs, group) {
  const g = shelf.groups[group].stages;
  const agg = key => {
    let med = 0, lo = 0, hi = 0;
    probs.forEach((p, i) => {
      const s = g[i + 1][key];
      const v = s ?? { median: 0, p25: 0, p75: 0 }; // already past that milestone
      med += p * v.median; lo += p * v.p25; hi += p * v.p75;
    });
    return { median: med, low: lo, high: hi };
  };
  return { toRipe: agg('toRipe'), toOverripe: agg('toOverripe') };
}
