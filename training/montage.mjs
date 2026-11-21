import sharp from 'sharp';
const [dir, out, ...files] = process.argv.slice(2);
const tiles = await Promise.all(files.map(f => sharp(`${dir}/${f}`).resize(200, 200).toBuffer()));
await sharp({ create: { width: 200 * tiles.length, height: 200, channels: 3, background: '#888' } })
  .composite(tiles.map((t, i) => ({ input: t, left: i * 200, top: 0 }))).jpeg().toFile(out);
