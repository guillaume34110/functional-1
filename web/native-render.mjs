export function nativeRenderer(keys, tiles) {
 const vocabulary = new Map(Array.from(keys, (key, index) => [key, tiles[index]]));
 return frame => {
  const pixels = new Uint8Array(160 * 144);
  for (let row = 0; row < 18; row++) {
   for (let col = 0; col < 20; col++) {
    const input = row * 4 * 80 + col * 4;
    const output = row * 8 * 160 + col * 8;
    let key = 0;
    for (let y = 0; y < 4; y++) {
     for (let x = 0; x < 4; x++) {
      key |= frame[input + y * 80 + x] << (2 * (4 * y + x));
     }
    }
    const tile = vocabulary.get(key >>> 0);
    for (let y = 0; y < 8; y++) {
     for (let x = 0; x < 8; x++) {
      pixels[output + y * 160 + x] = tile
       ? tile[y * 8 + x]
       : frame[input + (y >> 1) * 80 + (x >> 1)];
     }
    }
   }
  }
  return pixels;
 };
}
