export function colorRenderer(weights, unpack) {
 const { blocks, hud, board_shades: boardShades, hud_shades: hudShades } = weights;
 const [board, numbers] = weights.banks.map(bank =>
  new Map(Object.entries(bank).map(([key, value]) => [Number(key), unpack(value)])));
 const preview = new Set(blocks);
 const background = unpack(weights.screens.game);
 function keyAt(frame, row, col) {
  const start = row * 320 + col * 4;
  let key = 0;
  for (let y = 0; y < 4; y++) {
   for (let x = 0; x < 4; x++) key |= frame[start + y * 80 + x] << (2 * (4 * y + x));
  }
  return key >>> 0;
 }
 function paint(frame, result, row, col, targetRow, tile, shades) {
  const input = row * 320 + col * 4;
  const output = targetRow * 1280 + col * 8;
  for (let y = 0; y < 8; y++) {
   for (let x = 0; x < 8; x++) {
    result[output + y * 160 + x] = tile
     ? tile[y * 8 + x]
     : shades[frame[input + (y >> 1) * 80 + (x >> 1)]];
   }
  }
 }
 return frame => {
  const result = background.slice();
  for (let row = 0; row < 18; row++) {
   for (let col = 2; col < 12; col++) {
    paint(frame, result, row, col, row, board.get(keyAt(frame, row, col)), boardShades);
   }
  }
  for (const [row, col] of hud) {
   paint(frame, result, row, col, row, numbers.get(keyAt(frame, row, col)), hudShades);
  }
  for (let row = 0; row < 5; row++) {
   for (let col = 13; col < 19; col++) {
    const key = keyAt(frame, row, col);
    if (preview.has(key)) paint(frame, result, row, col, row + 1, board.get(key), boardShades);
   }
  }
  return result;
 };
}
