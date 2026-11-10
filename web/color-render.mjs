const digits = [
 [14, 17, 17, 17, 17, 17, 14],
 [4, 12, 4, 4, 4, 4, 14],
 [14, 17, 1, 2, 4, 8, 31],
 [30, 1, 1, 14, 1, 1, 30],
 [2, 6, 10, 18, 31, 2, 2],
 [31, 16, 16, 30, 1, 1, 30],
 [14, 16, 16, 30, 17, 17, 14],
 [31, 1, 2, 4, 8, 8, 8],
 [14, 17, 17, 14, 17, 17, 14],
 [14, 17, 17, 15, 1, 1, 14],
];

export function colorRenderer(weights, unpack, digitKeys = []) {
 const { blocks, hud, board_shades: boardShades, hud_shades: hudShades } = weights;
 const [board, numbers] = weights.banks.map(bank =>
  new Map(Object.entries(bank).map(([key, value]) => [Number(key), unpack(value)])));
 const preview = new Set(blocks);
 const counters = new Map(digitKeys.map((key, value) => [key, value]));
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
   const key = keyAt(frame, row, col);
   const value = counters.get(key);
   if (value === undefined) {
    paint(frame, result, row, col, row, numbers.get(key), hudShades);
   } else if (value < digits.length) {
    const origin = (row * 8 - 4) * 160 + col * 8 + 1;
    for (let y = 0; y < 11; y++) {
     const bits = digits[value][Math.floor(y * 7 / 11)];
     for (let x = 0; x < 5; x++) {
      if (bits & (16 >> x)) result[origin + y * 160 + x] = 5;
     }
    }
   }
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
