const screen = document.querySelector("#screen");
const art = document.querySelector("#launch-art");
const ctx = art.getContext("2d");
const skip = document.querySelector("#skip-intro");
const motion = matchMedia("(prefers-reduced-motion: reduce)");
const ink = "#21193b", space = "#181f43", cream = "#fff1d8", pink = "#efacd2", mint = "#b9e6ba";
const stars = [[12, 14], [38, 31], [53, 12], [81, 22], [106, 12], [138, 26],
 [149, 51], [119, 62], [15, 67], [38, 81], [74, 92], [134, 106], [22, 114], [57, 128]];
const meteors = [
 { at: 320, x: 174, y: 27, speed: .055, size: 7 },
 { at: 560, x: 181, y: 89, speed: .062, size: 9 },
 { at: 910, x: 158, y: 27, speed: .13, size: 8 },
];
const crash = 1440, ejection = 2130, landing = 3220, duration = 3840;
let intro = !motion.matches, request = 0, last = -Infinity, elapsed = 0;
const began = performance.now();

function box(color, x, y, w, h) {
 ctx.fillStyle = color;
 ctx.fillRect(Math.round(x), Math.round(y), w, h);
}

function sparkle(x, y, color) {
 box(color, x - 2, y, 5, 1);
 box(color, x, y - 2, 1, 5);
}

function bunny(x, y, alarmed = false) {
 box(ink, x + 1, y, 4, 8);
 box(ink, x + 8, y + 1, 4, 7);
 box(cream, x + 2, y + 1, 2, 7);
 box(cream, x + 9, y + 2, 2, 6);
 box(pink, x + 3, y + 3, 1, 3);
 box(pink, x + 9, y + 4, 1, 3);
 box(ink, x, y + 7, 13, 9);
 box(cream, x + 1, y + 8, 11, 7);
 box(ink, x + 3, y + 10, 1, 2);
 box(ink, x + 9, y + 10, 1, 2);
 box(alarmed ? ink : "#a45586", x + 6, y + 12, 1, alarmed ? 2 : 1);
 box(pink, x + 2, y + 13, 2, 1);
 box(pink, x + 9, y + 13, 2, 1);
}

function shuttle(x, y, damaged, time) {
 const flame = 8 + Math.floor(time / 90) % 3 * 3;
 box("#bd5b79", x - flame, y + 13, flame + 4, 8);
 box("#ffd084", x - flame + 3, y + 15, flame + 3, 4);
 box(ink, x + 3, y + 4, 32, 26);
 box(ink, x + 10, y, 20, 6);
 box(ink, x + 33, y + 9, 11, 16);
 box("#7864a7", x + 4, y + 7, 31, 20);
 box("#c9a6e1", x + 7, y + 5, 26, 18);
 box("#e7bee9", x + 10, y + 3, 20, 4);
 box(damaged ? "#57334c" : "#b27bb7", x + 35, y + 11, 7, 12);
 box("#384a74", x + 10, y + 7, 16, 13);
 box("#659caa", x + 11, y + 8, 14, 11);
 bunny(x + 12, y + 4, damaged);
 box(pink, x + 8, y + 24, 5, 4);
 box(pink, x + 27, y + 24, 5, 4);
 if (damaged) {
  box(ink, x + 29, y + 8, 4, 2);
  box(ink, x + 30, y + 10, 2, 5);
  box(ink, x + 27, y + 14, 5, 2);
  box("#ffbb7c", x + 34, y + 20, 3, 2);
 }
}

function asteroid(x, y, size) {
 box("#b85b73", x + size, y - 3, size + 11, 3);
 box("#f2a485", x + size, y, size + 6, 3);
 box(ink, x, y - 2, size, size);
 box("#845671", x + 1, y - 1, size - 2, size - 2);
 box("#bb8597", x + 2, y - 1, size - 4, 2);
 box("#553c65", x + 2, y + 2, 2, 2);
}

function drawLaunch(time) {
 const damaged = time >= crash;
 const escaped = time >= ejection;
 const landed = time >= landing;
 art.dataset.beat = landed ? "landing" : escaped ? "escape" : damaged ? "impact" :
  time >= meteors[0].at ? "asteroids" : "arrival";
 box(space, 0, 0, 160, 144);
 for (const [i, [x, y]] of stars.entries()) {
  const drift = (x + 160 - Math.floor(time / (70 + i * 4)) % 160) % 160;
  box((Math.floor(time / 240) + i) % 5 === 0 ? cream : "#6c739c", drift, y, 1, 1);
 }
 const planet = Math.max(0, (time - 2170) / 1150);
 if (planet > 0) {
  const top = 146 - Math.min(1, planet) * 37;
  box("#526a9a", 20, top, 120, 35);
  box("#526a9a", 0, top + 12, 160, 26);
  box("#548c8a", 8, top + 15, 144, 24);
  box(mint, 33, top + 15, 92, 4);
  box("#79ae99", 50, top + 19, 60, 5);
  box("#548c8a", 21, top + 24, 129, 20);
 }
 if (!escaped) {
  const shock = damaged ? Math.sin((time - crash) / 21) * Math.max(0, 4 - (time - crash) / 155) : 0;
  shuttle(-42 + Math.min(100, time * .073), 56 + Math.round(Math.sin(time / 190) * 3 + shock), damaged, time);
 } else if (time < landing) {
  const fall = (time - ejection) / (landing - ejection);
  const x = 69 + Math.round(Math.sin(fall * 5.5) * 6);
  const y = 48 + fall * 57;
  box("#b974ad", x - 9, y - 13, 31, 5);
  box(pink, x - 6, y - 16, 25, 5);
  box(cream, x + 2, y - 16, 9, 8);
  box("#b7a3dd", x - 5, y - 8, 1, 9);
  box("#b7a3dd", x + 17, y - 8, 1, 9);
  bunny(x, y);
 } else {
  bunny(73, 113 + Math.round(Math.sin((time - landing) / 80) * 2));
  sparkle(58, 115, cream);
  sparkle(102, 110, pink);
 }
 for (const rock of meteors) {
  const age = time - rock.at;
  if (age >= 0 && (rock !== meteors[2] || !damaged)) {
   asteroid(rock.x - age * rock.speed, rock.y + age * .082, rock.size);
  }
 }
 if (damaged && time < 2060) {
  const age = (time - crash) / 620;
  for (let i = 0; i < 8; i++) {
   const angle = i * Math.PI / 4;
   const radius = 5 + age * (20 + 5 * (i % 3));
   box(i % 3 === 0 ? "#a981b9" : "#ffc783", 88 + Math.cos(angle) * radius,
    68 + Math.sin(angle) * radius, age < .65 ? 3 : 2, 2);
  }
  if (age < .16) sparkle(88, 68, cream);
 }
 if (time > 3540) {
  ctx.globalAlpha = (time - 3540) / (duration - 3540);
  box(ink, 0, 0, 160, 144);
  ctx.globalAlpha = 1;
 }
}

function drawTitle(time) {
 ctx.clearRect(0, 0, 160, 144);
 if (time % 3600 < 145) {
  box(cream, 74, 31, 2, 3);
  box(cream, 83, 31, 2, 3);
  box(ink, 74, 33, 2, 1);
  box(ink, 83, 33, 2, 1);
 }
 const twinkles = [[21, 27], [42, 20], [133, 23], [117, 38], [27, 111], [138, 104]];
 const index = Math.floor(time / 340) % twinkles.length;
 sparkle(...twinkles[index], index % 2 ? cream : "#dbc5ff");
}

function finish() {
 intro = false;
 delete document.body.dataset.intro;
 delete art.dataset.beat;
 skip.hidden = true;
 sync();
}

function animate(now) {
 request = 0;
 if (!intro && (motion.matches || screen.dataset.screen !== "title")) return;
 if (now - last >= 1000 / 16) {
  if (intro) {
   elapsed += Number.isFinite(last) ? Math.min(now - last, 100) : 0;
   if (elapsed >= duration) {
    finish();
    return;
   }
   drawLaunch(elapsed);
  } else {
   drawTitle(now - began);
  }
  last = now;
 }
 request = requestAnimationFrame(animate);
}

function sync() {
 const visible = intro || (!motion.matches && screen.dataset.screen === "title");
 art.hidden = !visible;
 if (visible && !request) request = requestAnimationFrame(animate);
 if (!visible && request) {
  cancelAnimationFrame(request);
  request = 0;
 }
}

if (intro) {
 document.body.dataset.intro = "";
 skip.hidden = false;
 drawLaunch(0);
}
skip.addEventListener("click", finish);
window.addEventListener("keydown", event => {
 if (!intro || !["Enter", "Escape", " ", "z", "Z", "x", "X", "ArrowLeft", "ArrowRight",
  "ArrowUp", "ArrowDown"].includes(event.key)) return;
 event.preventDefault();
 event.stopImmediatePropagation();
 if (!event.key.startsWith("Arrow")) finish();
}, { capture: true });
motion.addEventListener("change", () => motion.matches && intro ? finish() : sync());
new MutationObserver(sync).observe(screen, { attributes: true, attributeFilter: ["data-screen"] });
sync();
