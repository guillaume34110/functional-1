import * as ort from "./vendor/ort.wasm.min.mjs";
import { nativeRenderer } from "./native-render.mjs";
import { colorRenderer } from "./color-render.mjs";

ort.env.wasm.numThreads = 1;

const requested = new URL(import.meta.url).searchParams.get("version");
const version = requested === "v1" || requested === "v3" ? requested : "v2";
const root = new URL(`./models/${version === "v3" ? "v2" : version}`, import.meta.url).href.split("?")[0];
const options = {
 executionProviders: ["wasm"],
 graphOptimizationLevel: "disabled",
 enableMemPattern: false,
};
const tick = 1000 / 15;
const unpack = encoded => Uint8Array.from(atob(encoded), byte => byte.charCodeAt(0));
const frameShape = [1, 1, 72, 80];

let held = "noop", paused = false, restart = false, toggleCredits = false;
const pulses = [];
self.onmessage = ({ data }) => {
 held = data.held;
 paused = data.paused;
 if (data.reset) restart = true;
 if (data.credits) toggleCredits = true;
 if (paused || data.reset) pulses.length = 0;
 const softDrop = version === "v3" && data.pulse === "down" && data.softDrop;
 if (data.pulse && !paused && pulses.length < (version === "v3" ? 32 : 16)) {
  pulses.push({ button: data.pulse, softDrop });
 }
};

async function play() {
 const readJson = async url => {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Could not load model metadata.");
  return response.json();
 };
 const [config, initial, recurrent, skin] = await Promise.all([
  readJson(root + ".json"),
  ort.InferenceSession.create(root + "-initial.onnx", options),
  ort.InferenceSession.create(root + "-recurrent.onnx", options),
  version === "v3" ? readJson("/models/v3-color.json") : null,
 ]);
 const render = skin ? colorRenderer(skin, unpack) :
  config.native ? nativeRenderer(config.native.keys, config.native.tiles) : pixels => pixels;
 const initialFrame = unpack(config.initial_frame);
 const levelFrames = config.level_frames?.map(unpack);
 if (skin && config.color_level_10) levelFrames.push(unpack(config.color_level_10));
 let level = 1;
 const pausePanel = skin ? unpack(skin.screens.pause) : config.pause ? unpack(config.pause.frame) : null;
 const title = skin ? unpack(skin.screens.title) : config.title ? unpack(config.title) : null;
 const logo = config.title ? unpack(config.title_logo) : null;
 const creditsPanel = skin ? unpack(skin.screens.credits) : null;
 const overPanel = skin ? unpack(skin.screens.over) : null;
 let credits = false, ending = false, endUntil = 0;
 const seedState = config.initial_state
  ? new ort.Tensor("float32", Float32Array.from(config.initial_state.flat(3)),
   [1, config.initial_state[0].length, 18, 20])
  : null;
 let pixels, carry, phase, suspended, menu, steps, previous;
 let rotationGap = 0;
 let renderedPixels, renderedFrame;
 function display(frame) {
  if (!config.native) return frame;
  if (!renderedPixels || frame.some((value, index) => value !== renderedPixels[index])) {
   renderedPixels = frame;
   renderedFrame = render(frame);
  }
  return renderedFrame;
 }

 function reset() {
  pixels = levelFrames?.[level - 1] ?? initialFrame;
  carry = seedState;
  phase = config.pause?.initial_phase ?? 0;
  suspended = false;
  credits = ending = false;
  endUntil = 0;
  steps = 0;
  rotationGap = 0;
 }
 function publish(image, elapsed) {
  const update = {
   steps,
   inference_ms: elapsed,
   menu: Boolean(menu) && !ending,
   level: levelFrames ? level : undefined,
   width: title ? 160 : 80,
   height: title ? 144 : 72,
   screen: credits ? "credits" : ending ? "over" : menu ? "title" : suspended ? "pause" : "game",
  };
  if (image !== previous && (!previous || image.some((value, index) => value !== previous[index]))) {
   previous = image;
   update.frame = new Uint8Array(image);
   if (skin) update.palette = skin.palette;
   postMessage(update, [update.frame.buffer]);
  } else {
   postMessage(update);
  }
 }

 async function advance(action) {
  if (menu) {
   if (credits) {
    if (config.starts[action] || config.actions[action] === "b") credits = false;
    return 0;
   }
   if (config.starts[action]) {
    menu = null;
    reset();
   } else if (levelFrames) {
    const button = config.actions[action];
    const change = button === "right" || button === "up" ? 1 :
     button === "left" || button === "down" ? -1 : 0;
    level = Math.max(1, Math.min(levelFrames.length, level + change));
   }
   return 0;
  }
  const started = performance.now();
  const control = config.pause;
  const mayPause = suspended || control?.transitions[0][action] > 0;
  if (mayPause && control.transitions[suspended ? 1 : phase][action] > 0) {
   suspended = true;
  } else {
   if (suspended) action = control.resume_action;
   suspended = false;
   const inputs = {
    frame: new ort.Tensor("uint8", pixels, frameShape),
    action: new ort.Tensor("int64", BigInt64Array.of(BigInt(action)), [1]),
   };
   if (carry) inputs.state = carry;
   const next = await (carry ? recurrent : initial).run(inputs);
   pixels = next.pixels.data;
   carry = next.carry;
   phase = next.phase ? Number(next.phase.data[0]) : 0;
   if (skin && Number(carry.data[(carry.dims[1] - 2) * 360]) < 0 && !ending) {
    ending = true;
    endUntil = performance.now() + 950;
   }
   if (logo?.every((value, index) => value === pixels[index])) menu = skin ? title : display(pixels);
  }
  steps++;
  return performance.now() - started;
 }

 reset();
 menu = title;
 publish(menu ?? display(pixels), 0);
 let deadline = performance.now();
 async function update() {
  const started = performance.now();
  if (restart) {
   restart = false;
   reset();
   menu = title;
   publish(menu ?? display(pixels), 0);
  } else if (toggleCredits) {
   toggleCredits = false;
   if (creditsPanel && menu && !ending) {
    credits = !credits;
    publish(credits ? creditsPanel : title, 0);
   }
  } else if (ending && menu) {
   if (performance.now() >= endUntil) {
    ending = false;
    pulses.length = 0;
    publish(title, 0);
   }
  } else if (!paused && (!menu || pulses.length)) {
   const button = pulses[0]?.button ?? held;
   // The learned drop can stall when Down follows a rotation with fewer than two intervening samples.
   const deferDrop = version === "v3" && !menu && !suspended && button === "down" && rotationGap > 0;
   const pulse = deferDrop ? undefined : pulses.shift();
   // A soft drop can take two samples in the learned dynamics, even for a tap.
   if (pulse?.softDrop && !menu && !suspended) pulses.unshift({ button: "down" });
   const action = config.actions.indexOf(deferDrop ? "noop" : button);
   if (version === "v3" && !menu && !suspended) {
    rotationGap = button === "a" || button === "b" ? 2 : Math.max(0, rotationGap - 1);
   }
   const elapsed = await advance(action);
   const image = credits ? creditsPanel : ending ? overPanel : suspended && skin ? pausePanel :
    menu ?? display(suspended ? pausePanel : pixels);
   publish(image, elapsed);
  }
  const now = performance.now();
  // Preserve the clock phase without accumulating missed ticks after a slow inference.
  deadline = version !== "v1" ? Math.max(deadline + tick, now) : started + tick;
  setTimeout(() => update().catch(fail), Math.max(0, deadline - now));
 }
 await update();
}

function fail(error) {
 postMessage({ error: error instanceof Error ? error.message : String(error) });
}

play().catch(fail);
