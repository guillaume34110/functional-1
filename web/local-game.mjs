import * as ort from "./vendor/ort.wasm.min.mjs";
import { nativeRenderer } from "./native-render.mjs";

ort.env.wasm.numThreads = 1;

const version = new URL(import.meta.url).searchParams.get("version") === "v1" ? "v1" : "v2";
const root = new URL(`./models/${version}`, import.meta.url).href.split("?")[0];
const options = {
 executionProviders: ["wasm"],
 graphOptimizationLevel: "disabled",
 enableMemPattern: false,
};
const tick = 1000 / 15;
const unpack = encoded => Uint8Array.from(atob(encoded), byte => byte.charCodeAt(0));
const frameShape = [1, 1, 72, 80];

let held = "noop", paused = false, restart = false;
const pulses = [];
self.onmessage = ({ data }) => {
 held = data.held;
 paused = data.paused;
 if (data.reset) restart = true;
 if (paused || restart) pulses.length = 0;
 if (data.pulse && !paused && pulses.length < 16) pulses.push(data.pulse);
};

async function play() {
 const [config, initial, recurrent] = await Promise.all([
  fetch(root + ".json").then(response => {
   if (!response.ok) throw new Error("Could not load model metadata.");
   return response.json();
  }),
  ort.InferenceSession.create(root + "-initial.onnx", options),
  ort.InferenceSession.create(root + "-recurrent.onnx", options),
 ]);
 const render = config.native ? nativeRenderer(config.native.keys, config.native.tiles) : pixels => pixels;
 const initialFrame = unpack(config.initial_frame);
 const pausePanel = config.pause ? unpack(config.pause.frame) : null;
 const title = config.title ? unpack(config.title) : null;
 const logo = title ? unpack(config.title_logo) : null;
 const seedState = config.initial_state
  ? new ort.Tensor("float32", Float32Array.from(config.initial_state.flat(3)),
   [1, config.initial_state[0].length, 18, 20])
  : null;
 let pixels, carry, phase, suspended, menu, steps, previous;

 function reset() {
  pixels = initialFrame;
  carry = seedState;
  phase = config.pause?.initial_phase ?? 0;
  suspended = false;
  steps = 0;
 }
 function publish(image, elapsed) {
  const update = {
   steps,
   inference_ms: elapsed,
   menu: Boolean(menu),
   width: title ? 160 : 80,
   height: title ? 144 : 72,
  };
  if (!previous || image.some((value, index) => value !== previous[index])) {
   previous = image;
   update.frame = new Uint8Array(image);
   postMessage(update, [update.frame.buffer]);
  } else {
   postMessage(update);
  }
 }

 async function advance(action) {
  if (menu) {
   if (config.starts[action]) {
    menu = null;
    reset();
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
   if (logo?.every((value, index) => value === pixels[index])) menu = render(pixels);
  }
  steps++;
  return performance.now() - started;
 }

 reset();
 menu = title;
 publish(menu ?? render(pixels), 0);
 async function update() {
  const started = performance.now();
  if (restart) {
   restart = false;
   reset();
   menu = title;
   publish(menu ?? render(pixels), 0);
  } else if (!paused && (!menu || pulses.length || held !== "noop")) {
   const action = config.actions.indexOf(pulses.shift() ?? held);
   const elapsed = await advance(action);
   publish(menu ?? render(suspended ? pausePanel : pixels), elapsed);
  }
  setTimeout(() => update().catch(fail), Math.max(0, tick - (performance.now() - started)));
 }
 await update();
}

function fail(error) {
 postMessage({ error: error instanceof Error ? error.message : String(error) });
}

play().catch(fail);
