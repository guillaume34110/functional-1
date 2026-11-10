"""Exercise short, sustained and overlapping controls on the V3 canvas."""
from playwright.sync_api import expect


PIECE = """() => {
  const canvas = document.querySelector('#screen');
  if (canvas.dataset.screen !== 'game') return null;
  const pixels = canvas.getContext('2d').getImageData(16, 0, 80, 144).data;
  const cells = [];
  for (let y = 0; y < 18; y++) for (let x = 0; x < 10; x++) {
    if (pixels[((y * 8 + 4) * 80 + x * 8 + 4) * 4] > 40) cells.push([y, x]);
  }
  if (cells.length !== 4) return null;
  const xs = cells.map(([, x]) => x);
  const row = cells[0][0], left = Math.min(...xs);
  return {row, left, right: Math.max(...xs),
    height: cells.at(-1)[0] - row + 1};
}"""


def exercise_v3_inputs(page, canvas, level):
    for _ in range(10 - int(level.inner_text())):
        page.keyboard.press("ArrowRight")
    expect(level).to_have_text("10")

    def start():
        page.get_by_role("button", name="Restart", exact=True).click()
        expect(canvas).to_have_attribute("data-screen", "title")
        page.get_by_role("button", name="Play", exact=True).click()
        expect(canvas).to_have_attribute("data-screen", "game")

    start()
    page.wait_for_function("Number(document.querySelector('#status').textContent.split(' ')[0]) >= 2")
    burst = page.evaluate(f"""() => new Promise((resolve, reject) => {{
      const read = {PIECE};
      const start = read();
      const status = document.querySelector('#status');
      const step = () => Number(status.textContent.split(' ')[0]);
      const end = step() + 10;
      const heights = [start.height];
      const observer = new MutationObserver(() => {{
        const piece = read();
        if (piece && piece.height !== heights.at(-1)) heights.push(piece.height);
        if (step() >= end) {{
          observer.disconnect();
          clearTimeout(timer);
          resolve({{heights, drop: piece && piece.row - start.row}});
        }}
      }});
      const timer = setTimeout(() => {{
        observer.disconnect();
        reject(new Error('No model steps after input burst'));
      }}, 5000);
      observer.observe(status, {{childList: true}});
      for (const key of ['z', 'x', 'z', 'ArrowDown', 'ArrowDown']) {{
        for (const type of ['keydown', 'keyup']) {{
          dispatchEvent(new KeyboardEvent(type, {{key, bubbles: true}}));
        }}
      }}
    }})""")
    assert burst["heights"] == [2, 3, 2, 3], burst
    assert burst["drop"] is not None and burst["drop"] >= 3, burst

    for arrow, side, column, turn, back in (
        ("ArrowLeft", "left", 0, "z", "x"), ("ArrowRight", "right", 9, "x", "z")
    ):
        start()
        page.keyboard.down(arrow)
        try:
            page.wait_for_function(f"column => ({PIECE})()?.{side} === column", arg=column, timeout=5000)
        finally:
            page.keyboard.up(arrow)
        page.keyboard.press(turn)
        page.wait_for_function(f"() => ({PIECE})()?.height === 3", timeout=3000)
        page.keyboard.press(back)
        page.wait_for_function(f"() => ({PIECE})()?.height === 2", timeout=3000)

    start()
    page.evaluate("""() => {
      const post = Worker.prototype.postMessage;
      window.v3Inputs = [];
      Worker.prototype.postMessage = function (message, ...args) {
        window.v3Inputs.push(message);
        return post.call(this, message, ...args);
      };
      window.restoreInputProbe = () => {
        Worker.prototype.postMessage = post;
        delete window.v3Inputs;
        delete window.restoreInputProbe;
      };
    }""")
    button = page.get_by_role("button", name="Down", exact=True)
    button.scroll_into_view_if_needed()
    bounds = button.bounding_box()
    page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
    try:
        page.mouse.down()
        page.keyboard.down("ArrowDown")
        page.keyboard.up("ArrowDown")
        assert page.evaluate("window.v3Inputs.at(-1).held") == "down"
        page.wait_for_function(f"() => ({PIECE})()?.row >= 5", timeout=1500)
        page.mouse.up()
        page.wait_for_function("window.v3Inputs.at(-1).held === 'noop'")
        assert page.evaluate("window.v3Inputs.filter(input => input.pulse === 'down').length") == 1

        page.keyboard.down("ArrowLeft")
        page.keyboard.down("ArrowRight")
        assert page.evaluate("window.v3Inputs.at(-1).held") == "right"
        page.keyboard.up("ArrowRight")
        assert page.evaluate("window.v3Inputs.at(-1).held") == "left"
        page.evaluate("dispatchEvent(new Event('blur'))")
        assert page.evaluate("window.v3Inputs.at(-1).held === 'noop' && window.v3Inputs.at(-1).paused")
        page.keyboard.up("ArrowLeft")
    finally:
        page.mouse.up()
        page.evaluate("window.restoreInputProbe()")
    print("V3 level 10 controls: five-tap burst, both walls, overlapping holds, held priority and blur", flush=True)


def exercise_v3_cadence(page, canvas):
    def delay_worker(route):
        response = route.fetch()
        source = response.text()
        anchor = "const elapsed = await advance(action);"
        assert source.count(anchor) == 1
        delayed = anchor + "\n   if (steps === 12) await new Promise(resolve => setTimeout(resolve, 150));"
        route.fulfill(status=response.status, content_type="application/javascript",
                      body=source.replace(anchor, delayed))

    page.route("**/local-game.mjs?version=v3", delay_worker)
    page.reload()
    expect(canvas).to_have_attribute("data-screen", "title", timeout=60000)
    page.evaluate("""() => {
      const status = document.querySelector('#status');
      window.stepTimes = {};
      new MutationObserver(() => {
        const step = Number(status.textContent.split(' ')[0]);
        if (Number.isInteger(step)) window.stepTimes[step] = performance.now();
      }).observe(status, {childList: true});
    }""")
    page.get_by_role("button", name="Play", exact=True).click()
    page.wait_for_function("window.stepTimes[13] !== undefined", timeout=10000)
    delayed, next_gap = page.evaluate("""() => [
      window.stepTimes[12] - window.stepTimes[11],
      window.stepTimes[13] - window.stepTimes[12]
    ]""")
    assert delayed > 130 and next_gap >= 20, (delayed, next_gap)
    print(f"V3 delayed step: {delayed:.1f} ms, next step: {next_gap:.1f} ms", flush=True)
