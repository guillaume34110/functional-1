"""Play both browser models with the network disabled after loading their assets."""
import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from v3_inputs import exercise_v3_cadence, exercise_v3_inputs


URL = os.environ.get("GAME_URL", "http://127.0.0.1:8765/").rstrip("/")
TITLE = "Press Z or Enter to play"
STEPS = re.compile(r"^\d+ neural steps")


def wait_steps(page, minimum, timeout=20000):
    page.wait_for_function(
        "(minimum) => Number(document.querySelector('#status').textContent.split(' ')[0]) >= minimum",
        arg=minimum, timeout=timeout,
    )


def cadence(page, label, start):
    wait_steps(page, start)
    rate = page.evaluate("""() => new Promise((resolve, reject) => {
      const status = document.querySelector('#status');
      const steps = () => Number(status.textContent.split(' ')[0]);
      const first = steps(), started = performance.now();
      const observer = new MutationObserver(() => {
        const advance = steps() - first;
        if (advance < 30 || !Number.isFinite(advance)) return;
        observer.disconnect();
        clearTimeout(timer);
        resolve(1000 * advance / (performance.now() - started));
      });
      const timer = setTimeout(() => {
        observer.disconnect();
        reject(new Error('The model did not advance 30 steps'));
      }, 20000);
      observer.observe(status, {childList: true});
    })""")
    print(f"{label} offline: {rate:.1f} steps/s", flush=True)


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL"))
    context = browser.new_context()
    page = context.new_page()
    page.on("console", lambda message: print(message.text, flush=True) if message.type == "error" else None)
    page.on("pageerror", lambda error: print(error, flush=True))
    requests, sockets = [], []
    shape_warnings = []
    page.on("request", lambda request: requests.append(request.url))
    page.on("websocket", lambda socket: sockets.append(socket.url))
    page.on("console", lambda message: shape_warnings.append(message.text)
            if "VerifyOutputSizes" in message.text or "Expected shape from model" in message.text else None)
    status = page.locator("#status")
    canvas = page.locator("#screen")
    page.goto(URL)
    expect(page.get_by_role("link", name="V2", exact=True)).to_have_attribute("aria-current", "page")
    expect(status).to_have_text(TITLE, timeout=60000)
    expect(canvas).to_have_attribute("width", "160")
    expect(canvas).to_have_attribute("height", "144")

    context.set_offline(True)
    before = len(requests)
    level = page.locator("#starting-level output")
    expect(level).to_have_text("1")
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.keyboard.press("ArrowRight")
    expect(level).to_have_text("2")
    page.keyboard.press("ArrowUp")
    expect(level).to_have_text("3")
    page.keyboard.press("ArrowDown")
    expect(level).to_have_text("2")
    play = page.get_by_role("button", name="Play", exact=True)
    play.focus()
    page.keyboard.press("Enter")
    expect(play).to_be_hidden()
    expect(level).to_be_hidden()
    wait_steps(page, 2)
    level_pixels = """canvas => Array.from(canvas.getContext('2d').getImageData(144, 120, 8, 8).data).join(',')"""
    level_two = canvas.evaluate(level_pixels)
    preview = """canvas => Array.from(canvas.getContext('2d').getImageData(104, 0, 48, 48).data).join(',')"""
    first_preview = canvas.evaluate(preview)
    page.keyboard.press("z")
    wait_steps(page, 5)
    page.keyboard.press("x")
    wait_steps(page, 8)
    assert canvas.evaluate(preview) == first_preview
    cadence(page, "V2", 10)
    image = canvas.evaluate("(canvas) => canvas.toDataURL()")
    page.keyboard.press("Enter")
    page.wait_for_function("(image) => document.querySelector('canvas').toDataURL() !== image", arg=image)
    paused_image = canvas.evaluate("(canvas) => canvas.toDataURL()")
    page.keyboard.press("Enter")
    page.wait_for_function("(image) => document.querySelector('canvas').toDataURL() !== image", arg=paused_image)
    page.get_by_role("button", name="Restart").click()
    expect(status).to_have_text(TITLE)
    expect(level).to_have_text("2")
    page.keyboard.press("ArrowLeft")
    expect(level).to_have_text("1")
    page.keyboard.press("z")
    expect(status).to_have_text(STEPS)
    assert canvas.evaluate(level_pixels) != level_two
    page.get_by_role("button", name="Restart").click()
    expect(status).to_have_text(TITLE)
    page.keyboard.press("ArrowRight")
    expect(level).to_have_text("2")
    page.keyboard.press("z")
    expect(status).to_have_text(STEPS)
    page.keyboard.down("ArrowDown")
    try:
        expect(status).to_have_text(TITLE, timeout=120000)
        expect(level).to_have_text("2")
    finally:
        page.keyboard.up("ArrowDown")
    page.keyboard.press("z")
    expect(status).to_have_text(STEPS)
    assert requests[before:] == [], requests[before:]

    context.set_offline(False)
    page.get_by_role("link", name="V1", exact=True).focus()
    page.keyboard.press("Enter")
    expect(page).to_have_url(URL + "/v1")
    expect(page.get_by_role("link", name="V1", exact=True)).to_have_attribute("aria-current", "page")
    expect(status).to_have_text(STEPS, timeout=60000)
    expect(level).to_be_hidden()
    expect(canvas).to_have_attribute("width", "80")
    expect(canvas).to_have_attribute("height", "72")
    context.set_offline(True)
    before = len(requests)
    cadence(page, "V1", 10)
    page.get_by_role("button", name="Pause", exact=True).click()
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.get_by_role("button", name="Restart").click()
    expect(status).to_contain_text("0 neural steps")
    page.get_by_role("button", name="Resume").click()
    wait_steps(page, 10)
    assert requests[before:] == [], requests[before:]
    assert sockets == [], sockets

    context.set_offline(False)
    page.get_by_role("link", name="V3", exact=True).click()
    expect(page).to_have_url(URL + "/v3")
    screenshot_dir = os.environ.get("V3_SCREENSHOT_DIR")

    def capture(name):
        if screenshot_dir:
            page.screenshot(path=Path(screenshot_dir) / f"{name}.png")

    intro = page.locator("#launch-art")
    skip_intro = page.get_by_role("button", name="Skip intro")
    expect(skip_intro).to_be_visible()
    page.wait_for_function("document.querySelector('#launch-art').dataset.beat === 'impact'", timeout=6000)
    capture("impact")
    page.wait_for_function("document.querySelector('#launch-art').dataset.beat === 'landing'", timeout=6000)
    capture("landing")
    expect(skip_intro).to_be_hidden(timeout=6000)
    expect(status).to_have_text(TITLE, timeout=60000)
    expect(canvas).to_have_attribute("data-screen", "title")
    expect(intro).to_be_visible()
    menu_art = intro.evaluate("canvas => canvas.toDataURL()")
    page.wait_for_function("pixels => document.querySelector('#launch-art').toDataURL() !== pixels", arg=menu_art)
    expect(canvas).to_have_attribute("width", "160")
    expect(canvas).to_have_attribute("height", "144")
    context.set_offline(True)
    before = len(requests)
    capture("title")
    page.get_by_role("button", name="Credits").click()
    expect(canvas).to_have_attribute("data-screen", "credits")
    expect(status).to_have_text("A game by Progsoft")
    page.get_by_role("button", name="Back to title").click()
    expect(canvas).to_have_attribute("data-screen", "title")
    level_one_title = canvas.evaluate("canvas => canvas.toDataURL()")
    page.keyboard.press("ArrowLeft")
    expect(level).to_have_text("1")
    for _ in range(10):
        page.keyboard.press("ArrowUp")
    expect(level).to_have_text("10")
    page.keyboard.press("ArrowRight")
    expect(level).to_have_text("10")
    level_ten_title = canvas.evaluate("canvas => canvas.toDataURL()")
    assert level_ten_title != level_one_title, "the level must change pixels inside the title screen"
    piece_extent = """() => {
      const ctx = document.querySelector('#screen').getContext('2d');
      const pixels = ctx.getImageData(16, 0, 80, 144).data;
      const positions = [];
      for (let row = 0; row < 18; row++) for (let col = 0; col < 10; col++) {
        if (pixels[((row * 8 + 4) * 80 + col * 8 + 4) * 4] > 40) positions.push(row);
      }
      return positions.length === 4 ? [Math.min(...positions), Math.max(...positions)] : null;
    }"""
    first_drop = f"""() => {{
      const rows = ({piece_extent})();
      return rows?.[0] > 0 && Number(document.querySelector('#status').textContent.split(' ')[0]);
    }}"""
    page.get_by_role("button", name="Play", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "game")
    expect(intro).to_be_hidden()
    tens_pixel = """canvas => {
      const digits = canvas.getContext('2d').getImageData(136, 120, 8, 8).data;
      return digits.some((red, i) => i % 4 === 0 && red > 150);
    }"""
    assert canvas.evaluate(tens_pixel), "level 10 should draw a tens digit in play"
    assert canvas.evaluate("""canvas => {
      const digit = canvas.getContext('2d').getImageData(145, 68, 5, 11).data;
      for (let y = 0; y < 11; y++) {
        if (!Array.from({length: 5}, (_, x) => (y * 5 + x) * 4)
          .some(i => digit[i] > 240 && digit[i + 1] > 230)) return false;
      }
      return true;
    }"""), "score glyphs should have 11 high-contrast rows"
    fast_drop = page.wait_for_function(first_drop).json_value()
    page.get_by_role("button", name="Restart", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "title")
    expect(level).to_be_visible()
    expect(level).to_have_text("10")
    assert canvas.evaluate("canvas => canvas.toDataURL()") == level_ten_title
    for _ in range(9):
        page.keyboard.press("ArrowDown")
    expect(level).to_have_text("1")
    assert canvas.evaluate("canvas => canvas.toDataURL()") == level_one_title
    page.keyboard.press("z")
    expect(canvas).to_have_attribute("data-screen", "game")
    slow_drop = page.wait_for_function(first_drop).json_value()
    assert fast_drop < slow_drop, (fast_drop, slow_drop)
    page.get_by_role("button", name="Restart", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "title")
    screen = canvas.bounding_box()
    page.mouse.click(screen["x"] + screen["width"] * 124 / 160,
                     screen["y"] + screen["height"] * 105 / 144)
    expect(level).to_have_text("2")
    page.get_by_role("button", name="Play", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "game")
    expect(level).to_be_hidden()
    # At this opening phase, Down used to stall after the second rotation.
    rotated, restored = page.evaluate(f"""() => new Promise((resolve, reject) => {{
      const status = document.querySelector('#status');
      const read = {piece_extent};
      const tap = key => {{
        for (const type of ['keydown', 'keyup']) {{
          window.dispatchEvent(new KeyboardEvent(type, {{key, bubbles: true}}));
        }}
      }};
      let turned;
      const watcher = new MutationObserver(() => {{
        const step = Number(status.textContent.split(' ')[0]);
        if (step === 12) tap('Z');
        if (step === 14) {{
          turned = read();
          document.querySelector('[data-action="b"]').click();
        }}
        if (step === 16) {{
          const back = read();
          tap('ArrowDown');
          watcher.disconnect();
          clearTimeout(deadline);
          resolve([turned, back]);
        }}
      }});
      const deadline = setTimeout(() => {{
        watcher.disconnect();
        reject(new Error('Did not reach the rotation and drop checkpoints'));
      }}, 5000);
      watcher.observe(status, {{childList: true}});
    }})""")
    assert rotated and rotated[1] - rotated[0] == 2
    assert restored and restored[1] - restored[0] == 1
    page.wait_for_function(f"row => {{ const rows = ({piece_extent})(); return rows && rows[0] > row; }}",
                           arg=restored[0], timeout=900)
    wait_steps(page, 2)
    assert canvas.evaluate("""canvas => {
      const pixels = canvas.getContext('2d').getImageData(16, 0, 80, 144).data;
      return pixels.some((red, index) => index % 4 === 0 && red > 170);
    }"""), "the falling piece should have a color distinct from the dark playfield"
    capture("game")
    page.get_by_role("button", name="Pause", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "pause")
    expect(status).to_have_text("Paused")
    capture("pause")
    page.get_by_role("button", name="Resume", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "game")
    cadence(page, "V3", 10)
    score_pixels = """() => document.querySelector('#screen').getContext('2d')
      .getImageData(112, 68, 40, 11).data.toString()"""
    before_score = page.evaluate(score_pixels)
    page.keyboard.down("ArrowDown")
    try:
        page.wait_for_function(f"before => ({score_pixels})() !== before", arg=before_score, timeout=12000)
        expect(canvas).to_have_attribute("data-screen", "over", timeout=120000)
        capture("game-over")
        expect(canvas).to_have_attribute("data-screen", "title", timeout=15000)
    finally:
        page.keyboard.up("ArrowDown")
    expect(level).to_have_text("2")
    page.keyboard.press("z")
    expect(canvas).to_have_attribute("data-screen", "game")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    expect(page.get_by_role("button", name="Rotate with Z")).to_be_visible()
    capture("mobile")
    page.get_by_role("button", name="Restart", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "title")
    exercise_v3_inputs(page, canvas, level)
    assert requests[before:] == [], requests[before:]
    assert sockets == [], sockets
    context.set_offline(False)
    page.reload()
    expect(skip_intro).to_be_visible()
    page.keyboard.press("Enter")
    expect(skip_intro).to_be_hidden()
    expect(canvas).to_have_attribute("data-screen", "title", timeout=60000)
    page.emulate_media(reduced_motion="reduce")
    page.reload()
    expect(canvas).to_have_attribute("data-screen", "title", timeout=60000)
    expect(skip_intro).to_be_hidden()
    expect(intro).to_be_hidden()
    exercise_v3_cadence(page, canvas)
    assert not shape_warnings, shape_warnings[:3]
    browser.close()
