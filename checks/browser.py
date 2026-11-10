"""Play both browser models with the network disabled after loading their assets."""
import os
import re
from pathlib import Path
from time import monotonic

from playwright.sync_api import expect, sync_playwright


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
    started = monotonic()
    wait_steps(page, start + 30)
    print(f"{label} offline: {30 / (monotonic() - started):.1f} steps/s", flush=True)


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
    canvas = page.locator("canvas")
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
    expect(status).to_have_text(TITLE, timeout=60000)
    expect(canvas).to_have_attribute("data-screen", "title")
    expect(canvas).to_have_attribute("width", "160")
    expect(canvas).to_have_attribute("height", "144")
    context.set_offline(True)
    before = len(requests)
    screenshot_dir = os.environ.get("V3_SCREENSHOT_DIR")

    def capture(name):
        if screenshot_dir:
            page.screenshot(path=Path(screenshot_dir) / f"{name}.png")

    capture("title")
    page.get_by_role("button", name="Credits").click()
    expect(canvas).to_have_attribute("data-screen", "credits")
    expect(status).to_have_text("A game by Progsoft")
    page.get_by_role("button", name="Back to title").click()
    expect(canvas).to_have_attribute("data-screen", "title")
    page.keyboard.press("ArrowRight")
    expect(level).to_have_text("2")
    page.get_by_role("button", name="Play", exact=True).click()
    expect(canvas).to_have_attribute("data-screen", "game")
    expect(level).to_be_hidden()
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
    page.keyboard.down("ArrowDown")
    try:
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
    assert requests[before:] == [], requests[before:]
    assert sockets == [], sockets
    assert not shape_warnings, shape_warnings[:3]
    browser.close()
