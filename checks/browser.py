"""Play both browser models with the network disabled after loading their assets."""
import os
import re
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
    page.on("request", lambda request: requests.append(request.url))
    page.on("websocket", lambda socket: sockets.append(socket.url))
    status = page.locator("#status")
    canvas = page.locator("canvas")
    page.goto(URL)
    expect(page.get_by_role("link", name="V2", exact=True)).to_have_attribute("aria-current", "page")
    expect(status).to_have_text(TITLE, timeout=60000)
    expect(canvas).to_have_attribute("width", "160")
    expect(canvas).to_have_attribute("height", "144")

    context.set_offline(True)
    before = len(requests)
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    play = page.get_by_role("button", name="Play", exact=True)
    play.focus()
    page.keyboard.press("Enter")
    expect(play).to_be_hidden()
    wait_steps(page, 2)
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
    page.keyboard.press("z")
    expect(status).to_have_text(STEPS)
    page.keyboard.down("ArrowDown")
    try:
        expect(status).to_have_text(TITLE, timeout=120000)
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
    browser.close()
