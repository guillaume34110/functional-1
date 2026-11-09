"""Check the Functional 2 stream, native canvas and restart in a real browser."""
import json
import os
from pathlib import Path
import re
from statistics import median
from time import monotonic

from playwright.sync_api import expect, sync_playwright


def capture_stream(page):
    frames = []

    def capture(socket):
        if socket.url.endswith("/stream"):
            socket.on("framereceived", lambda payload: frames.append((monotonic(), json.loads(payload))))

    page.on("websocket", capture)
    return frames


def check(page, frames):
    status = page.locator("#status")
    expect(status).to_have_text("Press Z or Enter to play")
    expect(page.locator("canvas")).to_have_attribute("width", "160")
    expect(page.locator("canvas")).to_have_attribute("height", "144")
    expect(page.locator("#release")).to_have_text("Functional 2")
    output = os.environ.get("TASK_WORKSPACE")
    if output:
        page.screenshot(path=str(Path(output) / "functional-2-title.png"))

    start = len(frames)
    page.keyboard.press("Enter")
    expect(status).to_have_text(re.compile(r"^[1-9]\d* neural steps"))
    page.wait_for_function("Number(document.querySelector('#status').textContent.split(' ')[0]) >= 45",
                           timeout=10000)
    gameplay = [(seen, frame) for seen, frame in frames[start:] if not frame["menu"]]
    assert len(gameplay[0][1]["frame"]) == 30720
    first, last = gameplay[0], gameplay[-1]
    rate = (last[1]["steps"] - first[1]["steps"]) / (last[0] - first[0])
    repeats = sum("frame" not in frame for _, frame in gameplay)
    assert 12 <= rate <= 17, rate
    assert repeats > 0
    inference = median(frame["inference_ms"] for _, frame in gameplay[1:])
    print(f"V2 {rate:.1f} steps/s, median {inference:.1f} ms inference, {repeats}/{len(gameplay)} duplicate frames omitted")

    start = len(frames)
    page.keyboard.press("Enter")
    # Wait until the pause command produces a changed screen followed by an identical tick.
    for _ in range(30):
        page.wait_for_timeout(67)
        updates = [frame for _, frame in frames[start:]]
        if len(updates) >= 2 and any("frame" in frame for frame in updates[:-1]) and "frame" not in updates[-1]:
            break
    else:
        raise AssertionError("in-game pause did not render")

    page.get_by_role("button", name="Pause", exact=True).click()
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.get_by_role("button", name="Restart").click()
    expect(status).to_have_text("Press Z or Enter to play")
    page.get_by_role("button", name="Resume").click()
    page.keyboard.press("z")
    expect(status).to_contain_text("neural steps")
    page.keyboard.down("ArrowDown")
    try:
        expect(status).to_have_text("Press Z or Enter to play", timeout=90000)
    finally:
        page.keyboard.up("ArrowDown")
    page.keyboard.press("z")
    expect(status).to_contain_text("neural steps")
    if output:
        page.screenshot(path=str(Path(output) / "functional-2-play.png"))


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL"))
        page = browser.new_page()
        frames = capture_stream(page)
        page.goto(os.environ.get("GAME_URL", "http://127.0.0.1:8765/"))
        check(page, frames)
        browser.close()
