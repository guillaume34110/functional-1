"""Check the Functional 2 menu, native canvas and restart in a real browser."""
import os
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright


def key(page, command, button):
    with page.expect_response(lambda response: response.url.endswith("/step")
                              and response.request.post_data_json.get("action") == command) as result:
        page.keyboard.press(button)
    assert result.value.ok
    return result.value.json()


def check(page):
    status = page.locator("#status")
    expect(status).to_have_text("Press Z or Enter to play")
    expect(page.locator("canvas")).to_have_attribute("width", "160")
    expect(page.locator("canvas")).to_have_attribute("height", "144")
    expect(page.locator("#release")).to_have_text("Functional 2")
    output = os.environ.get("TASK_WORKSPACE")
    if output:
        page.screenshot(path=str(Path(output) / "functional-2-title.png"))

    play = key(page, "start", "Enter")
    assert not play["menu"] and len(play["frame"]) == 30720
    expect(status).to_have_text(re.compile(r"^(?:1[7-9]|[2-9]\d) neural steps"))
    paused = key(page, "start", "Enter")
    assert not paused["menu"]
    with page.expect_response("**/step") as waiting:
        pass
    assert waiting.value.json()["frame"] == paused["frame"]
    resumed = key(page, "start", "Enter")
    assert not resumed["menu"] and resumed["frame"] != paused["frame"]
    page.get_by_role("button", name="Restart").click()
    expect(status).to_have_text("Press Z or Enter to play")

    assert not key(page, "a", "z")["menu"]
    page.keyboard.down("ArrowDown")
    try:
        expect(status).to_have_text("Press Z or Enter to play", timeout=90000)
    finally:
        page.keyboard.up("ArrowDown")
    assert not key(page, "a", "z")["menu"]
    if output:
        page.screenshot(path=str(Path(output) / "functional-2-play.png"))


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL"))
        page = browser.new_page()
        page.goto(os.environ.get("GAME_URL", "http://127.0.0.1:8765/"))
        check(page)
        browser.close()
