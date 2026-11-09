import os

from playwright.sync_api import expect, sync_playwright
from functional_two_browser import capture_stream, check


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL"))
    page = browser.new_page()
    url = os.environ.get("GAME_URL", "http://127.0.0.1:8765/").rstrip("/")
    page.goto(url)
    expect(page.get_by_role("link", name="V2", exact=True)).to_have_attribute("aria-current", "page")
    page.get_by_role("link", name="V1", exact=True).click()
    expect(page).to_have_url(url + "/v1")
    expect(page.locator("#release")).to_have_text("Functional 1")
    expect(page.get_by_role("link", name="V1", exact=True)).to_have_attribute("aria-current", "page")
    expect(page.locator("#status")).to_contain_text("neural steps")
    held_steps = []
    with page.expect_request("**/step"):
        page.route("**/step", lambda route: held_steps.append(route))
    page.get_by_role("button", name="Pause", exact=True).click()
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.get_by_role("button", name="Restart").click()
    assert len(held_steps) == 1
    held_steps[0].continue_()
    page.unroute("**/step")
    expect(page.locator("#status")).to_contain_text("0 neural steps")
    page.get_by_role("button", name="Resume").click()
    with page.expect_response(lambda response: response.url.endswith("/step")
                              and response.request.post_data_json.get("action") == "a") as pulse:
        page.keyboard.press("z")
    assert pulse.value.ok
    assert len(pulse.value.json()["frame"]) == 7680
    frames = capture_stream(page)
    page.get_by_role("link", name="V2", exact=True).click()
    expect(page).to_have_url(url + "/")
    check(page, frames)
    browser.close()
