import os

from playwright.sync_api import expect, sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL"))
    page = browser.new_page()
    page.goto(os.environ.get("GAME_URL", "http://127.0.0.1:8765/"))
    expect(page.locator("#status")).to_contain_text("neural steps")
    page.get_by_role("button", name="Pause", exact=True).click()
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.get_by_role("button", name="Restart").click()
    expect(page.locator("#status")).to_contain_text("0 neural steps")
    page.get_by_role("button", name="Resume").click()
    with page.expect_response(lambda response: response.url.endswith("/step")
                              and response.request.post_data_json.get("action") == "a") as pulse:
        page.keyboard.press("z")
    assert pulse.value.ok
    assert len(pulse.value.json()["frame"]) == 7680
    browser.close()
