from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        storage_state="auth.json",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        viewport={"width": 1280, "height": 800}
    )
    page = context.new_page()
    page.goto("https://www.amazon.in/product-reviews/B0FLYJ6H77/?pageNumber=1", wait_until="domcontentloaded")
    time.sleep(4)

    # Dump all data-hook values present on the page
    hooks = page.eval_on_selector_all("[data-hook]", "els => els.map(e => e.getAttribute('data-hook'))")
    print("[DATA-HOOKS FOUND]", hooks)

    # Dump a snippet of the page HTML around reviews section
    html = page.content()
    idx = html.find("review")
    print("[HTML SNIPPET]", html[max(0, idx-100):idx+500])

    browser.close()