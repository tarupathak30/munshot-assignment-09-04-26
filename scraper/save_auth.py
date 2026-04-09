from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("https://www.amazon.in")

    print("👉 Please login manually within 60 seconds...")
    page.wait_for_timeout(100000)  # 60 sec to login

    context.storage_state(path="auth.json")
    print("✅ Session saved to auth.json")

    browser.close()