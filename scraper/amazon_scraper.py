import time
import random
from playwright.sync_api import sync_playwright
import re


class AmazonScraper:
    def __init__(self, brand, max_products=5, max_reviews=20):
        self.brand = brand
        self.max_products = max_products
        self.max_reviews = max_reviews
        self.base_url = "https://www.amazon.in"

    def _random_delay(self):
        time.sleep(random.uniform(1, 3))

    def extract_review_count(self, page):
        try:
            text = page.locator("#acrCustomerReviewText").inner_text()
            return self.clean_number(text)
        except:
            try:
                alt = page.locator("span[data-hook='total-review-count']").inner_text()
                return self.clean_number(alt)
            except:
                return None

    def clean_number(self, text):
        if not text:
            return None

        text = text.replace(",", "")

        try:
            return int(float(re.sub(r"[^\d.]", "", text)))
        except:
            return None

    def search_products(self, page):
        query = f"{self.brand} luggage"
        url = f"{self.base_url}/s?k={query.replace(' ', '+')}"
        page.goto(url)
        self._random_delay()

    def extract_product_links(self, page):
        links = page.locator("div.s-main-slot a.a-link-normal.s-no-outline").all()

        product_links = []
        for link in links:
            href = link.get_attribute("href")

            if href and "/dp/" in href:  # only real products
                clean_url = href.split("?")[0]
                product_links.append(self.base_url + clean_url)

            if len(product_links) >= self.max_products:
                break

        return product_links

    def extract_price(self, page):
        try:
            price_text = page.locator(".a-price .a-offscreen").first.inner_text()
            return price_text  # already formatted like ₹1,999.00
        except:
            return None

    def extract_mrp(self, page):
        try:
            return page.locator("span.a-text-price span.a-offscreen").first.inner_text()
        except:
            return None

    def scrape_product_details(self, page, url):
        page.goto(url)
        self._random_delay()

        def safe_text(selector):
            try:
                return page.locator(selector).first.inner_text()
            except:
                return None

        review_count = self.extract_review_count(page)

        # sanity check
        if review_count and review_count > 5000:
            print("[WARNING] Suspicious review count:", review_count)

        product = {
            "brand": self.brand,
            "title": safe_text("#productTitle"),
            "price": self.clean_number(self.extract_price(page)),
            "mrp": self.clean_number(self.extract_mrp(page)),
            "review_count": review_count,
            "rating": safe_text("span.a-icon-alt"),
            "url": url,
        }

        print("[TITLE]", product["title"][:40])
        print("[REVIEWS RAW]", safe_text("#acrCustomerReviewText"))

        product["reviews"] = self.scrape_reviews(page, url)

        return product

    def scrape_reviews(self, page, product_url):
        reviews = []

        try:
            print("[INFO] Attempting to load reviews natively...")
            
            # Step 1: Look for the "See all reviews" button on the product page
            see_all_link = page.locator("a[data-hook='see-all-reviews-link-foot']")
            
            if see_all_link.count() > 0:
                # Emulate a human click and wait for the page to load
                see_all_link.click()
            else:
                # Fallback: Construct the URL with tracking parameters to look more organic
                asin_match = re.search(r"/dp/([A-Z0-9]{10})", product_url)
                if not asin_match:
                    print("[ASIN NOT FOUND]")
                    return []
                
                asin = asin_match.group(1)
                organic_url = f"{self.base_url}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
                page.goto(organic_url)

            # Step 2: Explicitly wait for the main review list container
            try:
                # Wait for the parent wrapper (#cm_cr-review_list) 
                # Increased timeout to 20 seconds to account for heavy DOM rendering
                page.wait_for_selector("#cm_cr-review_list", timeout=20000)
                
                # Small human-like pause after the container loads to ensure child nodes populate
                time.sleep(1) 
            except Exception:
                print(f"[REVIEW ERROR] Timeout waiting for review container. Check debug_screenshot.png")
                page.screenshot(path="debug_screenshot.png")
                return []

            print("[REVIEW PAGE URL]", page.url)

            # Step 3: Grab reviews
            review_blocks = page.locator("[data-hook='review']")
            count = review_blocks.count()
            print(f"[REVIEWS FOUND] {count}")

            for i in range(min(count, self.max_reviews)):
                r = review_blocks.nth(i)
                try:
                    # Use a short timeout here so one bad block doesn't hang the script
                    text = r.locator("span[data-hook='review-body']").inner_text(timeout=2000)
                    if text:
                        reviews.append({"text": text.strip()})
                except Exception:
                    continue

            return reviews

        except Exception as e:
            print("[REVIEW ERROR]", e)
            return []

    def run(self):
        all_products = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)

            context = browser.new_context(
                storage_state="auth.json",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 800}
            )

            page = context.new_page()
            page.goto("https://www.amazon.in")
            print("[CHECK LOGIN] Page title:", page.title())
            self.search_products(page)
            links = self.extract_product_links(page)

            print(f"[INFO] Found {len(links)} products for {self.brand}")

            for link in links:
                try:
                    product = self.scrape_product_details(page, link)
                    all_products.append(product)
                except Exception as e:
                    print(f"[ERROR] Failed for {link}: {e}")

            browser.close()

        return all_products


if __name__ == "__main__":
    scraper = AmazonScraper("Safari", max_products=3)
    data = scraper.run()

    from pprint import pprint
    pprint(data)