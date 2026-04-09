"""
Sample Data Generator
Creates realistic synthetic data for Safari, VIP, Skybags, and American Tourister
when live scraping is not available or for demo/testing purposes.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ─── Brand Profiles ────────────────────────────────────────────────────────────
BRAND_PROFILES = {
    "Safari": {
        "price_range": (2500, 8500),
        "avg_rating": 4.1,
        "rating_std": 0.4,
        "discount_range": (20, 45),
        "review_sentiment": 0.72,
        "product_lines": [
            "Safari Polycarbonate 55cm Cabin Trolley",
            "Safari Polyester 65cm Check-in Trolley",
            "Safari Hard Luggage 75cm Large Trolley",
            "Safari Tesseract 4 Wheel Trolley Bag",
            "Safari Stellar Cabin Bag Blue",
            "Safari Nitron 4 Wheel Trolley Bag",
            "Safari Scope Check-in Trolley Red",
            "Safari Korrekt 55cm Cabin Luggage",
            "Safari Fusion Hard Trolley Bag",
            "Safari Polyester 3-Piece Trolley Set",
            "Safari Thorium 4 Wheel Hard Luggage",
            "Safari Superlite Cabin Hard Case",
        ],
        "positive_themes": [
            "Good build quality for the price",
            "Smooth 360-degree spinner wheels",
            "Lightweight and easy to carry",
            "Nice color options available",
            "Sturdy handle mechanism",
        ],
        "negative_themes": [
            "Zipper quality could be better",
            "TSA lock feels flimsy",
            "Inner lining tears easily",
            "Handle wobbles after heavy use",
            "Limited warranty support",
        ],
    },
    "VIP": {
        "price_range": (3000, 9500),
        "avg_rating": 4.0,
        "rating_std": 0.45,
        "discount_range": (15, 40),
        "review_sentiment": 0.70,
        "product_lines": [
            "VIP Skybag 55cm Cabin Luggage",
            "VIP Odyssey Hard Trolley 65cm",
            "VIP Alpha Polycarbonate 75cm Luggage",
            "VIP Murano 4-Wheel Trolley Bag",
            "VIP Trevi Cabin Trolley Purple",
            "VIP Palladium Hard Luggage Black",
            "VIP Crest 360 Degree Trolley",
            "VIP Verve Polycarbonate Trolley",
            "VIP Beetle Hardside Spinner Bag",
            "VIP Blaze Hard Check-in 69cm",
            "VIP Nova Hard Cabin Trolley",
            "VIP Scope 4-Wheel Luggage Blue",
        ],
        "positive_themes": [
            "Trusted Indian brand with good reputation",
            "Durable hard shell construction",
            "Good customer service experience",
            "Value for money at this price point",
            "Spacious interior with good compartments",
        ],
        "negative_themes": [
            "Wheels not as smooth as competitors",
            "Heavy compared to similar size bags",
            "Color fades after 6 months",
            "Handle extension mechanism is stiff",
            "Locking mechanism feels cheap",
        ],
    },
    "Skybags": {
        "price_range": (2000, 7000),
        "avg_rating": 4.2,
        "rating_std": 0.35,
        "discount_range": (25, 55),
        "review_sentiment": 0.75,
        "product_lines": [
            "Skybags Sonic 55cm Cabin Trolley",
            "Skybags Rubik 65cm Hard Trolley",
            "Skybags Astro Polyester Trolley Bag",
            "Skybags Comet 4-Wheel Trolley Black",
            "Skybags Radius Cabin Hard Case",
            "Skybags Ozone Polycarbonate 75cm",
            "Skybags Mint Hard Trolley Pink",
            "Skybags Stratos Spinner Trolley",
            "Skybags Drake Hard Shell Luggage",
            "Skybags Trooper Cabin Bag Silver",
            "Skybags Matrix Hardside 55cm",
            "Skybags Fizz Softside Trolley Bag",
        ],
        "positive_themes": [
            "Trendy design and vibrant colors",
            "Excellent spinner wheel quality",
            "Lightweight polycarbonate shell",
            "Great value under budget segment",
            "TSA-approved combination lock",
        ],
        "negative_themes": [
            "Scratches easily on hard surface",
            "Limited laptop compartment space",
            "Zipper quality deteriorates with use",
            "Not fully waterproof in heavy rain",
            "Handle grip rubber peels off",
        ],
    },
    "American Tourister": {
        "price_range": (4500, 15000),
        "avg_rating": 4.3,
        "rating_std": 0.3,
        "discount_range": (30, 60),
        "review_sentiment": 0.79,
        "product_lines": [
            "American Tourister Linex 55cm Cabin Luggage",
            "American Tourister Twister Hard 65cm",
            "American Tourister Polyester 79cm Large Bag",
            "American Tourister Linex 4 Wheel Spinner",
            "American Tourister Curio Hardside Trolley",
            "American Tourister Splash Softside 79cm",
            "American Tourister iDrop Trolley Bag",
            "American Tourister Sefton Hard Cabin",
            "American Tourister Trolleycase 3-Piece",
            "American Tourister Trigard Spinner Bag",
            "American Tourister Volt Hard Luggage Blue",
            "American Tourister Crystal Hard Trolley",
        ],
        "positive_themes": [
            "Premium international brand quality",
            "Excellent warranty and after-sales service",
            "Ultra-smooth spinner wheels",
            "Strong and durable hard shell",
            "Elegant design and finish quality",
        ],
        "negative_themes": [
            "Premium pricing compared to Indian brands",
            "Heavy for its size category",
            "Some units have lock issues",
            "Spare parts difficult to find",
            "Inner fabric shows wear quickly",
        ],
    },
}

# ─── Review Templates ──────────────────────────────────────────────────────────
REVIEW_POOL = {
    "positive": [
        "Really happy with this purchase! The wheels roll so smoothly and the hard shell is very sturdy.",
        "Great product for the price. Used it for my trip to Goa and it worked perfectly.",
        "Excellent quality luggage. The TSA lock is easy to use and the zippers are smooth.",
        "Very lightweight despite being hard case. Fits perfectly in overhead cabin bin.",
        "Beautiful design and very spacious inside. The expandable feature is very helpful.",
        "Best value for money in this category. Highly recommend to anyone looking for budget luggage.",
        "Smooth 360-degree wheels make it very easy to maneuver in airports.",
        "Perfect size for cabin baggage. Passed SpiceJet, IndiGo, and Air India size checks.",
        "Build quality is impressive at this price point. Very satisfied with my purchase.",
        "Color is exactly as shown in photos. The bag looks premium and very stylish.",
        "Used for a 15-day Europe trip. The bag handled international travel without any issues.",
        "Sturdy construction with good interior organization. Packing is now much easier.",
        "The expandable zippered section gave extra 20% space which was very useful.",
        "Great gift for my parents. They loved the lightweight and easy-to-roll design.",
        "Bought two of these for a family trip. Both bags are excellent quality.",
        "Handle extends smoothly and locks at different heights. Very ergonomic design.",
        "Very impressed with the quality. The price-to-performance ratio is outstanding.",
        "Fast delivery and well-packaged. Product is exactly as described on the website.",
        "Used extensively on 3 international trips and still looks brand new.",
        "The hard shell protected my fragile items perfectly. Not a single breakage.",
    ],
    "negative": [
        "Zipper broke after just 2 uses. Very disappointing for the price paid.",
        "Wheels are already wobbling after 3 months of light use. Expected better.",
        "The lock mechanism feels very flimsy. I am worried it will break soon.",
        "Color started fading after washing. Should have better color retention.",
        "Handle gets stuck sometimes when trying to retract. Very frustrating.",
        "Inner lining tore on first use when I overpacked. Poor quality stitching.",
        "TSA lock is difficult to reset. Instructions are not clear at all.",
        "Scratches very easily even with gentle handling in airport.",
        "Heavier than expected. Not ideal if you have strict weight limits.",
        "Customer service was unhelpful when I reported the zipper issue.",
        "The expandable section zipper broke within a month.",
        "Not waterproof at all. My clothes got damp in moderate rain.",
        "Wheels make squeaking noise after first use. Very annoying.",
        "The hard case cracked when checked-in baggage was mishandled at airport.",
        "Too small compared to the dimensions listed on the product page.",
    ],
    "mixed": [
        "Good product overall but the wheels could be smoother. Satisfied for the price.",
        "Decent quality for the price range. Some minor issues with the zipper alignment.",
        "Nice design and lightweight but the lock quality is average.",
        "Good value bag. Works as expected. Nothing exceptional but no major complaints.",
        "Satisfactory product. Would have preferred a slightly bigger interior.",
        "Solid luggage for domestic travel. Might not hold up for heavy international use.",
        "Average quality. Gets the job done but don't expect premium experience.",
        "OK for the price. Wish the wheels were better quality.",
    ],
}


def generate_reviews(brand: str, product_idx: int, num_reviews: int = 15) -> List[str]:
    """Generate a realistic mix of reviews for a product."""
    profile = BRAND_PROFILES[brand]
    sentiment = profile["review_sentiment"]

    positive_count = int(num_reviews * sentiment)
    negative_count = int(num_reviews * (1 - sentiment) * 0.6)
    mixed_count = num_reviews - positive_count - negative_count

    reviews = []
    reviews.extend(random.choices(REVIEW_POOL["positive"], k=positive_count))
    reviews.extend(random.choices(REVIEW_POOL["negative"], k=max(0, negative_count)))
    reviews.extend(random.choices(REVIEW_POOL["mixed"], k=max(0, mixed_count)))

    # Add some brand-specific flavor
    branded_reviews = []
    for rev in reviews:
        if random.random() < 0.3:
            rev = f"[{brand}] " + rev
        branded_reviews.append(rev)

    random.shuffle(branded_reviews)
    return branded_reviews


def generate_brand_data(brand: str) -> List[Dict]:
    """Generate realistic product data for a brand."""
    profile = BRAND_PROFILES[brand]
    products = []

    for idx, product_name in enumerate(profile["product_lines"]):
        # Generate realistic price
        price_min, price_max = profile["price_range"]
        price = round(random.uniform(price_min, price_max), -1)  # Round to nearest 10

        # MRP is typically 20-60% higher than selling price
        mrp_multiplier = random.uniform(1.2, 1.7)
        original_price = round(price * mrp_multiplier, -1)
        discount = round((original_price - price) / original_price * 100, 1)

        # Rating with some variation
        rating = round(
            max(3.0, min(5.0, random.gauss(profile["avg_rating"], profile["rating_std"]))), 1
        )

        # Review count (popular products have more reviews)
        base_reviews = random.randint(500, 5000)
        review_count = base_reviews + (idx * random.randint(50, 200))

        # Generate actual review texts
        num_review_texts = random.randint(12, 20)
        reviews = generate_reviews(brand, idx, num_review_texts)

        product = {
            "asin": f"B0{random.randint(10000000, 99999999)}",
            "product_name": product_name,
            "brand": brand,
            "price": price,
            "original_price": original_price,
            "discount_percentage": discount,
            "rating": rating,
            "review_count": review_count,
            "product_url": f"https://www.amazon.in/dp/B0{random.randint(10000000, 99999999)}",
            "reviews": reviews,
        }
        products.append(product)

    return products


def generate_all_sample_data() -> Dict:
    """Generate sample data for all 4 brands."""
    all_data = {}

    for brand in ["Safari", "VIP", "Skybags", "American Tourister"]:
        print(f"  Generating sample data for {brand}...")
        products = generate_brand_data(brand)
        all_data[brand] = products

        # Save individual brand files
        json_path = DATA_DIR / f"{brand.lower().replace(' ', '_')}_raw.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        print(f"    {len(products)} products → {json_path}")

    # Save combined file
    combined_path = DATA_DIR / "all_brands_raw.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    total_products = sum(len(v) for v in all_data.values())
    total_reviews = sum(
        len(p["reviews"]) for v in all_data.values() for p in v
    )
    print(f"\n✅ Sample data generated:")
    print(f"   Brands: {len(all_data)}")
    print(f"   Products: {total_products}")
    print(f"   Reviews: {total_reviews}")
    print(f"   Saved to: {combined_path}")

    return all_data


if __name__ == "__main__":
    print("Generating sample data...")
    generate_all_sample_data()