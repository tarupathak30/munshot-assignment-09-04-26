# analysis/data_cleaner.py
import json
import pandas as pd
import os
from pathlib import Path

def load_all_brands(raw_dir="data/raw"):
    all_products = []
    for file in Path(raw_dir).glob("*.json"):
        with open(file) as f:
            products = json.load(f)
            all_products.extend(products)
    return all_products

def clean_rating(rating_str):
    # "4.2 out of 5 stars" → 4.2
    if not rating_str:
        return None
    try:
        return float(rating_str.split()[0])
    except:
        return None

def is_set_product(title):
    # Flag bundle/set products so they don't skew price avg
    keywords = ["set of", "set of 2", "set of 3", "combo", "pack of"]
    return any(k in title.lower() for k in keywords)

def deduplicate_reviews(products):
    # Reviews are shared across color variants of same ASIN family
    # Keep only unique review texts globally per brand
    seen = {}  # brand → set of review texts
    for product in products:
        brand = product["brand"]
        if brand not in seen:
            seen[brand] = set()
        unique_reviews = []
        for r in product.get("reviews", []):
            text = r["text"].strip()
            if text not in seen[brand]:
                seen[brand].add(text)
                unique_reviews.append(r)
        product["reviews"] = unique_reviews
    return products

def clean(raw_dir="data/raw", out_dir="data/clean"):
    os.makedirs(out_dir, exist_ok=True)
    products = load_all_brands(raw_dir)
    
    # Deduplicate reviews first
    products = deduplicate_reviews(products)
    
    rows = []
    for p in products:
        # Skip if no title
        if not p.get("title"):
            continue
        
        reviews = p.get("reviews", [])
        review_texts = [r["text"] for r in reviews if r.get("text", "").strip()]
        
        rows.append({
            "brand":          p["brand"],
            "title":          p["title"].strip(),
            "price":          p.get("price"),           # can be null, handle in analysis
            "mrp":            p.get("mrp"),
            "discount_pct":   round((1 - p["price"]/p["mrp"])*100, 1)
                              if p.get("price") and p.get("mrp") else None,
            "rating":         clean_rating(p.get("rating")),
            "review_count":   p.get("review_count"),
            "url":            p.get("url"),
            "is_set":         is_set_product(p.get("title", "")),
            "num_reviews_scraped": len(review_texts),
            "reviews_text":   " ||| ".join(review_texts),  # pipe-separated for LLM
        })
    
    df = pd.DataFrame(rows)
    
    # Drop rows with no price AND no MRP (completely unpriced)
    df = df[~(df["price"].isna() & df["mrp"].isna())]
    
    # Save full clean CSV
    df.to_csv(f"{out_dir}/products_clean.csv", index=False)
    print(f"[CLEAN] {len(df)} products saved → {out_dir}/products_clean.csv")
    
    # Save separate reviews file (one row per review for LLM batching)
    review_rows = []
    for p in products:
        for r in p.get("reviews", []):
            text = r["text"].strip()
            if text:
                review_rows.append({
                    "brand": p["brand"],
                    "product_url": p.get("url"),
                    "review_text": text
                })
    
    reviews_df = pd.DataFrame(review_rows)
    reviews_df.to_csv(f"{out_dir}/reviews_clean.csv", index=False)
    print(f"[CLEAN] {len(reviews_df)} reviews saved → {out_dir}/reviews_clean.csv")
    
    # Print summary per brand
    print("\n--- Per-brand summary ---")
    for brand in df["brand"].unique():
        b = df[df["brand"] == brand]
        b_solo = b[~b["is_set"]]  # exclude sets
        r = reviews_df[reviews_df["brand"] == brand]
        print(f"{brand}: {len(b)} products ({len(b_solo)} solo), "
              f"{len(r)} unique reviews, "
              f"avg price ₹{b_solo['price'].mean():.0f}")
    
    return df, reviews_df

if __name__ == "__main__":
    clean()