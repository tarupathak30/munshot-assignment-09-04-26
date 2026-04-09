"""
Data Cleaning & Processing Module
Normalizes prices, removes duplicates, handles missing values,
and produces a clean CSV dataset for analysis.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CSV = CLEAN_DIR / "products_clean.csv"
CLEAN_JSON = CLEAN_DIR / "products_clean.json"


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw_data() -> Dict:
    """Load raw brand data from JSON files."""
    combined_path = RAW_DIR / "all_brands_raw.json"

    if combined_path.exists():
        with open(combined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded combined data from: {combined_path}")
        return data

    # Fall back to individual brand files
    data = {}
    brands = ["Safari", "VIP", "Skybags", "American Tourister"]
    for brand in brands:
        filename = RAW_DIR / f"{brand.lower().replace(' ', '_')}_raw.json"
        if filename.exists():
            with open(filename, "r", encoding="utf-8") as f:
                data[brand] = json.load(f)
            print(f"  Loaded {len(data[brand])} products for {brand}")
        else:
            print(f"  [WARNING] No raw data found for {brand} at {filename}")

    return data


def flatten_to_dataframe(raw_data: Dict) -> pd.DataFrame:
    """
    Flatten the nested brand→products JSON into a flat DataFrame.
    Reviews are kept as a list column initially.
    """
    rows = []
    for brand, products in raw_data.items():
        for product in products:
            row = {
                "asin": product.get("asin"),
                "product_name": product.get("product_name"),
                "brand": brand,
                "price": product.get("price"),
                "original_price": product.get("original_price"),
                "discount_percentage": product.get("discount_percentage"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count"),
                "product_url": product.get("product_url"),
                "reviews": product.get("reviews", []),
                "review_count_actual": len(product.get("reviews", [])),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Flattened to DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


# ─── Cleaning Steps ───────────────────────────────────────────────────────────

def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure price and original_price are numeric floats.
    Fill missing original_price with price (0% discount).
    """
    for col in ["price", "original_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Where original_price is missing or less than price, use price
    mask = df["original_price"].isna() | (df["original_price"] < df["price"])
    df.loc[mask, "original_price"] = df.loc[mask, "price"]

    # Recompute discount where it's missing or zero
    valid_price = df["original_price"] > 0
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
    df.loc[valid_price, "discount_percentage"] = (
        (df.loc[valid_price, "original_price"] - df.loc[valid_price, "price"])
        / df.loc[valid_price, "original_price"]
        * 100
    ).round(1)

    df["discount_percentage"] = df["discount_percentage"].fillna(0.0)
    print(f"  ✓ Prices normalized")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy for each column:
    - rating: fill with brand median, then overall median
    - review_count: fill 0
    - price: drop rows with missing price (cannot be used)
    - product_name: fill with 'Unknown Product'
    """
    before = len(df)

    # Drop rows where price is missing (unusable for analysis)
    df = df.dropna(subset=["price"])
    dropped = before - len(df)
    if dropped:
        print(f"  ✓ Dropped {dropped} rows with missing price")

    # Fill rating with brand median, then overall median
    brand_medians = df.groupby("brand")["rating"].transform("median")
    overall_median = df["rating"].median()
    df["rating"] = df["rating"].fillna(brand_medians).fillna(overall_median)
    df["rating"] = df["rating"].round(1)

    # Fill review_count
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    df["review_count_actual"] = df["review_count_actual"].fillna(0).astype(int)

    # Fill product_name
    df["product_name"] = df["product_name"].fillna("Unknown Product")

    # Fill ASIN
    df["asin"] = df["asin"].fillna("UNKNOWN_ASIN")

    print(f"  ✓ Missing values handled. Remaining rows: {len(df)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate products based on ASIN, keeping the one with most reviews."""
    before = len(df)
    df = df.sort_values("review_count_actual", ascending=False)
    df = df.drop_duplicates(subset=["asin"], keep="first")
    df = df.sort_values(["brand", "product_name"]).reset_index(drop=True)
    removed = before - len(df)
    print(f"  ✓ Removed {removed} duplicates. Remaining: {len(df)}")
    return df


def clean_text(text: str) -> str:
    """Clean product names and text fields."""
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F]", "", text)
    return text


def normalize_brand_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize brand name casing."""
    brand_map = {
        "safari": "Safari",
        "vip": "VIP",
        "skybags": "Skybags",
        "american tourister": "American Tourister",
        "americantourister": "American Tourister",
    }
    df["brand"] = df["brand"].str.strip()
    df["brand_lower"] = df["brand"].str.lower()
    df["brand"] = df["brand_lower"].map(brand_map).fillna(df["brand"])
    df = df.drop(columns=["brand_lower"])
    print(f"  ✓ Brand names normalized: {df['brand'].unique().tolist()}")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out clearly erroneous data points."""
    before = len(df)

    # Price must be positive and reasonable for luggage (₹500 – ₹50,000)
    df = df[(df["price"] >= 500) & (df["price"] <= 50000)]

    # Rating must be 1–5
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]

    # Discount must be 0–90%
    df["discount_percentage"] = df["discount_percentage"].clip(0, 90)

    removed = before - len(df)
    if removed:
        print(f"  ✓ Removed {removed} rows with out-of-range values")
    return df


def engineer_review_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean combined review text column for LLM analysis.
    Joins review list into a single string separated by newlines.
    """
    def join_reviews(reviews):
        if isinstance(reviews, list):
            return "\n".join(r for r in reviews if isinstance(r, str) and len(r) > 10)
        if isinstance(reviews, str):
            return reviews
        return ""

    df["review_text_combined"] = df["reviews"].apply(join_reviews)
    df["review_count_actual"] = df["reviews"].apply(
        lambda r: len(r) if isinstance(r, list) else 0
    )
    print(f"  ✓ Review text column created")
    return df


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def clean_pipeline(raw_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    Full data cleaning pipeline.
    Loads raw data, cleans it, and saves clean CSV/JSON.
    """
    print("\n" + "="*60)
    print("  DATA CLEANING PIPELINE")
    print("="*60)

    # 1. Load data
    if raw_data is None:
        raw_data = load_raw_data()

    if not raw_data:
        raise ValueError("No raw data available. Run the scraper first.")

    # 2. Flatten to DataFrame
    df = flatten_to_dataframe(raw_data)

    print("\nCleaning steps:")

    # 3. Normalize brand names
    df = normalize_brand_names(df)

    # 4. Normalize prices
    df = normalize_prices(df)

    # 5. Handle missing values
    df = handle_missing_values(df)

    # 6. Remove duplicates
    df = remove_duplicates(df)

    # 7. Validate ranges
    df = validate_ranges(df)

    # 8. Clean text fields
    df["product_name"] = df["product_name"].apply(clean_text)

    # 9. Engineer review text column
    df = engineer_review_text_column(df)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("CLEAN DATASET SUMMARY:")
    print(f"{'─'*40}")
    print(f"  Total products : {len(df)}")
    print(f"  Total reviews  : {df['review_count_actual'].sum()}")
    for brand in df["brand"].unique():
        b = df[df["brand"] == brand]
        print(
            f"  {brand:20s}: {len(b)} products, "
            f"avg ₹{b['price'].mean():.0f}, "
            f"avg ⭐{b['rating'].mean():.2f}"
        )

    # ─── Save ─────────────────────────────────────────────────────────────────
    # Save CSV (without list columns)
    csv_df = df.drop(columns=["reviews"], errors="ignore")
    csv_df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"\n  ✅ Clean CSV saved → {CLEAN_CSV}")

    # Save JSON (with reviews list preserved)
    records = df.to_dict(orient="records")
    with open(CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Clean JSON saved → {CLEAN_JSON}")

    return df


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset. Run clean_pipeline() first if not available."""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Clean data not found at {CLEAN_CSV}. "
            "Run: python analysis/data_cleaner.py"
        )
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded clean data: {len(df)} products")
    return df


def load_clean_data_with_reviews() -> pd.DataFrame:
    """Load clean data including reviews list (from JSON)."""
    if not CLEAN_JSON.exists():
        raise FileNotFoundError(
            f"Clean JSON not found at {CLEAN_JSON}. "
            "Run: python analysis/data_cleaner.py"
        )
    with open(CLEAN_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    print(f"Loaded clean data with reviews: {len(df)} products")
    return df


if __name__ == "__main__":
    clean_pipeline()