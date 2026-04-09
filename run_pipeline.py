"""
run_pipeline.py
Master script to orchestrate the full pipeline:
  1. [Optional] Scrape Amazon India OR generate sample data
  2. Clean & normalize data
  3. LLM sentiment + theme analysis (Groq)
  4. Feature engineering
  5. Save analysis results for the dashboard

Usage:
  python run_pipeline.py --sample        # Use synthetic sample data (no scraping)
  python run_pipeline.py --scrape        # Live scraping (requires playwright)
  python run_pipeline.py --analyze-only  # Skip data gen, run analysis on existing data
  python run_pipeline.py --no-llm        # Skip LLM, use rule-based fallback
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# ─── Path Bootstrap ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

load_dotenv() 

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║   🧳  LUGGAGE BRAND COMPETITIVE INTELLIGENCE PIPELINE   ║
║          Amazon India | Safari · VIP · Skybags · AT     ║
╚══════════════════════════════════════════════════════════╝
    """)


def step_generate_sample_data():
    """Step 1a: Generate realistic synthetic data."""
    print("\n" + "─" * 60)
    print("STEP 1: Generating Sample Data")
    print("─" * 60)
    from scraper.sample_data_generator import generate_all_sample_data
    data = generate_all_sample_data()
    total = sum(len(v) for v in data.values())
    total_reviews = sum(len(p["reviews"]) for v in data.values() for p in v)
    print(f"\n✅ Sample data ready: {total} products, {total_reviews} reviews")
    return data


def step_scrape_data():
    """Step 1b: Live scraping from Amazon India."""
    print("\n" + "─" * 60)
    print("STEP 1: Scraping Amazon India (Live)")
    print("─" * 60)

    try:
        import asyncio
        from scraper.amazon_scraper import run_scraper
        data = asyncio.run(run_scraper())
        return data
    except ImportError:
        print("⚠️  Playwright not available. Falling back to sample data.")
        return step_generate_sample_data()
    except Exception as e:
        print(f"⚠️  Scraping failed: {e}")
        print("    Falling back to sample data...")
        return step_generate_sample_data()


def step_clean_data(raw_data=None):
    """Step 2: Clean and normalize the dataset."""
    print("\n" + "─" * 60)
    print("STEP 2: Cleaning & Normalizing Data")
    print("─" * 60)
    from analysis.data_cleaner import clean_pipeline
    df = clean_pipeline(raw_data)
    print(f"\n✅ Clean data ready: {len(df)} products")
    return df


def step_analyze_with_llm(df, skip_llm=False):
    """Step 3: LLM sentiment + theme analysis."""
    print("\n" + "─" * 60)
    print("STEP 3: LLM Analysis (Groq)")
    print("─" * 60)

    if skip_llm:
        print("⚠️  LLM analysis skipped (--no-llm flag). Using fallback analysis.")
        return _fallback_analysis(df)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️  GROQ_API_KEY not set. Using fallback rule-based analysis.")
        print("    To enable LLM: export GROQ_API_KEY='your_key_here'")
        return _fallback_analysis(df)

    try:
        from analysis.llm_analyzer import run_analysis_pipeline
        result_df, brand_analysis, insights = run_analysis_pipeline(df)
        print(f"\n✅ LLM analysis complete: {len(result_df)} products analyzed")
        return result_df, brand_analysis, insights
    except Exception as e:
        print(f"⚠️  LLM analysis error: {e}")
        print("    Falling back to rule-based analysis...")
        return _fallback_analysis(df)


def _fallback_analysis(df):
    """Rule-based fallback when Groq is not available."""
    import numpy as np
    import pandas as pd
    from analysis.llm_analyzer import (
        engineer_features,
        aggregate_brand_analysis,
        generate_fallback_insights,
        FEATURED_CSV,
        BRAND_ANALYSIS_JSON,
        INSIGHTS_JSON,
    )

    print("  Running rule-based sentiment estimation...")

    # Estimate sentiment from rating
    df = df.copy()
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = (df["rating"] - 1) / 4  # Normalize 1-5 to 0-1
        df["sentiment_score"] = df["sentiment_score"].clip(0, 1)

    # Add dummy theme columns
    df["positive_themes"] = [
        ["Good value", "Smooth wheels", "Lightweight", "Durable", "Spacious"]
        for _ in range(len(df))
    ]
    df["negative_themes"] = [
        ["Zipper quality", "Lock issues", "Heavy", "Scratches", "Warranty"]
        for _ in range(len(df))
    ]

    # Feature engineering
    df = engineer_features(df)

    # Brand aggregation
    brand_analysis = aggregate_brand_analysis(df)

    # Insights
    insights = generate_fallback_insights(brand_analysis)

    # Save
    save_df = df.drop(columns=["reviews"], errors="ignore")
    save_df["positive_themes"] = save_df["positive_themes"].apply(json.dumps)
    save_df["negative_themes"] = save_df["negative_themes"].apply(json.dumps)
    save_df.to_csv(FEATURED_CSV, index=False)

    with open(BRAND_ANALYSIS_JSON, "w") as f:
        json.dump(brand_analysis, f, indent=2)

    with open(INSIGHTS_JSON, "w") as f:
        json.dump(insights, f, indent=2)

    print(f"  ✅ Fallback analysis saved")
    return df, brand_analysis, insights


def print_summary(df, brand_analysis, insights):
    """Print a final summary of the pipeline run."""
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("═" * 60)

    print(f"\n  📦 Total Products   : {len(df)}")
    print(f"  🏷️  Total Brands     : {df['brand'].nunique()}")
    total_reviews = int(df["review_count"].sum()) if "review_count" in df.columns else 0
    print(f"  💬 Total Reviews    : {total_reviews:,}")

    if "sentiment_score" in df.columns:
        print(f"  😊 Avg Sentiment    : {df['sentiment_score'].mean():.3f}")

    print("\n  Brand Breakdown:")
    for brand in sorted(df["brand"].unique()):
        b = df[df["brand"] == brand]
        print(
            f"    {brand:22s}: {len(b):2d} products | "
            f"₹{b['price'].mean():,.0f} avg price | "
            f"{b['rating'].mean():.2f}⭐ | "
            f"{b.get('sentiment_score', b['rating']).mean():.2f} sentiment"
        )

    print(f"\n  🤖 AI Insights Generated: {len(insights)}")
    for i, ins in enumerate(insights, 1):
        print(f"    {i}. {ins[:80]}{'...' if len(ins) > 80 else ''}")

    print("\n  📂 Output Files:")
    output_files = [
        CLEAN_DIR / "products_clean.csv",
        CLEAN_DIR / "featured_dataset.csv",
        CLEAN_DIR / "brand_analysis.json",
        CLEAN_DIR / "agent_insights.json",
    ]
    for f in output_files:
        status = "✅" if f.exists() else "❌"
        print(f"    {status} {f}")

    print("\n  🚀 Launch Dashboard:")
    print("     streamlit run dashboard/app.py")
    print("\n" + "═" * 60)


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Luggage Brand Intelligence Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", action="store_true", help="Use sample data (default)")
    group.add_argument("--scrape", action="store_true", help="Live scrape Amazon India")
    group.add_argument("--analyze-only", action="store_true", help="Skip data gen, analyze existing data")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use rule-based fallback")
    args = parser.parse_args()

    start_time = time.time()
    raw_data = None

    # ── Step 1: Data Acquisition ───────────────────────────────────────────
    if args.analyze_only:
        print("\nSkipping data generation (--analyze-only mode)")
    elif args.scrape:
        raw_data = step_scrape_data()
    else:
        # Default: sample data
        raw_data = step_generate_sample_data()

    # ── Step 2: Clean ──────────────────────────────────────────────────────
    df = step_clean_data(raw_data)

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    df, brand_analysis, insights = step_analyze_with_llm(df, skip_llm=args.no_llm)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total pipeline time: {elapsed:.1f} seconds")
    print_summary(df, brand_analysis, insights)


if __name__ == "__main__":
    main()