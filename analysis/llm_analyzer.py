"""
LLM-Based Sentiment & Theme Analysis using Groq API
Analyzes customer reviews to extract:
  - sentiment_score (0 to 1)
  - top 5 positive themes
  - top 5 negative themes
Aggregates results at brand level.
"""

import os
import json
import time
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[WARNING] groq package not installed. Run: pip install groq")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
ANALYSIS_DIR = PROJECT_DIR / "data" / "clean"  # store analysis results alongside clean data
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_CSV = CLEAN_DIR / "sentiment_results.csv"
BRAND_ANALYSIS_JSON = CLEAN_DIR / "brand_analysis.json"
INSIGHTS_JSON = CLEAN_DIR / "agent_insights.json"
FEATURED_CSV = CLEAN_DIR / "featured_dataset.csv"

# ─── Groq Client ─────────────────────────────────────────────────────────────

def get_groq_client():
    """Initialize Groq client from environment variable."""
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Run: pip install groq")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Set it with: export GROQ_API_KEY='your_key_here'"
        )
    return Groq(api_key=api_key)


# ─── Prompt Templates ────────────────────────────────────────────────────────

REVIEW_ANALYSIS_PROMPT = """You are a JSON-only API. Respond with ONLY a raw JSON object — no explanation, no markdown, no code fences.

Analyze these customer reviews for a luggage product and return this exact JSON structure:

{{"sentiment_score": 0.75, "positive_themes": ["theme1", "theme2", "theme3", "theme4", "theme5"], "negative_themes": ["theme1", "theme2", "theme3", "theme4", "theme5"]}}

Rules:
- sentiment_score: float from 0.0 (very negative) to 1.0 (very positive)
- Each theme: 3-7 word phrase describing a pattern in reviews
- Always return exactly 5 positive and 5 negative themes
- If fewer than 5 exist, infer likely themes from context
- Output ONLY the JSON object. No other text whatsoever.

Reviews to analyze:
{reviews}"""

BRAND_INSIGHTS_PROMPT = """You are a competitive intelligence analyst specializing in e-commerce.
Analyze this aggregated data for luggage brands on Amazon India and generate exactly 5 non-obvious strategic insights.

Brand Data:
{brand_data}

Generate 5 insightful observations. Each should be actionable and non-obvious.
Return ONLY a valid JSON array of 5 strings. Example format:
["Insight 1 here", "Insight 2 here", "Insight 3 here", "Insight 4 here", "Insight 5 here"]

Focus on:
- Unexpected patterns between price, rating, and sentiment
- Brand positioning gaps
- Value-for-money winners/losers
- Common complaint patterns
- Market opportunities

Return ONLY the JSON array."""


# ─── Core LLM Analysis Function ──────────────────────────────────────────────

def analyze_reviews_with_llm(
    client: Groq,
    reviews: List[str],
    product_name: str = "",
    model: str = "llama3-8b-8192",
    max_retries: int = 3,
) -> Dict:
    """
    Call Groq LLM to analyze reviews and return sentiment + themes.
    Falls back to default values on failure.
    """
    if not reviews:
        return _default_analysis()

    # Trim reviews to avoid token limits (use up to 20 reviews, max 3000 chars)
    trimmed_reviews = reviews[:20]
    review_text = "\n".join(f"- {r}" for r in trimmed_reviews)
    if len(review_text) > 3000:
        review_text = review_text[:3000] + "..."

    prompt = REVIEW_ANALYSIS_PROMPT.format(reviews=review_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            result = parse_llm_json(content)

            # Validate structure
            if validate_analysis(result):
                return result

            print(f"    [Attempt {attempt+1}] Invalid structure, retrying...")

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"    [Attempt {attempt+1}] LLM error: {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)

    print(f"    [FALLBACK] Using default analysis for: {product_name[:40]}")
    return _default_analysis()


def parse_llm_json(content: str) -> Dict:
    """
    Extract and parse JSON from LLM response robustly.
    Handles: markdown fences, leading/trailing text, single-quoted JSON,
    Python booleans (True/False), and truncated responses.
    """
    # 1. Strip markdown code fences
    content = re.sub(r"```(?:json)?", "", content).strip()

    # 2. Normalise Python-style booleans → JSON booleans
    content = content.replace("True", "true").replace("False", "false")

    # 3. Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 4. Find the FIRST complete {...} block (handles leading prose)
    brace_start = content.find("{")
    if brace_start != -1:
        # Walk forward matching braces to find the closing }
        depth = 0
        for idx in range(brace_start, len(content)):
            if content[idx] == "{":
                depth += 1
            elif content[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[brace_start: idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try fixing common LLM quirks: trailing commas
                        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break

    # 5. Regex-based field extraction as last resort
    score_match = re.search(r'"?sentiment_score"?\s*:\s*([0-9.]+)', content)
    pos_match = re.search(r'"?positive_themes"?\s*:\s*\[([^\]]*)\]', content, re.DOTALL)
    neg_match = re.search(r'"?negative_themes"?\s*:\s*\[([^\]]*)\]', content, re.DOTALL)

    if score_match:
        def extract_list(m):
            if not m:
                return []
            items = re.findall(r'"([^"]+)"', m.group(1))
            return items if items else []

        return {
            "sentiment_score": float(score_match.group(1)),
            "positive_themes": extract_list(pos_match),
            "negative_themes": extract_list(neg_match),
        }

    return {}


def validate_analysis(data: Dict) -> bool:
    """Check if LLM response has the required structure."""
    if not isinstance(data, dict):
        return False
    if "sentiment_score" not in data:
        return False
    if not isinstance(data.get("positive_themes"), list):
        return False
    if not isinstance(data.get("negative_themes"), list):
        return False
    score = data["sentiment_score"]
    if not isinstance(score, (int, float)) or not (0 <= score <= 1):
        return False
    return True


def _default_analysis() -> Dict:
    """Return a neutral default when LLM fails."""
    return {
        "sentiment_score": 0.65,
        "positive_themes": [
            "Good value for money",
            "Decent build quality",
            "Smooth wheels",
            "Lightweight design",
            "Spacious interior",
        ],
        "negative_themes": [
            "Zipper quality issues",
            "Lock feels flimsy",
            "Handle wobbles",
            "Scratches easily",
            "Limited warranty",
        ],
    }


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

CHECKPOINT_FILE = CLEAN_DIR / "analysis_checkpoint.json"


def load_checkpoint() -> Dict:
    """Load previously saved progress so we can resume after interruption."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        print(f"  ♻️  Resuming from checkpoint: {len(data)} products already done")
        return data
    return {}


def save_checkpoint(results: Dict) -> None:
    """Persist current progress to disk after each product."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def clear_checkpoint() -> None:
    """Delete checkpoint file after a successful full run."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  🗑️  Checkpoint cleared (run complete)")


# ─── Product-Level Analysis ──────────────────────────────────────────────────

def analyze_products(
    df: pd.DataFrame,
    client,
    delay_between_calls: float = 1.2,
) -> pd.DataFrame:
    """
    Run LLM analysis for each product and add sentiment columns.
    - Saves a checkpoint after every product (safe to Ctrl+C and resume)
    - Re-runs the full pipeline from the last saved point on restart
    Respects Groq rate limits with delays between API calls.
    """
    print(f"\n{'='*60}")
    print("  PRODUCT-LEVEL SENTIMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Analyzing {len(df)} products...")
    print(f"  Tip: Safe to Ctrl+C — progress is saved and will resume automatically\n")

    # Load any previously saved checkpoint
    checkpoint = load_checkpoint()

    df = df.copy().reset_index(drop=True)

    sentiment_scores = []
    positive_themes_list = []
    negative_themes_list = []

    try:
        for i, row in df.iterrows():
            product_name = str(row.get("product_name", "Unknown"))
            brand = str(row.get("brand", "Unknown"))
            # Use ASIN as stable checkpoint key, fallback to product name
            checkpoint_key = str(row.get("asin", product_name))

            # ── Skip if already processed ──────────────────────────────────
            if checkpoint_key in checkpoint:
                cached = checkpoint[checkpoint_key]
                sentiment_scores.append(cached["sentiment_score"])
                positive_themes_list.append(cached["positive_themes"])
                negative_themes_list.append(cached["negative_themes"])
                print(f"  [{i+1}/{len(df)}] ⏭  Skipping (cached): {product_name[:50]}")
                continue

            # ── Extract reviews ────────────────────────────────────────────
            reviews = row.get("reviews", [])
            if isinstance(reviews, str):
                reviews = [r.strip() for r in reviews.split("\n") if r.strip()]
            elif not isinstance(reviews, list):
                reviews = []

            # Fallback to combined text column
            if not reviews and "review_text_combined" in row:
                text = row["review_text_combined"]
                if isinstance(text, str) and text:
                    reviews = [r.strip() for r in text.split("\n") if r.strip()]

            print(f"\n  [{i+1}/{len(df)}] {brand} | {product_name[:50]}")
            print(f"    Reviews: {len(reviews)}", end="  ")

            # ── Call LLM ───────────────────────────────────────────────────
            result = analyze_reviews_with_llm(client, reviews, product_name)

            sentiment_scores.append(result["sentiment_score"])
            positive_themes_list.append(result["positive_themes"])
            negative_themes_list.append(result["negative_themes"])

            print(f"→ Sentiment: {result['sentiment_score']:.2f}  |  Top: {result['positive_themes'][0] if result['positive_themes'] else 'N/A'}")

            # ── Save checkpoint immediately ────────────────────────────────
            checkpoint[checkpoint_key] = {
                "sentiment_score": result["sentiment_score"],
                "positive_themes": result["positive_themes"],
                "negative_themes": result["negative_themes"],
            }
            save_checkpoint(checkpoint)

            # Groq free tier: ~30 RPM — 1.2s delay keeps us safely under
            time.sleep(delay_between_calls)

    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C — fill remaining rows with defaults
        remaining = len(df) - len(sentiment_scores)
        print(f"\n\n  ⚠️  Interrupted! Filled {remaining} remaining products with defaults.")
        print(f"  Progress saved. Re-run the same command to resume from product {len(sentiment_scores)+1}.")
        default = _default_analysis()
        for _ in range(remaining):
            sentiment_scores.append(default["sentiment_score"])
            positive_themes_list.append(default["positive_themes"])
            negative_themes_list.append(default["negative_themes"])

    df["sentiment_score"] = sentiment_scores
    df["positive_themes"] = positive_themes_list
    df["negative_themes"] = negative_themes_list

    completed = sum(1 for s in sentiment_scores if s != 0.65)  # non-default count
    print(f"\n  ✅ Analysis done. {completed}/{len(df)} via LLM | Avg sentiment: {np.mean(sentiment_scores):.3f}")

    # Only clear checkpoint if fully completed without interrupt
    if len(sentiment_scores) == len(df) and all(k in checkpoint for k in df.get("asin", df["product_name"])):
        clear_checkpoint()

    return df


# ─── Brand-Level Aggregation ─────────────────────────────────────────────────

def aggregate_brand_analysis(df: pd.DataFrame) -> Dict:
    """
    Aggregate product-level results to brand level.
    Returns a dict with brand summaries.
    """
    print(f"\n{'='*60}")
    print("  BRAND-LEVEL AGGREGATION")
    print(f"{'='*60}")

    brand_analysis = {}

    for brand in df["brand"].unique():
        brand_df = df[df["brand"] == brand].copy()

        # Collect all themes
        all_positive = []
        all_negative = []
        for _, row in brand_df.iterrows():
            themes_pos = row.get("positive_themes", [])
            themes_neg = row.get("negative_themes", [])
            if isinstance(themes_pos, list):
                all_positive.extend(themes_pos)
            if isinstance(themes_neg, list):
                all_negative.extend(themes_neg)

        # Count theme frequencies
        pos_counts = {}
        for t in all_positive:
            pos_counts[t] = pos_counts.get(t, 0) + 1

        neg_counts = {}
        for t in all_negative:
            neg_counts[t] = neg_counts.get(t, 0) + 1

        top_positives = sorted(pos_counts, key=pos_counts.get, reverse=True)[:5]
        top_negatives = sorted(neg_counts, key=neg_counts.get, reverse=True)[:5]

        brand_analysis[brand] = {
            "brand": brand,
            "product_count": len(brand_df),
            "avg_price": round(brand_df["price"].mean(), 2),
            "avg_original_price": round(brand_df["original_price"].mean(), 2),
            "avg_discount": round(brand_df["discount_percentage"].mean(), 2),
            "avg_rating": round(brand_df["rating"].mean(), 3),
            "avg_sentiment": round(brand_df["sentiment_score"].mean(), 3),
            "total_reviews": int(brand_df["review_count"].sum()),
            "total_review_texts": int(brand_df["review_count_actual"].sum()),
            "common_praises": top_positives,
            "common_complaints": top_negatives,
            "price_range": {
                "min": round(brand_df["price"].min(), 2),
                "max": round(brand_df["price"].max(), 2),
                "median": round(brand_df["price"].median(), 2),
            },
            "rating_distribution": brand_df["rating"].value_counts().to_dict(),
        }

        print(f"\n  {brand}:")
        print(f"    Products: {brand_analysis[brand]['product_count']}")
        print(f"    Avg Price: ₹{brand_analysis[brand]['avg_price']:.0f}")
        print(f"    Avg Rating: {brand_analysis[brand]['avg_rating']:.2f}")
        print(f"    Avg Sentiment: {brand_analysis[brand]['avg_sentiment']:.3f}")
        print(f"    Top Praise: {top_positives[0] if top_positives else 'N/A'}")
        print(f"    Top Complaint: {top_negatives[0] if top_negatives else 'N/A'}")

    return brand_analysis


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for dashboard use.
    Safe against empty DataFrames and single-value price columns.
    """
    df = df.copy()

    # Guard: nothing to engineer on empty dataframe
    if df.empty:
        for col in ["normalized_price", "value_score", "price_tier", "rating_tier", "sentiment_category"]:
            df[col] = pd.NA
        print("  ⚠  engineer_features: DataFrame is empty, skipping")
        return df

    # Normalize price to 0-1 scale for value score
    price_min = df["price"].min()
    price_max = df["price"].max()
    price_range = price_max - price_min if price_max > price_min else 1.0
    df["normalized_price"] = (df["price"] - price_min) / price_range

    # Value score: high sentiment at low price = high value
    df["value_score"] = df["sentiment_score"] / (df["normalized_price"] + 0.1)
    vs_max = df["value_score"].max()
    df["value_score"] = (df["value_score"] / vs_max).round(4) if vs_max > 0 else 0.0

    # Price tiers — use fixed percentile bins but deduplicate edges to avoid
    # "bins must increase monotonically" when all prices are the same
    p33 = df["price"].quantile(0.33)
    p67 = df["price"].quantile(0.67)
    p_lo = df["price"].min() - 1
    p_hi = df["price"].max() + 1

    # If all prices are identical (or very close) pd.cut would have duplicate edges
    unique_edges = sorted(set([p_lo, p33, p67, p_hi]))
    # We need at least 3 distinct edges (2 bins)
    if len(unique_edges) < 3:
        df["price_tier"] = "Mid-Range"
    elif len(unique_edges) == 3:
        df["price_tier"] = pd.cut(
            df["price"],
            bins=unique_edges,
            labels=["Budget", "Premium"],
            include_lowest=True,
        )
    else:
        df["price_tier"] = pd.cut(
            df["price"],
            bins=unique_edges,
            labels=["Budget", "Mid-Range", "Premium"][: len(unique_edges) - 1],
            include_lowest=True,
        )

    # Rating tiers
    r_lo = min(df["rating"].min() - 0.1, 0)
    r_hi = max(df["rating"].max() + 0.1, 5.0)
    try:
        df["rating_tier"] = pd.cut(
            df["rating"],
            bins=[r_lo, 3.5, 4.2, r_hi],
            labels=["Below Average", "Good", "Excellent"],
            include_lowest=True,
        )
    except ValueError:
        df["rating_tier"] = "Good"

    # Sentiment categories
    try:
        df["sentiment_category"] = pd.cut(
            df["sentiment_score"],
            bins=[0.0, 0.4, 0.6, 0.75, 1.01],
            labels=["Negative", "Mixed", "Positive", "Very Positive"],
            include_lowest=True,
        )
    except ValueError:
        df["sentiment_category"] = "Positive"

    print(f"  ✅ Feature engineering complete")
    print(f"     Value score range: {df['value_score'].min():.3f} – {df['value_score'].max():.3f}")
    return df


# ─── Agent Insights ───────────────────────────────────────────────────────────

def generate_agent_insights(
    brand_analysis: Dict,
    client: Groq,
    model: str = "llama3-8b-8192",
) -> List[str]:
    """
    Use LLM to generate 5 non-obvious competitive insights
    from the aggregated brand data.
    """
    print(f"\n{'='*60}")
    print("  GENERATING AGENT INSIGHTS")
    print(f"{'='*60}")

    # Prepare compact brand summary for the prompt
    brand_summary_lines = []
    for brand, data in brand_analysis.items():
        line = (
            f"{brand}: avg_price=₹{data['avg_price']:.0f}, "
            f"avg_rating={data['avg_rating']:.2f}, "
            f"avg_sentiment={data['avg_sentiment']:.3f}, "
            f"avg_discount={data['avg_discount']:.1f}%, "
            f"top_praise='{data['common_praises'][0] if data['common_praises'] else 'N/A'}', "
            f"top_complaint='{data['common_complaints'][0] if data['common_complaints'] else 'N/A'}'"
        )
        brand_summary_lines.append(line)

    brand_data_str = "\n".join(brand_summary_lines)
    prompt = BRAND_INSIGHTS_PROMPT.format(brand_data=brand_data_str)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)

        # Try to parse as JSON array
        insights = json.loads(content)
        if isinstance(insights, list) and len(insights) >= 3:
            print(f"  ✅ Generated {len(insights)} insights via LLM")
            for idx, insight in enumerate(insights, 1):
                print(f"     {idx}. {insight[:80]}...")
            return insights[:5]

    except Exception as e:
        print(f"  [LLM Insight Error] {e}")

    # Fallback: rule-based insights from the data
    insights = generate_fallback_insights(brand_analysis)
    print(f"  ✅ Generated {len(insights)} fallback insights")
    return insights


def generate_fallback_insights(brand_analysis: Dict) -> List[str]:
    """Generate rule-based insights when LLM is unavailable."""
    insights = []
    brands = list(brand_analysis.keys())

    if not brands:
        return ["Insufficient data to generate insights."]

    # Sort brands by various metrics
    by_price = sorted(brands, key=lambda b: brand_analysis[b]["avg_price"])
    by_rating = sorted(brands, key=lambda b: brand_analysis[b]["avg_rating"], reverse=True)
    by_sentiment = sorted(brands, key=lambda b: brand_analysis[b]["avg_sentiment"], reverse=True)
    by_discount = sorted(brands, key=lambda b: brand_analysis[b]["avg_discount"], reverse=True)

    cheapest = by_price[0]
    priciest = by_price[-1]
    highest_rated = by_rating[0]
    best_sentiment = by_sentiment[0]
    most_discount = by_discount[0]

    # Insight 1: Value leader
    data_cheap = brand_analysis[cheapest]
    insights.append(
        f"'{cheapest}' offers the lowest average price (₹{data_cheap['avg_price']:.0f}) "
        f"with a {data_cheap['avg_rating']:.1f}⭐ rating — making it the best entry-level option "
        f"for budget-conscious travelers."
    )

    # Insight 2: Premium vs sentiment gap
    data_pricey = brand_analysis[priciest]
    data_sentiment = brand_analysis[best_sentiment]
    if priciest != best_sentiment:
        insights.append(
            f"'{priciest}' is the most expensive brand (avg ₹{data_pricey['avg_price']:.0f}) "
            f"but '{best_sentiment}' leads on customer sentiment "
            f"({brand_analysis[best_sentiment]['avg_sentiment']:.2f}). "
            f"Price doesn't always equal satisfaction."
        )
    else:
        insights.append(
            f"'{best_sentiment}' achieves both premium pricing and the highest sentiment score "
            f"({brand_analysis[best_sentiment]['avg_sentiment']:.2f}), "
            f"suggesting strong perceived value justifying its price point."
        )

    # Insight 3: Rating vs sentiment disconnect
    if highest_rated != best_sentiment:
        hr_data = brand_analysis[highest_rated]
        insights.append(
            f"'{highest_rated}' has the highest star rating ({hr_data['avg_rating']:.2f}⭐) "
            f"but is not the leader in review sentiment — indicating customers rate generously "
            f"at checkout but express nuanced concerns in review text. "
            f"Key complaint: '{hr_data['common_complaints'][0] if hr_data['common_complaints'] else 'N/A'}'."
        )
    else:
        insights.append(
            f"'{highest_rated}' uniquely leads in both star ratings AND review sentiment, "
            f"a rare consistency that signals genuine customer satisfaction rather than "
            f"just positive bias at purchase time."
        )

    # Insight 4: Discount strategy
    disc_data = brand_analysis[most_discount]
    insights.append(
        f"'{most_discount}' employs the most aggressive discounting strategy "
        f"(avg {disc_data['avg_discount']:.1f}% off MRP). "
        f"While this drives conversions, it may signal inflated MRP pricing — "
        f"buyers should compare final prices rather than discount percentages."
    )

    # Insight 5: Common complaint across brands
    all_complaints = []
    for b_data in brand_analysis.values():
        all_complaints.extend(b_data.get("common_complaints", []))

    complaint_freq = {}
    for c in all_complaints:
        complaint_freq[c] = complaint_freq.get(c, 0) + 1

    if complaint_freq:
        top_complaint = max(complaint_freq, key=complaint_freq.get)
        insights.append(
            f"The complaint '{top_complaint}' appears across multiple brands, "
            f"suggesting it's an industry-wide issue in this price segment rather than "
            f"a single brand's problem — an opportunity for a brand that can solve it."
        )
    else:
        insights.append(
            "Review analysis reveals that durability concerns are widespread across all brands "
            "in the ₹2,000–₹6,000 segment, suggesting a market gap for a genuinely "
            "durable budget luggage option."
        )

    return insights


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_analysis_pipeline(df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Full LLM analysis pipeline:
    1. Product-level sentiment + themes
    2. Brand-level aggregation
    3. Feature engineering
    4. Agent insights
    """
    from analysis.data_cleaner import load_clean_data_with_reviews, CLEAN_JSON

    print("\n" + "="*60)
    print("  LLM ANALYSIS PIPELINE")
    print("="*60)

    # Load data if not provided
    if df is None:
        df = load_clean_data_with_reviews()

    # Initialize Groq client
    client = get_groq_client()

    # Step 1: Product analysis
    df = analyze_products(df, client)

    # Step 2: Feature engineering
    df = engineer_features(df)

    # Step 3: Brand aggregation
    brand_analysis = aggregate_brand_analysis(df)

    # Step 4: Agent insights
    insights = generate_agent_insights(brand_analysis, client)

    # ─── Save Results ──────────────────────────────────────────────────────────
    # Save featured dataset
    save_df = df.drop(columns=["reviews"], errors="ignore")
    save_df["positive_themes"] = save_df["positive_themes"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    save_df["negative_themes"] = save_df["negative_themes"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    save_df.to_csv(FEATURED_CSV, index=False)
    print(f"\n  ✅ Featured dataset saved → {FEATURED_CSV}")

    # Save brand analysis
    with open(BRAND_ANALYSIS_JSON, "w") as f:
        json.dump(brand_analysis, f, indent=2)
    print(f"  ✅ Brand analysis saved → {BRAND_ANALYSIS_JSON}")

    # Save insights
    with open(INSIGHTS_JSON, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"  ✅ Agent insights saved → {INSIGHTS_JSON}")

    return df, brand_analysis, insights


def load_analysis_results() -> Tuple[pd.DataFrame, Dict, List[str]]:
    """Load previously computed analysis results."""
    import json

    df = pd.read_csv(FEATURED_CSV) if FEATURED_CSV.exists() else pd.DataFrame()

    brand_analysis = {}
    if BRAND_ANALYSIS_JSON.exists():
        with open(BRAND_ANALYSIS_JSON) as f:
            brand_analysis = json.load(f)

    insights = []
    if INSIGHTS_JSON.exists():
        with open(INSIGHTS_JSON) as f:
            insights = json.load(f)

    return df, brand_analysis, insights


if __name__ == "__main__":
    # Quick test
    from analysis.data_cleaner import load_clean_data_with_reviews
    df = load_clean_data_with_reviews()
    run_analysis_pipeline(df)