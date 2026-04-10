# analysis/llm_analyzer.py
import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SENTIMENT_PROMPT = """You are analyzing Amazon India customer reviews for luggage brand: {brand}

Reviews (separated by |||):
{reviews}

Return ONLY valid JSON, no explanation, no markdown, no backticks. Exactly this structure:
{{
  "sentiment_score": <0-100 integer, 100 = all positive>,
  "top_pros": ["<theme1>", "<theme2>", "<theme3>", "<theme4>", "<theme5>"],
  "top_cons": ["<theme1>", "<theme2>", "<theme3>", "<theme4>", "<theme5>"],
  "aspect_scores": {{
    "wheels": <0-100>,
    "handle": <0-100>,
    "material": <0-100>,
    "zipper": <0-100>,
    "size_space": <0-100>,
    "durability": <0-100>,
    "value_for_money": <0-100>
  }},
  "trust_flags": ["<flag1 if any suspicious pattern, else empty list>"],
  "brand_summary": "<2 sentence summary of what customers say about this brand>",
  "anomaly": "<one sentence: any surprising finding e.g. high rating despite recurring complaint>"
}}"""

INSIGHTS_PROMPT = """You are a competitive intelligence analyst. Here is a summary of 4 luggage brands on Amazon India:

{brand_summaries}

Generate exactly 5 non-obvious, decision-ready insights a product manager or brand strategist would find valuable.
Return ONLY valid JSON:
{{
  "insights": [
    {{
      "title": "<short title>",
      "detail": "<2-3 sentence explanation with specific brand names and data points>"
    }}
  ]
}}"""




def call_groq(prompt, model="llama-3.3-70b-versatile"):  # ← change model here
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,   # ← increase this too
    )
    return response.choices[0].message.content


def parse_json_safe(text):
    text = text.strip()
    if not text:
        print("[DEBUG] Groq returned empty string!")
        raise ValueError("Empty response from Groq")
    
    print(f"[DEBUG] Raw response preview: {text[:200]}")  # add this temporarily
    
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def analyze_brand(brand, reviews_df, products_df):
    brand_reviews = reviews_df[reviews_df["brand"] == brand]["review_text"].tolist()
    
    def clean_review(text):
        import re
        text = str(text).strip()
        text = text.replace("|||", " ")           # remove our separator if it appears in text
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII (emojis, unicode stars)
        text = re.sub(r'\s+', ' ', text)          # collapse whitespace
        return text[:500]                          # cap each review at 500 chars
    
    brand_reviews = [clean_review(r) for r in brand_reviews if str(r).strip()]
    
    chunk_size = 30
    chunks = [brand_reviews[i:i+chunk_size] for i in range(0, len(brand_reviews), chunk_size)]
    chunks = chunks[:2]
    
    all_results = []
    for i, chunk in enumerate(chunks):
        reviews_text = " ||| ".join(chunk)
        prompt = SENTIMENT_PROMPT.format(brand=brand, reviews=reviews_text)
        print(f"  [GROQ] {brand} chunk {i+1}/{len(chunks)}...")
        
        try:
            raw = call_groq(prompt)
            result = parse_json_safe(raw)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {brand} chunk {i+1}: {e}")
            continue
    
    if not all_results:
        return None
    
    # Merge multiple chunks by averaging scores
    merged = all_results[0]
    if len(all_results) > 1:
        for key in ["sentiment_score"]:
            merged[key] = int(sum(r[key] for r in all_results) / len(all_results))
        for aspect in merged["aspect_scores"]:
            scores = [r["aspect_scores"][aspect] for r in all_results if aspect in r.get("aspect_scores", {})]
            if scores:
                merged["aspect_scores"][aspect] = int(sum(scores) / len(scores))
    
    # Add pricing stats from products_df (exclude sets)
    brand_products = products_df[(products_df["brand"] == brand) & (~products_df["is_set"])]
    merged["avg_price"]       = round(brand_products["price"].dropna().mean(), 0)
    merged["avg_mrp"]         = round(brand_products["mrp"].dropna().mean(), 0)
    merged["avg_discount_pct"]= round(brand_products["discount_pct"].dropna().mean(), 1)
    merged["avg_rating"]      = round(brand_products["rating"].dropna().mean(), 2)
    merged["total_reviews"]   = int(brand_products["review_count"].dropna().sum())
    merged["product_count"]   = len(brand_products)
    merged["brand"]           = brand
    
    return merged


def generate_insights(brand_results):
    # Build a text summary of all brands for the insights prompt
    summaries = []
    for r in brand_results:
        summaries.append(
            f"Brand: {r['brand']}\n"
            f"  Avg price: ₹{r['avg_price']}, Avg MRP: ₹{r['avg_mrp']}, "
            f"Avg discount: {r['avg_discount_pct']}%\n"
            f"  Avg rating: {r['avg_rating']}, Sentiment score: {r['sentiment_score']}/100\n"
            f"  Top pros: {', '.join(r['top_pros'][:3])}\n"
            f"  Top cons: {', '.join(r['top_cons'][:3])}\n"
            f"  Aspect scores: {r['aspect_scores']}\n"
            f"  Summary: {r['brand_summary']}\n"
            f"  Anomaly: {r['anomaly']}"
        )
    
    prompt = INSIGHTS_PROMPT.format(brand_summaries="\n\n".join(summaries))
    print("[GROQ] Generating competitive insights...")
    
    try:
        raw = call_groq(prompt)
        return parse_json_safe(raw)
    except Exception as e:
        print(f"[ERROR] Insights generation failed: {e}")
        return {"insights": []}


def run_analysis(clean_dir="data/clean", out_dir="data/clean"):
    products_df = pd.read_csv(f"{clean_dir}/products_clean.csv")
    reviews_df  = pd.read_csv(f"{clean_dir}/reviews_clean.csv")
    
    brands = products_df["brand"].unique().tolist()
    brand_results = []
    
    for brand in brands:
        print(f"\n[ANALYZING] {brand}...")
        result = analyze_brand(brand, reviews_df, products_df)
        if result:
            brand_results.append(result)
            print(f"  Sentiment: {result['sentiment_score']}/100, "
                  f"Avg price: ₹{result['avg_price']}")
    
    # Generate cross-brand insights
    insights = generate_insights(brand_results)
    
    # Save everything
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/brand_analysis.json", "w") as f:
        json.dump(brand_results, f, indent=2)
    print(f"\n[SAVED] brand_analysis.json")
    
    with open(f"{out_dir}/insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print(f"[SAVED] insights.json")
    
    # Also save a flat CSV version of brand stats for Streamlit
    flat = []
    for r in brand_results:
        row = {k: v for k, v in r.items() 
               if k not in ["top_pros", "top_cons", "aspect_scores", "trust_flags"]}
        row["top_pros"]  = " | ".join(r.get("top_pros", []))
        row["top_cons"]  = " | ".join(r.get("top_cons", []))
        for aspect, score in r.get("aspect_scores", {}).items():
            row[f"aspect_{aspect}"] = score
        flat.append(row)
    
    pd.DataFrame(flat).to_csv(f"{out_dir}/brand_summary.csv", index=False)
    print(f"[SAVED] brand_summary.csv")
    
    return brand_results, insights


if __name__ == "__main__":
    run_analysis()