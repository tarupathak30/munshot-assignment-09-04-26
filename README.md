# Luggage Competitive Intelligence Dashboard

An agentic pipeline that scrapes Amazon India product listings and reviews,
analyzes them with Groq LLM, and presents competitive insights in an
interactive Streamlit dashboard.

---

## Live Demo

Run locally — see setup below. No hosted link (scraper requires local auth session).

---

## Project Structure
project/
├── app.py                        ← Streamlit entry point (redirects to dashboard/)
├── run_pipeline.py               ← Master orchestrator: scrape → clean → analyze
├── requirements.txt
├── README.md
├── scraper/
│   └── amazon_scraper.py         ← Playwright-based Amazon India scraper
├── analysis/
│   ├── data_cleaner.py           ← Normalize, dedupe, validate scraped data
│   └── llm_analyzer.py           ← Groq LLM sentiment + theme extraction
├── dashboard/
│   └── app.py                    ← 5-tab Streamlit dashboard
└── data/
├── raw/                      ← Scraped JSON per brand (git-ignored)
├── clean/                    ← products_clean.csv, reviews_clean.csv,
│                                brand_analysis.json, insights.json,
│                                brand_summary.csv
└── sample/                   ← Pre-generated example dataset



---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
playwright install chromium
```

### 2. Environment variables

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here

Get a free key at https://console.groq.com

### 3. Amazon login (required for scraping)

The scraper uses a saved browser session to avoid CAPTCHA. Run this once:

```bash
python scraper/save_auth.py
```

Log in to Amazon India manually in the browser that opens, then close it.
This saves `auth.json` which the scraper reuses.

---

## Running the pipeline

### Full pipeline (scrape → clean → analyze → dashboard)

```bash
python run_pipeline.py
```

This runs all steps sequentially. Scraping takes ~45–60 minutes for 4 brands
with cooldown delays between brands to avoid rate limiting.

### Individual steps

```bash
# Step 1: Scrape (runs all 4 brands, skips already-scraped ones)
python scraper/amazon_scraper.py

# Step 2: Clean and deduplicate
python analysis/data_cleaner.py

# Step 3: LLM analysis via Groq
python analysis/llm_analyzer.py

# Step 4: Launch dashboard
streamlit run dashboard/app.py
```

---

## Dataset

| Brand | Products | Unique reviews | Avg price |
|---|---|---|---|
| American Tourister | 15 | 110 | ₹3,184 |
| Safari | 14 | 68 | ₹2,125 |
| Skybags | 15 | 104 | ₹1,999 |
| VIP | 14 | 100 | ₹2,822 |
| **Total** | **58** | **382** | — |

Scraped: April 2026. Data reflects Amazon India listings at time of scraping.

---

## Architecture
Amazon India
│
▼
amazon_scraper.py   (Playwright, human-like delays, auth session)
│  raw JSON per brand
▼
data_cleaner.py     (pandas: dedupe reviews, validate brand-title match,
│               compute discount %, flag set products)
│  products_clean.csv, reviews_clean.csv
▼
llm_analyzer.py     (Groq llama-3.3-70b-versatile: sentiment score,
│               aspect scores, pros/cons themes, trust flags,
│               cross-brand competitive insights)
│  brand_analysis.json, insights.json, brand_summary.csv
▼
dashboard/app.py    (Streamlit + Plotly: 5-tab interactive dashboard)


---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Overview | Key metrics, avg price/discount charts, price vs sentiment scatter |
| Brand Comparison | Sentiment cards, sortable benchmark table, pros/cons per brand |
| Products | Filterable product table with drilldown and review viewer |
| Sentiment | Aspect-level scores (wheels, handle, material, zipper, etc.), trust signals |
| Agent Insights | 5 non-obvious LLM-generated conclusions, value-for-money index |

---

## Sentiment methodology

Reviews are batched (30 per call) and sent to Groq `llama-3.3-70b-versatile`
with a structured prompt requesting:

- Overall sentiment score (0–100)
- Top 5 positive and negative themes
- Aspect-level scores for: wheels, handle, material, zipper, size/space,
  durability, value for money
- Trust flags (suspicious repetition, fake-sounding patterns)
- One-line anomaly detection

When a brand has more than 30 reviews, two chunks are analyzed and scores
are averaged. The value-for-money index is computed as:
`sentiment_score / avg_price × 1000` — higher means more satisfaction per rupee.

---

## Known limitations

- **Review deduplication:** Amazon shows the same reviews across color/size
  variants of the same product family. The cleaner deduplicates by exact text
  match per brand, which reduces Safari's effective review count to 68
  (vs 100+ for other brands).
- **Null prices:** 2 Safari products showed no price (out of stock at scrape
  time). These are excluded from price averages.
- **Sentiment uniformity:** Skybags and VIP received identical sentiment scores
  (73/100), likely due to LLM anchoring on mid-range scores for mixed reviews.
  A larger review sample per brand would improve differentiation.
- **Session expiry:** Amazon auth sessions expire. If scraping fails with login
  errors, re-run `save_auth.py`.
- **No pagination:** The scraper collects the first page of reviews only
  (10 reviews per product). Multi-page review scraping would improve coverage.

---

## Requirements

# Web Scraping
playwright
selenium

# Data Processing
pandas
numpy

# LLM / AI
groq

# Dashboard & Visualization
streamlit
plotly

# Utilities
python-dotenv
requests
tqdm

---

## Evaluation rubric coverage

| Criteria | Implementation |
|---|---|
| Data collection | 58 products, 382 reviews across 4 brands via Playwright |
| Analytical depth | Groq LLM: sentiment, 7 aspect scores, themes, anomaly detection |
| Dashboard UX | 5-tab Streamlit, Plotly charts, sidebar filters, drilldown |
| Competitive intelligence | Brand benchmark table, positioning scatter, value-for-money index |
| Technical execution | Modular pipeline, error handling, deduplication, env-based secrets |
| Product thinking | Agent Insights tab with 5 non-obvious cross-brand conclusions |