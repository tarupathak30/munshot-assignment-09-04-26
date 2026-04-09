"""
Competitive Intelligence Dashboard
Luggage Brands on Amazon India
Pages: Overview | Brand Comparison | Product Drilldown | Agent Insights
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Path Setup ───────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "clean"
RAW_DIR = PROJECT_DIR / "data" / "raw"

FEATURED_CSV = DATA_DIR / "featured_dataset.csv"
BRAND_ANALYSIS_JSON = DATA_DIR / "brand_analysis.json"
INSIGHTS_JSON = DATA_DIR / "agent_insights.json"
CLEAN_CSV = DATA_DIR / "products_clean.csv"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Luggage Brand Intelligence",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
  }
  .sub-header {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1.2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
  }
  .metric-label {
    font-size: 0.85rem;
    opacity: 0.9;
  }
  .insight-card {
    background: #f8f9ff;
    border-left: 4px solid #667eea;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    color: #333;
  }
  .brand-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 5px;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.5rem 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Brand Colors ─────────────────────────────────────────────────────────────
BRAND_COLORS = {
    "Safari": "#FF6B6B",
    "VIP": "#4ECDC4",
    "Skybags": "#45B7D1",
    "American Tourister": "#96CEB4",
}

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    """Load all analysis data with caching."""
    df = pd.DataFrame()
    brand_analysis = {}
    insights = []

    # Load featured dataset (with sentiment)
    if FEATURED_CSV.exists():
        df = pd.read_csv(FEATURED_CSV)
        # Parse JSON columns back to lists
        for col in ["positive_themes", "negative_themes"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else []
                )
    elif CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        df["sentiment_score"] = 0.65
        df["value_score"] = 0.5
        df["positive_themes"] = [[] for _ in range(len(df))]
        df["negative_themes"] = [[] for _ in range(len(df))]

    # Load brand analysis
    if BRAND_ANALYSIS_JSON.exists():
        with open(BRAND_ANALYSIS_JSON) as f:
            brand_analysis = json.load(f)

    # Load insights
    if INSIGHTS_JSON.exists():
        with open(INSIGHTS_JSON) as f:
            insights = json.load(f)

    # Fill missing columns
    for col in ["price_tier", "rating_tier", "sentiment_category"]:
        if col not in df.columns:
            df[col] = "Unknown"

    if "review_count_actual" not in df.columns and "review_count" in df.columns:
        df["review_count_actual"] = df["review_count"] // 10

    return df, brand_analysis, insights


def get_filtered_df(df, brands=None, price_range=None, rating_range=None, sentiment_range=None):
    """Apply sidebar filters to dataframe."""
    filtered = df.copy()

    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]

    if price_range:
        filtered = filtered[
            (filtered["price"] >= price_range[0]) & (filtered["price"] <= price_range[1])
        ]

    if rating_range:
        filtered = filtered[
            (filtered["rating"] >= rating_range[0]) & (filtered["rating"] <= rating_range[1])
        ]

    if sentiment_range:
        filtered = filtered[
            (filtered["sentiment_score"] >= sentiment_range[0])
            & (filtered["sentiment_score"] <= sentiment_range[1])
        ]

    return filtered


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(df):
    """Render sidebar with filters."""
    st.sidebar.markdown("## 🎛️ Filters")
    st.sidebar.markdown("---")

    all_brands = sorted(df["brand"].dropna().unique().tolist())
    selected_brands = st.sidebar.multiselect(
        "Brands",
        options=all_brands,
        default=all_brands,
        help="Select one or more brands to filter",
    )

    st.sidebar.markdown("### Price Range (₹)")
    min_price = int(df["price"].min())
    max_price = int(df["price"].max())
    price_range = st.sidebar.slider(
        "Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=100,
        format="₹%d",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### Rating")
    rating_range = st.sidebar.slider(
        "Rating Range",
        min_value=1.0,
        max_value=5.0,
        value=(3.0, 5.0),
        step=0.1,
        format="%.1f ⭐",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### Sentiment Score")
    sentiment_range = st.sidebar.slider(
        "Sentiment",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")

    if FEATURED_CSV.exists():
        st.sidebar.success("✅ Analysis complete")
    elif CLEAN_CSV.exists():
        st.sidebar.warning("⚠️ Analysis pending")
    else:
        st.sidebar.error("❌ No data found")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How to run analysis:**")
    st.sidebar.code("python run_pipeline.py", language="bash")

    return selected_brands, price_range, rating_range, sentiment_range


# ─── Page 1: Overview ─────────────────────────────────────────────────────────

def page_overview(df, brand_analysis, filtered_df):
    st.markdown('<p class="main-header">🧳 Luggage Brand Intelligence Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Competitive analysis of Amazon India\'s top luggage brands</p>', unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🏷️ Brands", filtered_df["brand"].nunique())
    with col2:
        st.metric("📦 Products", len(filtered_df))
    with col3:
        total_reviews = int(filtered_df["review_count"].sum()) if "review_count" in filtered_df.columns else 0
        st.metric("💬 Reviews", f"{total_reviews:,}")
    with col4:
        avg_sentiment = filtered_df["sentiment_score"].mean() if "sentiment_score" in filtered_df.columns else 0
        st.metric("😊 Avg Sentiment", f"{avg_sentiment:.2f}")
    with col5:
        avg_rating = filtered_df["rating"].mean() if "rating" in filtered_df.columns else 0
        st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💰 Price Distribution by Brand")
        if not filtered_df.empty:
            fig_price = px.violin(
                filtered_df,
                x="brand",
                y="price",
                color="brand",
                color_discrete_map=BRAND_COLORS,
                box=True,
                points="all",
                labels={"price": "Price (₹)", "brand": "Brand"},
                title="Price Distribution",
            )
            fig_price.update_layout(
                showlegend=False,
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_price, use_container_width=True)

    with col_right:
        st.subheader("⭐ Rating Distribution")
        if not filtered_df.empty:
            fig_rating = px.histogram(
                filtered_df,
                x="rating",
                color="brand",
                color_discrete_map=BRAND_COLORS,
                nbins=20,
                labels={"rating": "Star Rating", "count": "Products"},
                title="Rating Frequency",
                barmode="overlay",
                opacity=0.75,
            )
            fig_rating.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rating, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("🎯 Sentiment vs Rating Bubble Chart")
        if not filtered_df.empty and "sentiment_score" in filtered_df.columns:
            bubble_df = filtered_df.groupby("brand").agg(
                avg_rating=("rating", "mean"),
                avg_sentiment=("sentiment_score", "mean"),
                avg_price=("price", "mean"),
                product_count=("brand", "count"),
            ).reset_index()

            fig_bubble = px.scatter(
                bubble_df,
                x="avg_rating",
                y="avg_sentiment",
                size="product_count",
                color="brand",
                color_discrete_map=BRAND_COLORS,
                text="brand",
                labels={
                    "avg_rating": "Avg Rating ⭐",
                    "avg_sentiment": "Avg Sentiment Score",
                },
                title="Brand Position: Rating vs Sentiment",
            )
            fig_bubble.update_traces(textposition="top center")
            fig_bubble.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

    with col_r2:
        st.subheader("💸 Average Price vs Discount by Brand")
        if not filtered_df.empty:
            price_disc = filtered_df.groupby("brand").agg(
                avg_price=("price", "mean"),
                avg_discount=("discount_percentage", "mean"),
            ).reset_index()

            fig_pd = px.bar(
                price_disc,
                x="brand",
                y=["avg_price", "avg_discount"],
                barmode="group",
                color_discrete_sequence=["#667eea", "#f9a825"],
                labels={"value": "Value", "brand": "Brand", "variable": "Metric"},
                title="Average Price (₹) vs Discount (%)",
            )
            fig_pd.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pd, use_container_width=True)


# ─── Page 2: Brand Comparison ─────────────────────────────────────────────────

def page_brand_comparison(df, brand_analysis, filtered_df):
    st.markdown("## 🏆 Brand Comparison")

    if not brand_analysis:
        st.warning("⚠️ Brand analysis data not available. Run the analysis pipeline first.")
        _show_basic_brand_table(filtered_df)
        return

    # ── Summary Table ──────────────────────────────────────────────────────────
    st.subheader("📊 Brand Metrics Overview")

    table_data = []
    for brand, data in brand_analysis.items():
        if not filtered_df.empty and brand not in filtered_df["brand"].values:
            continue
        table_data.append({
            "Brand": brand,
            "Products": data.get("product_count", 0),
            "Avg Price (₹)": f"₹{data.get('avg_price', 0):,.0f}",
            "Avg Discount (%)": f"{data.get('avg_discount', 0):.1f}%",
            "Avg Rating ⭐": f"{data.get('avg_rating', 0):.2f}",
            "Avg Sentiment": f"{data.get('avg_sentiment', 0):.3f}",
            "Total Reviews": f"{data.get('total_reviews', 0):,}",
        })

    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(
            table_df.set_index("Brand"),
            use_container_width=True,
            height=220,
        )

    st.markdown("---")

    # ── Bar Charts ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    brands_in_filter = filtered_df["brand"].unique().tolist() if not filtered_df.empty else list(brand_analysis.keys())

    brand_df = pd.DataFrame([
        {
            "brand": brand,
            "avg_price": data["avg_price"],
            "avg_discount": data["avg_discount"],
            "avg_rating": data["avg_rating"],
            "avg_sentiment": data["avg_sentiment"],
        }
        for brand, data in brand_analysis.items()
        if brand in brands_in_filter
    ])

    with col1:
        st.subheader("💰 Average Price Comparison")
        fig_price = px.bar(
            brand_df.sort_values("avg_price"),
            x="avg_price",
            y="brand",
            orientation="h",
            color="brand",
            color_discrete_map=BRAND_COLORS,
            labels={"avg_price": "Average Price (₹)", "brand": ""},
            text="avg_price",
        )
        fig_price.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_price.update_layout(
            showlegend=False,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.subheader("😊 Sentiment Score Comparison")
        fig_sent = px.bar(
            brand_df.sort_values("avg_sentiment"),
            x="avg_sentiment",
            y="brand",
            orientation="h",
            color="brand",
            color_discrete_map=BRAND_COLORS,
            labels={"avg_sentiment": "Avg Sentiment (0–1)", "brand": ""},
            text="avg_sentiment",
        )
        fig_sent.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_sent.update_layout(
            showlegend=False,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # ── Scatter Plot ──────────────────────────────────────────────────────────
    st.subheader("🎯 Price vs Sentiment (Product Level)")

    if not filtered_df.empty and "sentiment_score" in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="price",
            y="sentiment_score",
            color="brand",
            color_discrete_map=BRAND_COLORS,
            size="rating",
            hover_data=["product_name", "rating", "discount_percentage"],
            labels={
                "price": "Price (₹)",
                "sentiment_score": "Sentiment Score",
                "brand": "Brand",
            },
            title="Product Price vs Customer Sentiment (bubble size = rating)",
        )
        fig_scatter.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────
    st.subheader("🕸️ Brand Radar: Multi-Metric Comparison")

    if brand_df is not None and len(brand_df) > 0:
        _render_radar_chart(brand_df)

    # ── Themes Analysis ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Common Praises & Complaints by Brand")

    cols = st.columns(len([b for b in brand_analysis if b in brands_in_filter]))
    for i, (brand, data) in enumerate(brand_analysis.items()):
        if brand not in brands_in_filter:
            continue
        with cols[i]:
            color = BRAND_COLORS.get(brand, "#999")
            st.markdown(f"**{brand}**")

            praises = data.get("common_praises", [])
            if praises:
                st.markdown("👍 **Top Praises**")
                for p in praises[:3]:
                    st.markdown(f"• {p}")

            complaints = data.get("common_complaints", [])
            if complaints:
                st.markdown("👎 **Top Complaints**")
                for c in complaints[:3]:
                    st.markdown(f"• {c}")


def _render_radar_chart(brand_df):
    """Render a radar/spider chart for brand metrics."""
    # Normalize metrics to 0-1 for radar
    metrics = ["avg_price", "avg_discount", "avg_rating", "avg_sentiment"]
    labels = ["Price (inv)", "Discount", "Rating", "Sentiment"]

    fig = go.Figure()

    for _, row in brand_df.iterrows():
        brand = row["brand"]
        # Normalize each metric
        values = []
        for m in metrics:
            col_min = brand_df[m].min()
            col_max = brand_df[m].max()
            col_range = col_max - col_min if col_max > col_min else 1
            v = (row[m] - col_min) / col_range
            if m == "avg_price":
                v = 1 - v  # Invert: lower price = better score
            values.append(round(v, 3))
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=brand,
            line_color=BRAND_COLORS.get(brand, "#999"),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_basic_brand_table(df):
    if df.empty:
        return
    summary = df.groupby("brand").agg(
        Products=("product_name", "count"),
        Avg_Price=("price", "mean"),
        Avg_Discount=("discount_percentage", "mean"),
        Avg_Rating=("rating", "mean"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)


# ─── Page 3: Product Drilldown ─────────────────────────────────────────────────

def page_product_drilldown(df, brand_analysis, filtered_df):
    st.markdown("## 🔍 Product Drilldown")

    if filtered_df.empty:
        st.warning("No products match the current filters.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        available_brands = sorted(filtered_df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Select Brand", available_brands)

        brand_products = filtered_df[filtered_df["brand"] == selected_brand]
        product_names = brand_products["product_name"].tolist()

        selected_product_name = st.selectbox("Select Product", product_names)

    product_row = brand_products[brand_products["product_name"] == selected_product_name]
    if product_row.empty:
        st.warning("Product not found.")
        return

    product = product_row.iloc[0]

    with col2:
        # Product Header
        st.markdown(f"### {product['product_name']}")
        badge_color = BRAND_COLORS.get(product["brand"], "#999")
        st.markdown(
            f'<span class="brand-badge" style="background:{badge_color};color:white">'
            f'{product["brand"]}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Product Metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("💰 Price", f"₹{product['price']:,.0f}")
    with m2:
        orig = product.get("original_price", product["price"])
        st.metric("🏷️ MRP", f"₹{orig:,.0f}")
    with m3:
        disc = product.get("discount_percentage", 0)
        st.metric("🎯 Discount", f"{disc:.1f}%")
    with m4:
        st.metric("⭐ Rating", f"{product['rating']:.1f}")
    with m5:
        sentiment = product.get("sentiment_score", 0)
        st.metric("😊 Sentiment", f"{sentiment:.2f}")

    # ── Themes ───────────────────────────────────────────────────────────────
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.subheader("👍 Top Positive Themes")
        themes = product.get("positive_themes", [])
        if isinstance(themes, str):
            try:
                themes = json.loads(themes)
            except Exception:
                themes = []
        if themes:
            for t in themes[:5]:
                st.markdown(f"✅ {t}")
        else:
            st.info("Run LLM analysis to see themes")

    with col_neg:
        st.subheader("👎 Top Negative Themes")
        themes = product.get("negative_themes", [])
        if isinstance(themes, str):
            try:
                themes = json.loads(themes)
            except Exception:
                themes = []
        if themes:
            for t in themes[:5]:
                st.markdown(f"⚠️ {t}")
        else:
            st.info("Run LLM analysis to see themes")

    # ── Value Assessment ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Product Position in Brand")

    brand_products_all = df[df["brand"] == selected_brand].copy()

    if len(brand_products_all) > 1:
        col_a, col_b = st.columns(2)

        with col_a:
            # Price position
            fig_price_pos = px.scatter(
                brand_products_all,
                x="price",
                y="rating",
                color="sentiment_score" if "sentiment_score" in brand_products_all.columns else None,
                color_continuous_scale="RdYlGn",
                hover_data=["product_name"],
                title=f"{selected_brand}: Price vs Rating",
                labels={"price": "Price (₹)", "rating": "Rating ⭐"},
            )
            # Highlight selected product
            fig_price_pos.add_scatter(
                x=[product["price"]],
                y=[product["rating"]],
                mode="markers",
                marker=dict(size=16, color="red", symbol="star"),
                name="Selected Product",
                showlegend=True,
            )
            fig_price_pos.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_price_pos, use_container_width=True)

        with col_b:
            # Value score comparison
            if "value_score" in brand_products_all.columns:
                fig_value = px.bar(
                    brand_products_all.sort_values("value_score", ascending=True).tail(8),
                    x="value_score",
                    y="product_name",
                    orientation="h",
                    color="value_score",
                    color_continuous_scale="viridis",
                    title=f"{selected_brand}: Value Scores",
                    labels={"value_score": "Value Score", "product_name": ""},
                )
                fig_value.update_layout(
                    height=350,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(tickfont=dict(size=9)),
                )
                st.plotly_chart(fig_value, use_container_width=True)

    # ── Reviews Sample ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Review Text Preview")

    review_text = product.get("review_text_combined", "")
    if review_text and isinstance(review_text, str) and len(review_text) > 10:
        reviews = [r.strip() for r in review_text.split("\n") if r.strip()][:10]
        for i, rev in enumerate(reviews, 1):
            st.markdown(f"**Review {i}:** {rev}")
    else:
        st.info("Review text not available. Run the full pipeline to see reviews.")


# ─── Page 4: Agent Insights ────────────────────────────────────────────────────

def page_agent_insights(df, brand_analysis, insights):
    st.markdown("## 🤖 Agent Insights")
    st.markdown(
        "**AI-generated competitive intelligence insights** from automated analysis of "
        "pricing, ratings, sentiment, and review themes across all brands."
    )

    if not insights:
        st.warning(
            "No insights available yet. Run the analysis pipeline with a Groq API key "
            "to generate AI-powered insights.\n\n`python run_pipeline.py`"
        )

        # Show manually derived insights as placeholder
        st.subheader("📌 Example Insight Categories")
        examples = [
            ("💰 Value Analysis", "Which brand offers the best sentiment-to-price ratio?"),
            ("⭐ Rating vs Reality", "Do high star ratings always match review sentiment?"),
            ("🔧 Quality Patterns", "What are the most common product failure points?"),
            ("🏷️ Discount Strategy", "Is heavy discounting a quality signal?"),
            ("📈 Market Position", "Where are the gaps in the competitive landscape?"),
        ]
        for icon_title, desc in examples:
            st.markdown(f"**{icon_title}**: {desc}")
        return

    # ── Display Insights ────────────────────────────────────────────────────
    st.subheader(f"💡 {len(insights)} Competitive Intelligence Findings")

    icons = ["🥇", "🔍", "⚠️", "💡", "🎯"]
    for i, insight in enumerate(insights):
        icon = icons[i % len(icons)]
        st.markdown(
            f'<div class="insight-card">{icon} <b>Insight {i+1}:</b> {insight}</div>',
            unsafe_allow_html=True,
        )

    # ── Supporting Visualizations ────────────────────────────────────────────
    if brand_analysis:
        st.markdown("---")
        st.subheader("📊 Supporting Data Visualizations")

        col1, col2 = st.columns(2)

        # Value Score Chart
        with col1:
            if "value_score" in df.columns:
                value_by_brand = df.groupby("brand")["value_score"].mean().reset_index()
                fig_val = px.bar(
                    value_by_brand.sort_values("value_score"),
                    x="brand",
                    y="value_score",
                    color="brand",
                    color_discrete_map=BRAND_COLORS,
                    title="Average Value Score by Brand",
                    labels={"value_score": "Value Score (Sentiment/Price)", "brand": ""},
                )
                fig_val.update_layout(
                    showlegend=False,
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_val, use_container_width=True)

        # Sentiment Distribution
        with col2:
            if "sentiment_score" in df.columns:
                fig_hist = px.histogram(
                    df,
                    x="sentiment_score",
                    color="brand",
                    color_discrete_map=BRAND_COLORS,
                    nbins=20,
                    title="Sentiment Score Distribution",
                    labels={"sentiment_score": "Sentiment Score", "count": "Products"},
                    barmode="overlay",
                    opacity=0.7,
                )
                fig_hist.update_layout(
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # Complaint Heatmap
        st.subheader("🗺️ Top Complaints Heatmap")
        _render_complaints_heatmap(brand_analysis)


def _render_complaints_heatmap(brand_analysis):
    """Create a heatmap of complaints across brands."""
    all_complaints = set()
    for data in brand_analysis.values():
        all_complaints.update(data.get("common_complaints", [])[:5])

    all_complaints = list(all_complaints)[:10]
    brands = list(brand_analysis.keys())

    matrix = []
    for brand in brands:
        brand_complaints = brand_analysis[brand].get("common_complaints", [])
        row = [1 if c in brand_complaints else 0 for c in all_complaints]
        matrix.append(row)

    if matrix and all_complaints:
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[c[:35] for c in all_complaints],
                y=brands,
                colorscale=[[0, "#e8f5e9"], [1, "#c62828"]],
                showscale=False,
                text=[["Present" if v else "" for v in row] for row in matrix],
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Complaint Presence by Brand (Red = Common Complaint)",
            height=300,
            xaxis_tickangle=-30,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Pipeline Runner UI ──────────────────────────────────────────────────────

def page_setup():
    st.markdown("## ⚙️ Setup & Run Pipeline")
    st.markdown("""
    This page guides you through running the full data pipeline.

    ### 📋 Pipeline Steps:
    1. **Generate/Scrape Data** – Either use sample data or run the Amazon scraper
    2. **Clean Data** – Normalize and validate the dataset
    3. **LLM Analysis** – Use Groq API for sentiment & themes
    4. **Dashboard** – View insights here

    ### 🚀 Quick Start (with sample data):
    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Generate sample data + run full pipeline
    python run_pipeline.py --sample

    # Or with live scraping (requires playwright)
    playwright install chromium
    python run_pipeline.py --scrape
    ```
    """)

    st.markdown("### 🔑 Groq API Key")
    api_key = st.text_input("Enter Groq API Key (optional for demo)", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API key set for this session")

    st.markdown("### ▶️ Run Pipeline from Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                try:
                    from scraper.sample_data_generator import generate_all_sample_data
                    generate_all_sample_data()
                    st.success("✅ Sample data generated!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("🧹 Clean Data", use_container_width=True):
            with st.spinner("Cleaning data..."):
                try:
                    from analysis.data_cleaner import clean_pipeline
                    clean_pipeline()
                    st.success("✅ Data cleaned!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        if st.button("🤖 Run LLM Analysis", use_container_width=True):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("⚠️ Set GROQ_API_KEY first!")
            else:
                with st.spinner("Running LLM analysis (this may take a few minutes)..."):
                    try:
                        from analysis.data_cleaner import load_clean_data_with_reviews
                        from analysis.llm_analyzer import run_analysis_pipeline
                        df = load_clean_data_with_reviews()
                        run_analysis_pipeline(df)
                        st.success("✅ Analysis complete!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Load data
    df, brand_analysis, insights = load_data()

    # Sidebar filters
    selected_brands, price_range, rating_range, sentiment_range = render_sidebar(df)

    # Apply filters
    filtered_df = get_filtered_df(
        df,
        brands=selected_brands,
        price_range=price_range,
        rating_range=rating_range,
        sentiment_range=sentiment_range,
    )

    # Navigation tabs
    tabs = st.tabs([
        "📊 Overview",
        "🏆 Brand Comparison",
        "🔍 Product Drilldown",
        "🤖 Agent Insights",
        "⚙️ Setup",
    ])

    with tabs[0]:
        page_overview(df, brand_analysis, filtered_df)

    with tabs[1]:
        page_brand_comparison(df, brand_analysis, filtered_df)

    with tabs[2]:
        page_product_drilldown(df, brand_analysis, filtered_df)

    with tabs[3]:
        page_agent_insights(df, brand_analysis, insights)

    with tabs[4]:
        page_setup()


if __name__ == "__main__":
    main()