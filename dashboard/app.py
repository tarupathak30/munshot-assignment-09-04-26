# dashboard/app.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Luggage Intelligence Dashboard", layout="wide")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    products  = pd.read_csv("data/clean/products_clean.csv")
    reviews   = pd.read_csv("data/clean/reviews_clean.csv")
    with open("data/clean/brand_analysis.json") as f:
        brand_analysis = json.load(f)
    with open("data/clean/insights.json") as f:
        insights = json.load(f)
    summary = pd.DataFrame([
        {**{k: v for k, v in b.items()
            if k not in ["top_pros","top_cons","aspect_scores","trust_flags"]},
         "top_pros": b.get("top_pros", []),
         "top_cons": b.get("top_cons", []),
         "aspect_scores": b.get("aspect_scores", {}),
         "trust_flags": b.get("trust_flags", []),
        } for b in brand_analysis
    ])
    return products, reviews, brand_analysis, insights, summary

products_df, reviews_df, brand_analysis, insights_data, summary_df = load_data()

BRAND_COLORS = {
    "American Tourister": "#E05C2A",
    "Safari":             "#2A7AE0",
    "Skybags":            "#1DAB72",
    "VIP":                "#C4830A",
}

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
all_brands = sorted(products_df["brand"].unique().tolist())
selected_brands = st.sidebar.multiselect("Brands", all_brands, default=all_brands)

price_min, price_max = int(products_df["price"].dropna().min()), int(products_df["price"].dropna().max())
price_range = st.sidebar.slider("Price range (₹)", price_min, price_max, (price_min, price_max))

min_rating = st.sidebar.selectbox("Min rating", [0.0, 3.5, 4.0, 4.2], index=0,
                                   format_func=lambda x: f"{x}+" if x > 0 else "Any")

# Apply filters
filtered_products = products_df[
    (products_df["brand"].isin(selected_brands)) &
    (products_df["price"].between(price_range[0], price_range[1], inclusive="both")) &
    (products_df["rating"] >= min_rating) &
    (~products_df["is_set"])
]
filtered_summary = summary_df[summary_df["brand"].isin(selected_brands)]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Luggage Competitive Intelligence")
st.caption(f"Amazon India · {len(products_df)} products · {len(reviews_df)} reviews · Groq LLM analysis")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Brand Comparison", "Products", "Sentiment", "Agent Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brands tracked",    len(selected_brands))
    c2.metric("Products analyzed", len(filtered_products))
    c3.metric("Reviews processed", len(reviews_df[reviews_df["brand"].isin(selected_brands)]))
    avg_sent = int(filtered_summary["sentiment_score"].mean()) if not filtered_summary.empty else 0
    c4.metric("Avg sentiment",     f"{avg_sent}/100")
    avg_disc = filtered_products["discount_pct"].mean()
    c5.metric("Avg discount",      f"{avg_disc:.1f}%" if not pd.isna(avg_disc) else "N/A")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Avg selling price by brand")
        price_data = (filtered_products.groupby("brand")["price"]
                      .mean().reset_index().sort_values("price", ascending=True))
        fig = px.bar(price_data, x="price", y="brand", orientation="h",
                     color="brand", color_discrete_map=BRAND_COLORS,
                     labels={"price": "Avg price (₹)", "brand": ""},
                     text=price_data["price"].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0,r=20,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Avg discount % by brand")
        disc_data = (filtered_products.groupby("brand")["discount_pct"]
                     .mean().reset_index().sort_values("discount_pct", ascending=True))
        fig2 = px.bar(disc_data, x="discount_pct", y="brand", orientation="h",
                      color="brand", color_discrete_map=BRAND_COLORS,
                      labels={"discount_pct": "Avg discount (%)", "brand": ""},
                      text=disc_data["discount_pct"].apply(lambda x: f"{x:.1f}%"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=300, margin=dict(l=0,r=20,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price vs sentiment positioning")
    if not filtered_summary.empty:
        fig3 = px.scatter(
            filtered_summary, x="avg_price", y="sentiment_score",
            size="total_reviews", color="brand",
            color_discrete_map=BRAND_COLORS,
            text="brand",
            labels={"avg_price": "Avg price (₹)", "sentiment_score": "Sentiment score (0-100)"},
            size_max=60,
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Bubble size = total review count. Top-right = high sentiment at higher price (premium justified). Top-left = high sentiment at low price (value winner).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BRAND COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    sort_by = st.selectbox("Sort brands by",
                           ["sentiment_score","avg_price","avg_discount_pct","avg_rating"],
                           format_func=lambda x: x.replace("_"," ").title())
    sorted_summary = filtered_summary.sort_values(sort_by, ascending=False)

    # Brand cards
    cols = st.columns(len(sorted_summary)) if len(sorted_summary) > 0 else st.columns(1)
    for col, (_, row) in zip(cols, sorted_summary.iterrows()):
        with col:
            sent = row["sentiment_score"]
            color = "#1DAB72" if sent >= 75 else "#C4830A" if sent >= 60 else "#E05C2A"
            st.markdown(f"""
            <div style="border:1px solid #ddd;border-radius:10px;padding:14px;text-align:center">
              <div style="font-size:16px;font-weight:600">{row['brand']}</div>
              <div style="font-size:28px;font-weight:700;color:{color};margin:6px 0">{sent}</div>
              <div style="font-size:11px;color:#888">sentiment score</div>
              <hr style="margin:8px 0">
              <div style="font-size:13px">⭐ {row['avg_rating']:.2f} avg rating</div>
              <div style="font-size:13px">₹{row['avg_price']:,.0f} avg price</div>
              <div style="font-size:13px">{row['avg_discount_pct']:.1f}% avg discount</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Side-by-side benchmark")

    table_rows = []
    for _, row in sorted_summary.iterrows():
        table_rows.append({
            "Brand":        row["brand"],
            "Avg price":    f"₹{row['avg_price']:,.0f}",
            "Avg MRP":      f"₹{row['avg_mrp']:,.0f}",
            "Discount":     f"{row['avg_discount_pct']:.1f}%",
            "Rating":       f"{row['avg_rating']:.2f} ⭐",
            "Total reviews":f"{int(row['total_reviews']):,}",
            "Sentiment":    f"{row['sentiment_score']}/100",
            "Positioning":  "Premium" if row['avg_price'] > 2800 else "Value",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pros & cons by brand")
    for _, row in sorted_summary.iterrows():
        with st.expander(f"{row['brand']} — sentiment {row['sentiment_score']}/100"):
            lc, rc = st.columns(2)
            with lc:
                st.markdown("**Top praise themes**")
                for p in row["top_pros"]:
                    st.markdown(f"✅ {p}")
            with rc:
                st.markdown("**Top complaint themes**")
                for c in row["top_cons"]:
                    st.markdown(f"⚠️ {c}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Products ({len(filtered_products)} shown)")

    sort_col = st.selectbox("Sort by", ["price","rating","discount_pct","review_count"],
                             format_func=lambda x: x.replace("_"," ").title())
    sort_asc = st.checkbox("Ascending", value=False)

    display_df = (filtered_products[["brand","title","price","mrp","discount_pct",
                                      "rating","review_count","url"]]
                  .sort_values(sort_col, ascending=sort_asc)
                  .copy())

    display_df["price"]        = display_df["price"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
    display_df["mrp"]          = display_df["mrp"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
    display_df["discount_pct"] = display_df["discount_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    display_df["rating"]       = display_df["rating"].apply(lambda x: f"{x:.1f} ⭐" if pd.notna(x) else "N/A")
    display_df["review_count"] = display_df["review_count"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    display_df.columns        = ["Brand","Title","Price","MRP","Discount","Rating","Reviews","URL"]

    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 column_config={"URL": st.column_config.LinkColumn("Link")})

    # Product drilldown
    st.divider()
    st.subheader("Product drilldown")
    brand_sel = st.selectbox("Select brand", selected_brands, key="drillbrand")
    brand_prods = filtered_products[filtered_products["brand"] == brand_sel]
    if not brand_prods.empty:
        prod_sel = st.selectbox("Select product",
                                brand_prods["title"].tolist(), key="drillprod")
        prod_row = brand_prods[brand_prods["title"] == prod_sel].iloc[0]

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Price",    f"₹{prod_row['price']:,.0f}" if pd.notna(prod_row['price']) else "N/A")
        dc2.metric("MRP",      f"₹{prod_row['mrp']:,.0f}"   if pd.notna(prod_row['mrp'])   else "N/A")
        dc3.metric("Discount", f"{prod_row['discount_pct']:.1f}%" if pd.notna(prod_row['discount_pct']) else "N/A")
        dc4.metric("Rating",   f"{prod_row['rating']:.1f} ⭐" if pd.notna(prod_row['rating']) else "N/A")

        prod_reviews = reviews_df[reviews_df["product_url"] == prod_row["url"]]
        st.markdown(f"**Reviews scraped for this product:** {len(prod_reviews)}")
        for _, r in prod_reviews.head(5).iterrows():
            st.markdown(f"> {r['review_text'][:300]}...")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Aspect-level sentiment scores")
    aspects = ["wheels","handle","material","zipper","size_space","durability","value_for_money"]

    aspect_rows = []
    for _, row in filtered_summary.iterrows():
        scores = row.get("aspect_scores", {})
        for asp in aspects:
            aspect_rows.append({
                "brand": row["brand"],
                "aspect": asp.replace("_"," ").title(),
                "score": scores.get(asp, 0)
            })
    aspect_df = pd.DataFrame(aspect_rows)

    if not aspect_df.empty:
        fig_asp = px.bar(
            aspect_df, x="aspect", y="score", color="brand",
            barmode="group", color_discrete_map=BRAND_COLORS,
            labels={"score": "Score (0-100)", "aspect": ""},
            range_y=[0, 100],
        )
        fig_asp.update_layout(height=400, legend_title="Brand")
        st.plotly_chart(fig_asp, use_container_width=True)

    st.divider()
    st.subheader("Sentiment score by brand")
    if not filtered_summary.empty:
        fig_sent = px.bar(
            filtered_summary.sort_values("sentiment_score"),
            x="sentiment_score", y="brand", orientation="h",
            color="brand", color_discrete_map=BRAND_COLORS,
            labels={"sentiment_score": "Sentiment score (0-100)", "brand": ""},
            text="sentiment_score", range_x=[0, 100],
        )
        fig_sent.update_traces(textposition="outside")
        fig_sent.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()
    st.subheader("Trust signals & anomalies")
    for _, row in filtered_summary.iterrows():
        flags = row.get("trust_flags", [])
        anomaly = row.get("anomaly", "")
        with st.expander(f"{row['brand']}"):
            if flags:
                for f in flags:
                    st.warning(f"🚩 {f}")
            else:
                st.success("No suspicious patterns detected")
            if anomaly:
                st.info(f"💡 Anomaly: {anomaly}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Agent-generated competitive insights")
    st.caption("Non-obvious conclusions derived from review synthesis + pricing data via Groq LLM")

    insights_list = insights_data.get("insights", [])
    if insights_list:
        for i, insight in enumerate(insights_list, 1):
            with st.container():
                st.markdown(f"**{i}. {insight['title']}**")
                st.markdown(insight["detail"])
                st.divider()
    else:
        st.info("No insights generated — re-run llm_analyzer.py")

    st.subheader("Value-for-money analysis")
    if not filtered_summary.empty:
        vfm_df = filtered_summary.copy()
        # Score = sentiment per rupee (normalized)
        vfm_df["vfm_score"] = (vfm_df["sentiment_score"] / vfm_df["avg_price"] * 1000).round(2)
        vfm_df = vfm_df.sort_values("vfm_score", ascending=False)

        fig_vfm = px.bar(
            vfm_df, x="brand", y="vfm_score",
            color="brand", color_discrete_map=BRAND_COLORS,
            labels={"vfm_score": "Sentiment per ₹1000 spent", "brand": ""},
            text="vfm_score",
        )
        fig_vfm.update_traces(textposition="outside")
        fig_vfm.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_vfm, use_container_width=True)
        st.caption("Higher = more customer satisfaction per rupee spent. The true value-for-money winner.")