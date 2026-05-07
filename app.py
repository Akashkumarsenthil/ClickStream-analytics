import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import boto3
from botocore.exceptions import ClientError
import requests
import os

st.set_page_config(
    page_title="ClickStream Lakehouse Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

S3_BUCKET = "clickstream-analytics-akash"
S3_LIVE_PREFIX = "live-events/"
S3_ML_SUMMARY_KEY = "ml-artifacts/ml_summary.json"
S3_BENCHMARK_KEY = "benchmark/results.json"

API_BASE_URL = "https://decade-progress-thievish.ngrok-free.dev"

HISTORICAL_TOP_PRODUCTS = pd.DataFrame([
    {"product_id": "1004856", "category_code": "electronics.smartphone", "brand": "samsung", "historical_views": 942167, "price": 129.05},
    {"product_id": "1005115", "category_code": "electronics.smartphone", "brand": "apple", "historical_views": 910725, "price": 947.09},
    {"product_id": "1004767", "category_code": "electronics.smartphone", "brand": "samsung", "historical_views": 861675, "price": 247.06},
    {"product_id": "4804056", "category_code": "electronics.audio.headphone", "brand": "apple", "historical_views": 497431, "price": 162.00},
    {"product_id": "1005105", "category_code": "electronics.smartphone", "brand": "apple", "historical_views": 473651, "price": 1374.04},
    {"product_id": "1002544", "category_code": "electronics.smartphone", "brand": "apple", "historical_views": 409169, "price": 468.92},
])

PIPELINE_STATUS = [
    {"Layer": "Raw", "Technology": "S3 CSV", "Status": "Complete", "Output": "109.9M REES46 raw events"},
    {"Layer": "Bronze", "Technology": "Parquet", "Status": "Complete", "Output": "114.9M total rows"},
    {"Layer": "Batch", "Technology": "Apache Iceberg", "Status": "Complete", "Output": "Historical SQL and snapshots"},
    {"Layer": "Speed", "Technology": "Delta Lake", "Status": "Complete", "Output": "Trending products and hourly volume"},
    {"Layer": "Serving", "Technology": "Apache Hudi", "Status": "Complete", "Output": "15.1M user profiles"},
    {"Layer": "Gold", "Technology": "MLflow / S3", "Status": "Complete", "Output": "CTR, ALS, Product2Vec, K-Means"},
    {"Layer": "API", "Technology": "FastAPI", "Status": "Live", "Output": "Serving endpoints and live ingestion"},
]

ICON_DATABASE = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none">
<rect x="4" y="4" width="16" height="5" rx="2" stroke="#5EEAD4" stroke-width="1.8"/>
<rect x="4" y="10" width="16" height="5" rx="2" stroke="#5EEAD4" stroke-width="1.8"/>
<rect x="4" y="16" width="16" height="4" rx="2" stroke="#5EEAD4" stroke-width="1.8"/>
</svg>
"""

ICON_ACTIVITY = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none">
<path d="M4 13h4l2-6 4 12 2-6h4" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_MODEL = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="12" r="3" stroke="#C084FC" stroke-width="2"/>
<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" stroke="#C084FC" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

ICON_API = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none">
<path d="M8 9l-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_CHART = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none">
<path d="M5 19V9M12 19V5M19 19v-7" stroke="#34D399" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(168,85,247,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.1rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.68);
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 115px;
    }
    .metric-label {
        color: rgba(255,255,255,0.58);
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
    }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: rgba(34,197,94,0.14);
        color: #4ADE80;
        border: 1px solid rgba(34,197,94,0.28);
        font-size: 0.78rem;
        font-weight: 700;
    }
    .muted {
        color: rgba(255,255,255,0.60);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_s3_client():
    # Attempt to use local credentials or environment variables
    return boto3.client("s3")

def read_s3_json(key):
    try:
        obj = get_s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

def get_live_clicks(limit=250):
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_LIVE_PREFIX)
        if "Contents" not in response:
            return pd.DataFrame()

        files = [f for f in response["Contents"] if f["Key"].endswith(".json")]
        files = sorted(files, key=lambda x: x["LastModified"], reverse=True)[:limit]

        rows = []
        for f in files:
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=f["Key"])
                row = json.loads(obj["Body"].read().decode("utf-8"))
                row["_s3_key"] = f["Key"]
                row["_last_modified"] = f["LastModified"]
                rows.append(row)
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["event_time"] = pd.to_datetime(df.get("event_time"), errors="coerce", utc=True)
        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)

        for col in ["event_type", "product_id", "category_code", "brand", "user_id", "session_id", "source"]:
            if col not in df.columns:
                df[col] = "unknown"

        return df

    except Exception as e:
        st.warning(f"Could not fetch S3 live events: {e}")
        return pd.DataFrame()

def check_api_health():
    try:
        # Check health endpoint if it exists, else just root
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(f"{API_BASE_URL}/", timeout=3)
            return r.status_code == 200
        except:
            return False

def metric_card(icon, label, value, detail=""):
    st.markdown(
        f"""
        <div class="metric-card">
            {icon}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.title("ClickStream")
    st.caption("Lakehouse analytics and real-time event simulation")

    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.slider("Refresh interval", 2, 15, 5)

    st.divider()

    st.subheader("Data Sources")
    st.code(f"s3://{S3_BUCKET}/bronze/rees46/")
    st.code(f"s3://{S3_BUCKET}/batch/iceberg/")
    st.code(f"s3://{S3_BUCKET}/speed/delta/")
    st.code(f"s3://{S3_BUCKET}/serving/hudi/")
    st.code(f"s3://{S3_BUCKET}/live-events/")

    st.divider()

    if check_api_health():
        st.success("FastAPI online")
    else:
        st.error("FastAPI unavailable")

    st.caption(API_BASE_URL)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">E-Commerce Clickstream Lakehouse Dashboard</div>
        <div class="hero-subtitle">
            Real REES46 clickstream events and Criteo-style CTR data processed through
            S3 Bronze Parquet, Apache Iceberg, Delta Lake, Apache Hudi, MLflow models,
            FastAPI serving, and live GitHub Pages event ingestion.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

df_live = get_live_clicks()
ml_summary = read_s3_json(S3_ML_SUMMARY_KEY)
benchmark_results = read_s3_json(S3_BENCHMARK_KEY)

total_live_events = len(df_live)
active_users = df_live["user_id"].nunique() if not df_live.empty else 0
live_gmv = df_live["price"].sum() if not df_live.empty else 0
latest_event_time = df_live["event_time"].max() if not df_live.empty else None

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card(ICON_DATABASE, "Historical Events", "109.95M", "REES46")
with c2:
    metric_card(ICON_DATABASE, "CTR Rows", "5.0M", "Criteo-schema")
with c3:
    metric_card(ICON_MODEL, "User Profiles", "15.1M", "Hudi")
with c4:
    metric_card(ICON_ACTIVITY, "Live Events", f"{total_live_events:,}", f"{active_users} active users")
with c5:
    metric_card(ICON_CHART, "Live Demo Value", f"${live_gmv:,.2f}", "S3 live-events")

if latest_event_time is not None:
    st.caption(f"Latest live event: {latest_event_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
else:
    st.caption("Waiting for live events from GitHub Pages to arrive in S3.")

tab_live, tab_lakehouse, tab_ml, tab_benchmark, tab_events = st.tabs([
    "Live Analytics",
    "Lakehouse Layers",
    "Machine Learning",
    "Benchmarks",
    "Event Feed"
])

with tab_live:
    st.markdown('<div class="section-title">Live clickstream overlay</div>', unsafe_allow_html=True)

    if df_live.empty:
        st.info("No live events found yet. Click products on the GitHub Pages demo to generate S3 events.")
        st.code(f"aws s3 ls s3://{S3_BUCKET}/live-events/ --recursive | tail")
    else:
        left, right = st.columns([2, 1])

        with left:
            live_counts = (
                df_live.groupby(["product_id", "brand", "category_code"], dropna=False)
                .size()
                .reset_index(name="live_events")
            )

            overlay = HISTORICAL_TOP_PRODUCTS.merge(
                live_counts,
                on=["product_id", "brand", "category_code"],
                how="left"
            )
            overlay["live_events"] = overlay["live_events"].fillna(0).astype(int)
            overlay["combined_views"] = overlay["historical_views"] + overlay["live_events"]

            fig = px.bar(
                overlay.sort_values("combined_views", ascending=True),
                x="combined_views",
                y="product_id",
                color="brand",
                orientation="h",
                text="live_events",
                title="Historical baseline with live event overlay",
                labels={
                    "combined_views": "Historical views plus live events",
                    "product_id": "Product ID"
                },
                template="plotly_dark"
            )
            fig.update_traces(texttemplate="+%{text} live", textposition="outside")
            fig.update_layout(height=460, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            funnel_counts = (
                df_live["event_type"]
                .value_counts()
                .reindex(["view", "cart", "purchase"])
                .fillna(0)
                .astype(int)
            )

            fig_funnel = go.Figure(go.Funnel(
                y=["Views", "Carts", "Purchases"],
                x=[
                    int(funnel_counts.get("view", 0)),
                    int(funnel_counts.get("cart", 0)),
                    int(funnel_counts.get("purchase", 0))
                ]
            ))
            fig_funnel.update_layout(
                title="Live funnel",
                template="plotly_dark",
                height=320,
                margin=dict(l=10, r=10, t=55, b=10)
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

            top_category = df_live["category_code"].mode().iloc[0] if not df_live["category_code"].empty else "unknown"
            st.markdown(
                f"""
                <div class="metric-card">
                    {ICON_ACTIVITY}
                    <div class="metric-label">Top live category</div>
                    <div class="metric-value" style="font-size:1.05rem;">{top_category}</div>
                    <span class="pill">LIVE</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        b1, b2 = st.columns(2)

        with b1:
            brand_counts = df_live["brand"].value_counts().reset_index()
            brand_counts.columns = ["brand", "events"]
            fig_brand = px.pie(
                brand_counts,
                names="brand",
                values="events",
                hole=0.48,
                title="Live brand interest",
                template="plotly_dark"
            )
            fig_brand.update_layout(height=360)
            st.plotly_chart(fig_brand, use_container_width=True)

        with b2:
            timeline = df_live.copy()
            timeline["minute"] = timeline["event_time"].dt.floor("min")
            time_counts = timeline.groupby("minute").size().reset_index(name="events")

            fig_time = px.line(
                time_counts.sort_values("minute"),
                x="minute",
                y="events",
                markers=True,
                title="Live events over time",
                template="plotly_dark"
            )
            fig_time.update_layout(height=360)
            st.plotly_chart(fig_time, use_container_width=True)

with tab_lakehouse:
    st.markdown('<div class="section-title">Lakehouse architecture status</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame(PIPELINE_STATUS), use_container_width=True, hide_index=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown(f"""
        <div class="metric-card">
            {ICON_DATABASE}
            <div class="metric-value" style="font-size:1.25rem;">Apache Iceberg</div>
            <div class="muted">Batch analytics, metadata snapshots, historical SQL</div>
            <br><span class="pill">COMPLETE</span>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown(f"""
        <div class="metric-card">
            {ICON_ACTIVITY}
            <div class="metric-value" style="font-size:1.25rem;">Delta Lake</div>
            <div class="muted">Speed layer, transaction log, trending products</div>
            <br><span class="pill">COMPLETE</span>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown(f"""
        <div class="metric-card">
            {ICON_API}
            <div class="metric-value" style="font-size:1.25rem;">Apache Hudi</div>
            <div class="muted">Mutable user profiles and upsert-ready serving data</div>
            <br><span class="pill">COMPLETE</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Verified scale")
    scale_df = pd.DataFrame([
        {"Metric": "REES46 events processed", "Value": "109,950,743"},
        {"Metric": "Criteo-schema rows processed", "Value": "5,000,000"},
        {"Metric": "Total Bronze records", "Value": "114,950,743"},
        {"Metric": "Hudi user profiles", "Value": "15,095,144"},
        {"Metric": "Product2Vec embeddings", "Value": "37,157"},
    ])
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

with tab_ml:
    st.markdown('<div class="section-title">Machine learning layer</div>', unsafe_allow_html=True)

    if not ml_summary:
        st.warning("ML summary not found in S3.")
        st.code(f"aws s3 cp s3://{S3_BUCKET}/{S3_ML_SUMMARY_KEY} -")
    else:
        metrics = ml_summary.get("metrics", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card(ICON_MODEL, "CTR AUC-ROC", f"{metrics.get('ctr_auc_roc', 0):.4f}", "Criteo-style CTR")
        with m2:
            metric_card(ICON_MODEL, "ALS Samples", metrics.get("als_sample_recommendation_users", 0), "Recommendations")
        with m3:
            metric_card(ICON_MODEL, "Embeddings", f"{metrics.get('product_embedding_count', 0):,}", "Product2Vec")
        with m4:
            metric_card(ICON_MODEL, "Clusters", len(metrics.get("kmeans_segment_counts", {})), "K-Means")

        model_rows = [{"Model": k, "Purpose": v} for k, v in ml_summary.get("models", {}).items()]
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        segment_counts = metrics.get("kmeans_segment_counts", {})
        if segment_counts:
            seg_df = pd.DataFrame([
                {"cluster": str(k), "users": int(v)}
                for k, v in segment_counts.items()
            ])

            label_map = {
                "0": "Browsers / Low Intent",
                "1": "Cart Abandoners",
                "2": "High Intent Buyers",
                "3": "Loyal Whales",
                "4": "Window Shoppers"
            }
            seg_df["segment_label"] = seg_df["cluster"].map(label_map).fillna("Segment")

            fig_seg = px.bar(
                seg_df,
                x="segment_label",
                y="users",
                color="segment_label",
                title="Customer segment distribution",
                template="plotly_dark",
                text="users"
            )
            fig_seg.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_seg.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_seg, use_container_width=True)

with tab_benchmark:
    st.markdown('<div class="section-title">Table format benchmark</div>', unsafe_allow_html=True)

    if not benchmark_results:
        st.warning("Benchmark results not found in S3.")
        st.code(f"aws s3 cp s3://{S3_BUCKET}/{S3_BENCHMARK_KEY} -")
    else:
        bench_df = pd.DataFrame(benchmark_results)
        bench_df["avg_query_sec"] = pd.to_numeric(bench_df["avg_query_sec"])

        fig_bench = px.bar(
            bench_df.sort_values("avg_query_sec"),
            x="format",
            y="avg_query_sec",
            color="format",
            text="avg_query_sec",
            title="Average query time by table format",
            labels={"avg_query_sec": "Average query seconds", "format": "Table format"},
            template="plotly_dark"
        )
        fig_bench.update_traces(texttemplate="%{text:.2f}s", textposition="outside")
        fig_bench.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_bench, use_container_width=True)

        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.info(
            "These benchmark results are specific to this project's single-node EC2/S3 workload. "
            "They should be presented as workload-specific, not universal table-format rankings."
        )

with tab_events:
    st.markdown('<div class="section-title">Raw live event feed from S3</div>', unsafe_allow_html=True)

    if df_live.empty:
        st.info("No live events available yet.")
    else:
        show_cols = [
            "event_time",
            "event_type",
            "product_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "session_id",
            "source",
            "_s3_key"
        ]
        show_cols = [c for c in show_cols if c in df_live.columns]

        feed = df_live[show_cols].sort_values("event_time", ascending=False).copy()
        feed["event_time"] = feed["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        st.dataframe(feed, use_container_width=True, hide_index=True, height=520)

        st.download_button(
            "Download live events as CSV",
            feed.to_csv(index=False).encode("utf-8"),
            file_name="live_clickstream_events.csv",
            mime="text/csv"
        )

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()