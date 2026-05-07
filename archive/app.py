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

ICON_DATABASE = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><rect x="4" y="4" width="16" height="5" rx="2" stroke="#5EEAD4" stroke-width="1.8"/><rect x="4" y="10" width="16" height="5" rx="2" stroke="#5EEAD4" stroke-width="1.8"/><rect x="4" y="16" width="16" height="4" rx="2" stroke="#5EEAD4" stroke-width="1.8"/></svg>"""
ICON_ACTIVITY = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M4 13h4l2-6 4 12 2-6h4" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
ICON_MODEL = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" stroke="#C084FC" stroke-width="2"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" stroke="#C084FC" stroke-width="2" stroke-linecap="round"/></svg>"""
ICON_API = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M8 9l-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
ICON_CHART = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M5 19V9M12 19V5M19 19v-7" stroke="#34D399" stroke-width="2" stroke-linecap="round"/></svg>"""

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(168,85,247,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.1rem;
    }
    .hero-title { font-size: 2.15rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -0.02em; }
    .hero-subtitle { color: rgba(255,255,255,0.68); font-size: 1.02rem; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_s3_client():
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
        if "Contents" not in response: return pd.DataFrame()
        files = [f for f in response["Contents"] if f["Key"].endswith(".json")]
        files = sorted(files, key=lambda x: x["LastModified"], reverse=True)[:limit]
        rows = []
        for f in files:
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=f["Key"])
                row = json.loads(obj["Body"].read().decode("utf-8"))
                row["_event_time_parsed"] = f["LastModified"]
                rows.append(row)
            except: continue
        df = pd.DataFrame(rows)
        if df.empty: return df
        df["event_time"] = pd.to_datetime(df.get("event_time"), errors="coerce", utc=True)
        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
        return df
    except: return pd.DataFrame()

def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=5, headers={"ngrok-skip-browser-warning": "true"})
        return r.status_code == 200
    except: return False

def metric_card(icon, label, value, detail=""):
    with st.container(border=True):
        st.markdown(icon, unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        st.markdown(f"### {value}")
        if detail: st.caption(detail)

with st.sidebar:
    st.title("ClickStream")
    st.caption("Lakehouse analytics and real-time event simulation")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.slider("Refresh interval", 2, 15, 5)
    st.divider()
    st.subheader("Data Sources")
    st.code(f"s3://{S3_BUCKET}/live-events/")
    st.divider()
    if check_api_health(): st.success("FastAPI online")
    else: st.error("FastAPI unavailable")
    st.caption(API_BASE_URL)

st.markdown("""<div class="hero"><div class="hero-title">E-Commerce Clickstream Lakehouse Dashboard</div><div class="hero-subtitle">Real-time S3 ingestion from GitHub Pages, processed via Spark and MLflow.</div></div>""", unsafe_allow_html=True)

df_live = get_live_clicks()
ml_summary = read_s3_json(S3_ML_SUMMARY_KEY)
benchmark_results = read_s3_json(S3_BENCHMARK_KEY)

total_live_events = len(df_live)
active_users = df_live["user_id"].nunique() if not df_live.empty else 0
live_gmv = df_live["price"].sum() if not df_live.empty else 0
latest_event_time = df_live["event_time"].max() if not df_live.empty else None

c1, c2, c3, c4, c5 = st.columns(5)
with c1: metric_card(ICON_DATABASE, "Historical Events", "109.95M", "REES46")
with c2: metric_card(ICON_DATABASE, "CTR Rows", "5.0M", "Criteo-schema")
with c3: metric_card(ICON_MODEL, "User Profiles", "15.1M", "Hudi")
with c4: metric_card(ICON_ACTIVITY, "Live Events", f"{total_live_events:,}", f"{active_users} active users")
with c5: metric_card(ICON_CHART, "Live Demo Value", f"${live_gmv:,.2f}", "S3 live-events")

tab_live, tab_lakehouse, tab_ml, tab_benchmark, tab_events, tab_api = st.tabs([
    "Live Analytics", "Lakehouse Layers", "Machine Learning", "Benchmarks", "Event Feed", "Serving Layer"
])

with tab_live:
    if not df_live.empty:
        required_cols = ["event_time", "event_type", "product_id", "category_code", "brand", "price", "user_id", "session_id"]
        present_cols = [c for c in required_cols if c in df_live.columns]
        schema_match_pct = round(len(present_cols) / len(required_cols) * 100, 1)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Schema Match", f"{schema_match_pct}%")
        q2.metric("Unique Products", df_live["product_id"].nunique())
        q3.metric("Unique Sessions", df_live["session_id"].nunique() if "session_id" in df_live.columns else 0)
        q4.metric("Avg Event Price", f"${df_live['price'].mean():.2f}")

        left, right = st.columns([2, 1])
        with left:
            live_counts = df_live.groupby(["product_id", "brand", "category_code"], dropna=False).size().reset_index(name="live_events")
            overlay = HISTORICAL_TOP_PRODUCTS.merge(live_counts, on=["product_id", "brand", "category_code"], how="left")
            overlay["live_events"] = overlay["live_events"].fillna(0).astype(int)
            overlay["combined_views"] = overlay["historical_views"] + overlay["live_events"]
            fig = px.bar(overlay.sort_values("combined_views"), x="combined_views", y="product_id", color="brand", orientation="h", text="live_events", title="Historical baseline with live event overlay", template="plotly_dark")
            fig.update_traces(texttemplate="+%{text} live", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            top_category = df_live["category_code"].mode().iloc[0] if not df_live["category_code"].empty else "unknown"
            with st.container(border=True):
                st.markdown(ICON_ACTIVITY, unsafe_allow_html=True)
                st.markdown("**Top live category**")
                st.markdown(f"### {top_category}")
                st.success("LIVE")
    else:
        st.info("No live events found yet. Click products on the GitHub Pages demo to generate S3 events.")

with tab_lakehouse:
    summary_df = pd.DataFrame([
        {"Component": "Bronze Layer", "Technology": "Parquet on S3", "Status": "Complete", "Result": "114,950,743 records"},
        {"Component": "Batch Layer", "Technology": "Apache Iceberg", "Status": "Complete", "Result": "109.95M REES46 rows queryable"},
        {"Component": "Speed Layer", "Technology": "Delta Lake", "Status": "Complete", "Result": "Trending products and hourly volume"},
        {"Component": "Serving Layer", "Technology": "Apache Hudi", "Status": "Complete", "Result": "15,095,144 user profiles"},
        {"Component": "ML Layer", "Technology": "MLflow + Spark ML", "Status": "Complete", "Result": "4 models trained"},
        {"Component": "API Layer", "Technology": "FastAPI", "Status": "Live", "Result": "6 endpoints tested"},
        {"Component": "Live Demo", "Technology": "GitHub Pages + S3", "Status": "Live", "Result": "Frontend clicks written to S3"}
    ])
    st.subheader("Project completion summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown(ICON_DATABASE, unsafe_allow_html=True)
            st.markdown("### Apache Iceberg")
            st.caption("Batch analytics, metadata snapshots, historical SQL")
            st.success("COMPLETE")
    with b:
        with st.container(border=True):
            st.markdown(ICON_ACTIVITY, unsafe_allow_html=True)
            st.markdown("### Delta Lake")
            st.caption("Speed layer, transaction log, trending products")
            st.success("COMPLETE")
    with c:
        with st.container(border=True):
            st.markdown(ICON_API, unsafe_allow_html=True)
            st.markdown("### Apache Hudi")
            st.caption("Mutable user profiles and upsert-ready serving data")
            st.success("COMPLETE")

with tab_ml:
    if ml_summary:
        metrics = ml_summary.get("metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card(ICON_MODEL, "CTR AUC-ROC", f"{metrics.get('ctr_auc_roc', 0):.4f}", "Criteo-style CTR")
        with m2: metric_card(ICON_MODEL, "ALS Samples", metrics.get('als_sample_recommendation_users', 0), "Recommendations")
        with m3: metric_card(ICON_MODEL, "Embeddings", f"{metrics.get('product_embedding_count', 0):,}", "Product2Vec")
        with m4: metric_card(ICON_MODEL, "Clusters", len(metrics.get("kmeans_segment_counts", {})), "K-Means")

        model_results_df = pd.DataFrame([
            {"Model": "CTR Prediction", "Dataset": "Criteo-schema click logs", "Input": "I1-I13 numerical features", "Output": "Click probability", "Metric": f"AUC-ROC: {metrics.get('ctr_auc_roc', 0):.4f}", "Serving Endpoint": "/predict/ctr"},
            {"Model": "ALS Recommender", "Dataset": "REES46 interactions", "Input": "user_id, product_id, event_type", "Output": "Top recommendations", "Metric": f"Sample users: {metrics.get('als_sample_recommendation_users', 0)}", "Serving Endpoint": "/recommend/{user_id}"},
            {"Model": "Product2Vec", "Dataset": "REES46 sequences", "Input": "Session sequences", "Output": "Similar products", "Metric": f"Embeddings: {metrics.get('product_embedding_count', 0):,}", "Serving Endpoint": "/similar/{product_id}"},
            {"Model": "K-Means Segmentation", "Dataset": "Hudi user profiles", "Input": "views, carts, purchases, spend", "Output": "Customer segment", "Metric": f"Clusters: {len(metrics.get('kmeans_segment_counts', {}))}", "Serving Endpoint": "/user/{user_id}/segment"}
        ])
        st.subheader("Trained model inventory")
        st.dataframe(model_results_df, use_container_width=True, hide_index=True)

with tab_benchmark:
    if benchmark_results:
        bench_df = pd.DataFrame(benchmark_results)
        bench_df["avg_query_sec"] = pd.to_numeric(bench_df["avg_query_sec"])
        fig_bench = px.bar(bench_df.sort_values("avg_query_sec"), x="format", y="avg_query_sec", color="format", text="avg_query_sec", title="Average query time by table format", template="plotly_dark")
        st.plotly_chart(fig_bench, use_container_width=True)

        st.subheader("Benchmark results")
        st.dataframe(bench_df[["format", "avg_query_sec", "runs", "feature"]], use_container_width=True, hide_index=True)
        fastest = bench_df.sort_values("avg_query_sec").iloc[0]
        st.info(f"For this single-node EC2/S3 workload, {fastest['format']} had the lowest average query time at {fastest['avg_query_sec']:.2f} seconds.")

with tab_events:
    if not df_live.empty:
        st.dataframe(df_live.sort_values("event_time", ascending=False), use_container_width=True)

with tab_api:
    api_df = pd.DataFrame([
        {"Endpoint": "/", "Purpose": "API health and project status", "Status": "Tested"},
        {"Endpoint": "/trending", "Purpose": "Returns Delta speed-layer trending products", "Status": "Tested"},
        {"Endpoint": "/recommend/{user_id}", "Purpose": "Returns ALS-style recommendations", "Status": "Tested"},
        {"Endpoint": "/predict/ctr", "Purpose": "Returns CTR click probability", "Status": "Tested"},
        {"Endpoint": "/user/{user_id}/segment", "Purpose": "Returns K-Means customer segment", "Status": "Tested"},
        {"Endpoint": "/similar/{product_id}", "Purpose": "Returns Product2Vec similar products", "Status": "Tested"},
        {"Endpoint": "/live-event", "Purpose": "Writes GitHub Pages click event to S3", "Status": "Tested"}
    ])
    st.subheader("Serving API endpoints")
    st.dataframe(api_df, use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()