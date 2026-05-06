import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import boto3
import os

# Page config
st.set_page_config(
    page_title="🖱️ ClickStream Analytics - LIVE S3",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🖱️ ClickStream Analytics — Real-Time Dashboard")
st.markdown("### 👀 Watching `s3://clickstream-analytics-akash/live-events/` in real-time")

# --- S3 Configuration ---
S3_BUCKET = "clickstream-analytics-akash"
S3_LIVE_PREFIX = "live-events/"

@st.cache_resource
def get_s3_client():
    # Uses local credentials or env vars
    return boto3.client('s3')

def get_live_clicks():
    s3 = get_s3_client()
    try:
        # List files in the live-events prefix
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_LIVE_PREFIX)
        if 'Contents' not in response:
            return pd.DataFrame()
            
        # Sort by last modified to get newest first, limit to last 100 events
        files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[:100]
        
        data = []
        for f in files:
            if not f['Key'].endswith('.json'):
                continue
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f['Key'])
            content = json.loads(obj['Body'].read().decode('utf-8'))
            data.append(content)
            
        df = pd.DataFrame(data)
        if not df.empty:
             df['event_time'] = pd.to_datetime(df['event_time'])
        return df
    except Exception as e:
        st.warning(f"Error fetching S3 data: {e}")
        return pd.DataFrame()

# Auto-refresh loop
placeholder = st.empty()

while True:
    df = get_live_clicks()
    
    with placeholder.container():
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if not df.empty:
            active_users = df['user_id'].nunique()
            total_clicks = len(df)
            top_category = df['category_code'].mode()[0] if 'category_code' in df and not df['category_code'].isnull().all() else "N/A"
            total_value = df['price'].sum() if 'price' in df else 0
            
            col1.metric("👥 Live Users", active_users)
            col2.metric("🖱️ Live Clicks", total_clicks)
            col3.metric("📂 Top Category", top_category)
            col4.metric("💰 Live GMV ($)", f"{total_value:,.2f}")
        else:
            col1.metric("👥 Live Users", 0)
            col2.metric("🖱️ Live Clicks", 0)
            col3.metric("📂 Top Category", "N/A")
            col4.metric("💰 Live GMV ($)", "0.00")
        
        st.markdown("---")
        
        # Live Click Feed
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader("🔴 Live Event Distribution")
            if not df.empty:
                # Sunburst diagram of category -> brand -> event_type
                try:
                    # Clean up data for sunburst
                    plot_df = df.copy()
                    plot_df['category_code'] = plot_df['category_code'].fillna('unknown')
                    plot_df['brand'] = plot_df['brand'].fillna('generic')
                    
                    fig = px.sunburst(
                        plot_df,
                        path=['category_code', 'brand', 'event_type'],
                        color='event_type',
                        template="plotly_dark",
                        color_discrete_map={'view':'#636EFA', 'cart':'#EF553B', 'purchase':'#00CC96'}
                    )
                    fig.update_layout(height=450, margin=dict(t=0, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Generating live visualization...")
            else:
                st.info("Waiting for live events from GitHub Pages...")
        
        with right_col:
            st.subheader("📋 Recent Events")
            if not df.empty:
                # Display clean dataframe
                cols_to_show = ['user_id', 'event_type', 'brand', 'price', 'event_time']
                display_df = df[[c for c in cols_to_show if c in df]].copy()
                if not display_df.empty:
                    display_df['event_time'] = display_df['event_time'].dt.strftime('%H:%M:%S')
                    st.dataframe(
                        display_df.head(15),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.write("Listening for events at `s3://.../live-events/`...")
        
        # Historical Comparison Mock (as suggested in your prompt)
        st.subheader("📊 Historical Context (REES46 Baseline)")
        h_col1, h_col2 = st.columns(2)
        
        with h_col1:
            # Mock historical data vs live
            historical_total = 942167 # example
            live_total = len(df)
            st.write(f"**Total Views (Samsung S20):**")
            st.write(f"- Historical: {historical_total:,}")
            st.write(f"- Live Demo: {live_total}")
            st.progress(min(live_total / 100, 1.0)) # Progress bar for demo effect
            
        with h_col2:
            if not df.empty and 'brand' in df:
                brand_counts = df['brand'].value_counts().reset_index()
                brand_counts.columns = ['brand', 'count']
                fig3 = px.bar(
                    brand_counts, x='brand', y='count', 
                    title="Live Brand Interest",
                    template="plotly_dark",
                    color='brand'
                )
                fig3.update_layout(height=250)
                st.plotly_chart(fig3, use_container_width=True)
    
    time.sleep(2)  # Refresh every 2 seconds
