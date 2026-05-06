import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from collections import defaultdict
import streamlit.components.v1 as components
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Page config
st.set_page_config(
    page_title="🖱️ ClickStream Analytics - LIVE",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🖱️ ClickStream Analytics — Real-Time Dashboard")
st.markdown("### 👀 Watch audience clicks appear in real-time!")

# --- Firebase/Firestore or JSON-based real-time data store ---
# For simplicity, we'll use Streamlit's session state + a shared JSON file
# In production, use Firebase Realtime DB or Supabase for true real-time sync

@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        # Check for environment variable
        firebase_key = os.environ.get("FIREBASE_KEY")
        if not firebase_key:
            st.error("Missing FIREBASE_KEY environment variable. Please add it to your Streamlit secrets.")
            st.stop()
        
        try:
            cred_dict = json.loads(firebase_key)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Error initializing Firebase: {e}")
            st.stop()
            
    return firestore.client()

try:
    db = init_firebase()
except Exception as e:
    st.error(f"Could not connect to Firebase: {e}")
    st.stop()

# Real-time listener
def get_live_clicks():
    try:
        clicks_ref = db.collection("clickstream")
        # Ordering by timestamp and limiting to 100
        docs = clicks_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            # Convert firestore timestamp to python datetime
            if 'timestamp' in d and d['timestamp']:
                try:
                    # Some versions return datetime directly, others need conversion
                    if hasattr(d['timestamp'], 'to_datetime'):
                        d['timestamp'] = d['timestamp'].to_datetime()
                except:
                    pass
            data.append(d)
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
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
            pages_visited = df['page'].nunique()
            
            # Calculate avg session time if possible
            try:
                # Group by user and calculate (max - min) time
                session_times = df.groupby('user_id')['timestamp'].agg(['max', 'min'])
                avg_session = (session_times['max'] - session_times['min']).dt.total_seconds().mean()
            except:
                avg_session = 0
                
            col1.metric("👥 Active Users", active_users)
            col2.metric("🖱️ Total Clicks", total_clicks)
            col3.metric("📄 Pages Visited", pages_visited)
            col4.metric("⏱️ Avg Session (s)", round(avg_session, 1))
        else:
            col1.metric("👥 Active Users", 0)
            col2.metric("🖱️ Total Clicks", 0)
            col3.metric("📄 Pages Visited", 0)
            col4.metric("⏱️ Avg Session (s)", 0)
        
        st.markdown("---")
        
        # Live Click Feed
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader("🔴 Live Click Stream")
            if not df.empty:
                # Sunburst diagram of user flows
                try:
                    flow_df = df.groupby(['page', 'action']).size().reset_index(name='count')
                    fig = px.sunburst(
                        flow_df,
                        path=['page', 'action'],
                        values='count',
                        color='count',
                        color_continuous_scale='RdYlGn',
                        template="plotly_dark"
                    )
                    fig.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Waiting for more data to generate flow chart...")
            else:
                st.info("No data yet. Clicks will appear here as they happen!")
        
        with right_col:
            st.subheader("📋 Recent Clicks")
            if not df.empty:
                # Display clean dataframe
                display_df = df[['user_id', 'page', 'action', 'timestamp']].copy()
                if not display_df.empty:
                    # Format timestamp for display
                    try:
                        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                    except:
                        pass
                    st.dataframe(
                        display_df.head(15),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.write("Listening for incoming events...")
        
        # Heatmap of clicks over time
        st.subheader("🗺️ Click Heatmap by Page")
        if not df.empty:
            try:
                # Simple heatmap of page vs time (minute)
                df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
                heatmap_data = df.groupby(['page', 'minute']).size().unstack(fill_value=0)
                if not heatmap_data.empty:
                    fig2 = px.imshow(
                        heatmap_data, 
                        color_continuous_scale='Hot', 
                        aspect='auto',
                        labels=dict(x="Minute", y="Page", color="Clicks"),
                        template="plotly_dark"
                    )
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                pass
    
    time.sleep(2)  # Refresh every 2 seconds
