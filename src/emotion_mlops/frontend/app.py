"""
Streamlit frontend for the Emotion Classification API.

Modern, dark-themed UI with Plotly visualizations and prediction history.

Run locally with:
    streamlit run src/emotion_mlops/frontend/app.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# --- Settings -------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
# Production API base, set via env var (e.g. the on-prem server URL).
SERVER_API_URL = os.getenv("SERVER_API_URL", "http://localhost:2001")


# --- Emotion metadata (emoji, color, description) ------------------------
EMOTION_META = {
    "anger": {
        "emoji": "",
        "color": "#FF4B4B",
        "desc": "Strong displeasure or hostility",
    },
    "disgust": {
        "emoji": "",
        "color": "#7BC74D",
        "desc": "Strong aversion or revulsion",
    },
    "fear": {
        "emoji": "",
        "color": "#9B59B6",
        "desc": "Anxiety, worry, or apprehension",
    },
    "happiness": {
        "emoji": "",
        "color": "#FFC107",
        "desc": "Joy, contentment, or pleasure",
    },
    "neutral": {
        "emoji": "",
        "color": "#9E9E9E",
        "desc": "No strong emotional signal",
    },
    "sadness": {
        "emoji": "",
        "color": "#3498DB",
        "desc": "Sorrow, grief, or unhappiness",
    },
    "surprise": {
        "emoji": "",
        "color": "#FF9800",
        "desc": "Astonishment or unexpected reaction",
    },
}


# --- Page configuration ---------------------------------------------------
st.set_page_config(
    page_title="EmotionScope",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session state initialization -----------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# --- Custom CSS -----------------------------------------------------------
st.markdown(
    """
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B6E, #FFC107);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, rgba(255, 75, 110, 0.1), rgba(255, 193, 7, 0.05));
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 75, 110, 0.2);
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-emoji {
        font-size: 5rem;
        margin: 0;
    }
    .emotion-name {
        font-size: 2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0.5rem 0;
    }
    .emotion-desc {
        color: #AAA;
        font-style: italic;
        margin: 0.5rem 0 1rem;
    }
    .confidence-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFC107;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    api_choice = st.radio(
        "API source",
        ["Local (development)", "Server (production)"],
        index=0,
    )
    API_URL = (
        DEFAULT_API_URL
        if api_choice == "Local (development)"
        else SERVER_API_URL
    )
    st.markdown(f"**Current API:** `{API_URL}`")

    st.divider()

    # History stats in sidebar
    st.markdown("### Session Stats")
    st.metric("Total predictions", len(st.session_state.history))

    if st.session_state.history:
        # Most common emotion
        emotions_list = [h["emotion"] for h in st.session_state.history]
        most_common = max(set(emotions_list), key=emotions_list.count)
        st.metric(
            "Most frequent", f"{EMOTION_META[most_common]['emoji']} {most_common}"
        )

    st.divider()

    st.markdown("### About")
    st.markdown(
        "**EmotionScope** is a fine-tuned DeBERTa-based emotion classifier "
        "trained on TV transcripts and GoEmotions data."
    )


# --- Header ---------------------------------------------------------------
st.markdown('<h1 class="main-title">EmotionScope</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Real-time emotion analysis powered by DeBERTa</p>',
    unsafe_allow_html=True,
)


# --- API health check -----------------------------------------------------
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success(f"Connected, model: `{health.get('model_version')}`")
    else:
        st.warning("API is reachable but model is not loaded.")
        st.stop()
except Exception:
    st.error(f"Cannot reach API at {API_URL}. Is the server running?")
    st.stop()


# --- Tabs for organization ------------------------------------------------
tab_predict, tab_history, tab_analytics = st.tabs(
    ["Predict", "History", "Analytics"]
)


# ==========================================================================
# TAB 1: PREDICTION
# ==========================================================================
with tab_predict:
    st.markdown("### Enter a sentence")

    samples = [
        "I cannot believe this is happening to me!",
        "I love spending time with my family.",
        "I feel so lonely and empty inside.",
        "That smells absolutely revolting.",
        "Listen to me right now!",
        "What a wonderful surprise!",
    ]

    col_input, col_samples = st.columns([3, 1])

    with col_input:
        text_input = st.text_area(
            "Type your sentence here:",
            placeholder="e.g. I'm having the best day ever!",
            height=120,
            label_visibility="collapsed",
            key="text_input_main",
        )

    with col_samples:
        st.markdown("**Quick samples:**")
        for i, sample in enumerate(samples[:3]):
            if st.button(
                f"Sample {i+1}", key=f"sample_{i}", use_container_width=True
            ):
                st.session_state["sample_text"] = sample
                st.rerun()

    if "sample_text" in st.session_state:
        text_input = st.session_state.pop("sample_text")

    predict_button = st.button(
        "Analyze Emotion",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
        if not text_input.strip():
            st.warning("Please enter a sentence first.")
        else:
            with st.spinner("Analyzing emotion..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/text",
                        json={"text": text_input},
                        timeout=10,
                    )
                    response.raise_for_status()
                    result = response.json()

                    emotion = result["emotion"]
                    confidence = result["confidence"]
                    meta = EMOTION_META.get(emotion, EMOTION_META["neutral"])

                    # Add to history
                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "text": text_input,
                            "emotion": emotion,
                            "confidence": confidence,
                            "model_version": result["model_version"],
                        }
                    )

                    # Result card
                    st.markdown(
                        f"""
                    <div class="emotion-card">
                        <p class="emotion-emoji">{meta['emoji']}</p>
                        <p class="emotion-name" style="color: {meta['color']};">{emotion}</p>
                        <p class="emotion-desc">{meta['desc']}</p>
                        <p class="confidence-value">{confidence * 100:.1f}%</p>
                        <p style="color: #888; margin: 0;">confidence</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Confidence bar chart
                    st.markdown("### Confidence Score")
                    fig = go.Figure(
                        go.Bar(
                            x=[confidence * 100],
                            y=[emotion.upper()],
                            orientation="h",
                            marker=dict(color=meta["color"]),
                            text=[f"{confidence * 100:.1f}%"],
                            textposition="inside",
                            textfont=dict(size=18, color="white"),
                        )
                    )
                    fig.update_layout(
                        xaxis=dict(range=[0, 100], title="Confidence (%)"),
                        yaxis=dict(title=""),
                        height=200,
                        margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Words", len(text_input.split()))
                    col_b.metric("Characters", len(text_input))
                    col_c.metric("Model", result["model_version"])

                    with st.expander("View raw API response"):
                        st.json(result)

                    st.success("Saved to history. Check the **History** tab.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")


# ==========================================================================
# TAB 2: HISTORY
# ==========================================================================
with tab_history:
    st.markdown("### Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet. Make a prediction in the **Predict** tab.")
    else:
        # Convert history to DataFrame for nice display
        df = pd.DataFrame(st.session_state.history)
        df["confidence_pct"] = (df["confidence"] * 100).round(1).astype(str) + "%"

        # Display options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input("Search text", "", key="search_history")
        with col2:
            emotion_filter = st.selectbox(
                "Filter by emotion",
                ["All"] + list(EMOTION_META.keys()),
                key="emotion_filter",
            )
        with col3:
            st.markdown(" ")  # spacing
            st.markdown(" ")
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        # Apply filters
        filtered_df = df.copy()
        if search:
            filtered_df = filtered_df[
                filtered_df["text"].str.contains(search, case=False)
            ]
        if emotion_filter != "All":
            filtered_df = filtered_df[filtered_df["emotion"] == emotion_filter]

        # Add emoji column for display
        display_df = filtered_df.copy()
        display_df["emotion"] = display_df["emotion"].apply(
            lambda e: f"{EMOTION_META[e]['emoji']} {e}"
        )
        display_df = display_df[["timestamp", "text", "emotion", "confidence_pct"]]
        display_df.columns = ["Time", "Text", "Emotion", "Confidence"]

        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} predictions**")

        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Download buttons
        st.markdown("### Export")
        col_csv, col_json = st.columns(2)

        with col_csv:
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "vDownload as CSV",
                data=csv,
                file_name=f"emotion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_json:
            json_data = filtered_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "vDownload as JSON",
                data=json_data,
                file_name=f"emotion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


# ==========================================================================
# TAB 3: ANALYTICS
# ==========================================================================
with tab_analytics:
    st.markdown("### Analytics Dashboard")

    if not st.session_state.history:
        st.info("Make some predictions first to see analytics here.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total predictions", len(df))
        col2.metric("Avg confidence", f"{df['confidence'].mean() * 100:.1f}%")
        col3.metric("Unique emotions", df["emotion"].nunique())
        col4.metric("Avg text length", f"{df['text'].str.len().mean():.0f} chars")

        st.divider()

        # Emotion distribution pie chart
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("#### Emotion Distribution")
            emotion_counts = df["emotion"].value_counts()
            colors = [EMOTION_META[e]["color"] for e in emotion_counts.index]

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            f"{EMOTION_META[e]['emoji']} {e}"
                            for e in emotion_counts.index
                        ],
                        values=emotion_counts.values,
                        hole=0.4,
                        marker=dict(colors=colors),
                    )
                ]
            )
            fig_pie.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.markdown("#### Average Confidence by Emotion")
            avg_conf = df.groupby("emotion")["confidence"].mean().sort_values()
            colors_bar = [EMOTION_META[e]["color"] for e in avg_conf.index]

            fig_bar = go.Figure(
                go.Bar(
                    x=avg_conf.values * 100,
                    y=[f"{EMOTION_META[e]['emoji']} {e}" for e in avg_conf.index],
                    orientation="h",
                    marker=dict(color=colors_bar),
                    text=[f"{v * 100:.1f}%" for v in avg_conf.values],
                    textposition="inside",
                    textfont=dict(color="white"),
                )
            )
            fig_bar.update_layout(
                xaxis=dict(range=[0, 100], title="Confidence (%)"),
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# --- Footer ---------------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Built with FastAPI + Streamlit · Block D Project"
    "</p>",
    unsafe_allow_html=True,
)
