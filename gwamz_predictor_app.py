import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Gwamz Music Analytics",
    page_icon="🎵",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    try:
        with open('gwamz_streams_model.pkl', 'rb') as f:
            streams_model = pickle.load(f)
        with open('gwamz_popularity_model.pkl', 'rb') as f:
            pop_model = pickle.load(f)
        return streams_model, pop_model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('gwamz_data.csv')
        df['release_date'] = pd.to_datetime(df['release_date'])
        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

# Initialize app
streams_model, pop_model = load_models()
df = load_data()

if streams_model is None or df.empty:
    st.stop()

# Calculate reference values
first_release_date = df['release_date'].min()
avg_streams = df['streams'].mean()
avg_popularity = df['track_popularity'].mean()

# Main app
st.title("🎵 Gwamz Music Performance Predictor")

tab1, tab2 = st.tabs(["📊 Predictor", "📈 Historical Analysis"])

with tab1:
    st.header("Song Performance Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            release_date = st.date_input("Release Date", datetime.today())
            album_type = st.selectbox("Album Type", ["single", "album"])
            total_tracks = st.number_input("Total Tracks", min_value=1, value=1)
            
        with col2:
            track_number = st.number_input("Track Number", min_value=1, value=1)
            explicit = st.checkbox("Explicit Content", value=True)
            version_type = st.selectbox("Version", ["Original", "Sped Up", "Jersey"])
        
        submitted = st.form_submit_button("Predict", type="primary")
    
    if submitted:
        # Prepare input data
        input_data = {
            'artist_followers': 7937,
            'artist_popularity': 41,
            'album_type_single': 1 if album_type == 'single' else 0,
            'release_year': release_date.year,
            'total_tracks_in_album': total_tracks,
            'available_markets_count': 185,
            'track_number': track_number,
            'disc_number': 1,
            'explicit': int(explicit),
            'release_month': release_date.month,
            'release_dayofweek': release_date.weekday(),
            'is_sped_up': 1 if version_type == "Sped Up" else 0,
            'is_jersey': 1 if version_type == "Jersey" else 0,
            'is_remix': 0,
            'is_instrumental': 0,
            'album_version_count': total_tracks,
            'days_since_first_release': (release_date - first_release_date.date()).days,
            'track_sequence': track_number,
            'rolling_avg_streams': avg_streams,
            'rolling_avg_popularity': avg_popularity
        }
        
        # Make predictions
        input_df = pd.DataFrame([input_data])
        streams_pred = streams_model.predict(input_df)[0]
        pop_pred = pop_model.predict(input_df)[0]
        
        # Show results
        st.success("Prediction Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Streams", f"{int(streams_pred):,}")
        with col2:
            st.metric("Predicted Popularity", f"{int(pop_pred)}/100")
        
        # Popularity gauge
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([''], [pop_pred], color='#1DB954')
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.text(pop_pred/2, 0, f"{int(pop_pred)}", 
                ha='center', va='center', color='white', fontsize=24)
        st.pyplot(fig)

with tab2:
    st.header("Historical Performance")
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(12, 4))
    df.set_index('release_date')['streams'].plot(ax=ax, color='#1DB954')
    st.pyplot(fig)
    
    # Version comparison
    version_comparison = df.groupby(['is_sped_up', 'is_jersey'])['streams'].mean().reset_index()
    st.dataframe(version_comparison)
