import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import io
import shap
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Gwamz Music Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and data
@st.cache_resource
def load_models():
    with open('gwamz_streams_model.pkl', 'rb') as f:
        streams_model = pickle.load(f)
    with open('gwamz_popularity_model.pkl', 'rb') as f:
        pop_model = pickle.load(f)
    return streams_model, pop_model

@st.cache_data
def load_data():
    df = pd.read_csv('gwamz_data.csv')
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
    return df

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# SHAP visualization function
def get_shap_plot(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    plt.figure()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Load data and models
try:
    streams_model, pop_model = load_models()
    df = load_data()
    
    # Calculate reference values
    first_release_date = df['release_date'].min()
    avg_streams = df['streams'].mean()
    avg_popularity = df['track_popularity'].mean()
    
    # Get feature names from model
    feature_names = streams_model.feature_names_in_
    
except Exception as e:
    st.error(f"Error loading data or models: {str(e)}")
    st.stop()

# App layout
st.title("🎵 Gwamz Music Performance Predictor")
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    .st-b7 {
        color: #1DB954;
    }
    .css-1aumxhk {
        background-color: #191414;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Predictor", "📈 Historical Analysis", "⚙️ Optimization"])

with tab1:
    st.header("Song Performance Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            release_date = st.date_input("Release Date", datetime.today())
            album_type = st.selectbox("Album Type", ["single", "album"])
            total_tracks = st.number_input("Total Tracks in Album", min_value=1, value=1)
            markets = st.number_input("Available Markets", min_value=1, max_value=200, value=185)
            
        with col2:
            track_number = st.number_input("Track Number", min_value=1, value=1)
            disc_number = st.number_input("Disc Number", min_value=1, value=1)
            explicit = st.checkbox("Explicit Content", value=True)
            version_type = st.selectbox("Version Type", 
                                       ["Original", "Sped Up", "Jersey", "Remix", "Instrumental"])
        
        submitted = st.form_submit_button("Predict Performance", type="primary")
        
    if submitted:
        with st.spinner('Making predictions...'):
            # Calculate days since first release
            days_since_first = (release_date - first_release_date.date()).days
            
            # Prepare input data
            input_data = {
                'artist_followers': 7937,
                'artist_popularity': 41,
                'album_type_single': 1 if album_type == 'single' else 0,
                'release_year': release_date.year,
                'total_tracks_in_album': total_tracks,
                'available_markets_count': markets,
                'track_number': track_number,
                'disc_number': disc_number,
                'explicit': int(explicit),
                'release_month': release_date.month,
                'release_dayofweek': release_date.weekday(),
                'is_sped_up': 1 if version_type == "Sped Up" else 0,
                'is_jersey': 1 if version_type == "Jersey" else 0,
                'is_remix': 1 if version_type == "Remix" else 0,
                'is_instrumental': 1 if version_type == "Instrumental" else 0,
                'album_version_count': total_tracks,
                'days_since_first_release': days_since_first,
                'track_sequence': track_number,
                'rolling_avg_streams': avg_streams,
                'rolling_avg_popularity': avg_popularity
            }
            
            # Create DataFrame and ensure all features are present
            input_df = pd.DataFrame([input_data])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]
            
            # Make predictions
            streams_pred = streams_model.predict(input_df)[0]
            pop_pred = pop_model.predict(input_df)[0]
            
            st.session_state.prediction_made = True
            st.session_state.streams_pred = streams_pred
            st.session_state.pop_pred = pop_pred
            st.session_state.input_df = input_df
    
    if st.session_state.prediction_made:
        st.success("Prediction Complete!")
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            delta_streams = (st.session_state.streams_pred - avg_streams)/avg_streams
            st.metric("Predicted Streams", 
                     f"{int(st.session_state.streams_pred):,}", 
                     delta=f"{delta_streams:.1%} vs average")
        with col2:
            delta_pop = st.session_state.pop_pred - avg_popularity
            st.metric("Predicted Popularity", 
                     f"{int(st.session_state.pop_pred)}/100", 
                     delta=f"{delta_pop:.1f} vs average")
        
        # Popularity gauge
        st.subheader("Popularity Score")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([''], [st.session_state.pop_pred], color='#1DB954', height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.text(st.session_state.pop_pred/2, 0, 
                f"{int(st.session_state.pop_pred)}", 
                ha='center', va='center', 
                color='white', fontsize=24)
        st.pyplot(fig)
        
        # Explanation
        with st.expander("📊 See prediction explanation"):
            st.markdown("""
                ### How this prediction was calculated
            
                The model analyzed Gwamz's historical performance data and identified patterns 
                between song characteristics and their streaming performance. Here are the key 
                factors influencing this prediction:
            """)
            
            # SHAP plot
            shap_html = f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{get_shap_plot(streams_model, st.session_state.input_df)}" 
                     style="max-width: 100%; height: auto;">
            </div>
            """
            st.components.v1.html(shap_html, height=400)
            
            st.markdown("""
                - **Positive values** indicate features increasing predicted streams
                - **Negative values** indicate features decreasing predicted streams
                - **Size** shows relative importance of each feature
            """)

with tab2:
    st.header("Historical Performance Analysis")
    
    # Time series plot
    st.subheader("Streams Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    df.set_index('release_date')['streams'].plot(
        ax=ax, marker='o', color='#1DB954', linewidth=2, markersize=8)
    ax.set_ylabel("Streams", fontsize=12)
    ax.set_title("Gwamz's Track Streaming Performance Over Time", fontsize=14)
    st.pyplot(fig)
    
    # Version comparison
    st.subheader("Version Performance Comparison")
    
    version_data = []
    for v in ["Sped Up", "Jersey", "Remix", "Instrumental"]:
        subset = df[df['track_name'].str.contains(v, case=False)]
        if not subset.empty:
            version_data.append({
                'Version': v,
                'Avg Streams': subset['streams'].mean(),
                'Avg Popularity': subset['track_popularity'].mean()
            })
    
    # Add original version
    original = df[~df['track_name'].str.contains('|'.join(["Sped Up", "Jersey", "Remix", "Instrumental"]))]
    version_data.append({
        'Version': 'Original',
        'Avg Streams': original['streams'].mean(),
        'Avg Popularity': original['track_popularity'].mean()
    })
    
    version_df = pd.DataFrame(version_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average Streams by Version**")
        st.bar_chart(version_df.set_index('Version')['Avg Streams'], color='#1DB954')
    with col2:
        st.markdown("**Average Popularity by Version**")
        st.bar_chart(version_df.set_index('Version')['Avg Popularity'], color='#1DB954')
    
    # Raw data explorer
    st.subheader("Track Data Explorer")
    st.dataframe(df.sort_values('release_date', ascending=False), height=400)

with tab3:
    st.header("Release Strategy Optimization")
    
    # Best release month analysis
    st.subheader("Optimal Release Timing")
    month_df = df.groupby('release_month')['streams'].mean().reset_index()
    best_month = month_df.loc[month_df['streams'].idxmax(), 'release_month']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Best Performing Month: {datetime(2023, best_month, 1).strftime('%B')}**")
        st.bar_chart(month_df.set_index('release_month'), color='#1DB954')
    
    with col2:
        st.markdown("**Month Performance Insights**")
        st.write(f"- Highest average streams in month {best_month}")
        st.write(f"- {month_df['streams'].max():,.0f} average streams in best month")
        st.write(f"- {month_df['streams'].min():,.0f} average streams in worst month")
    
    # Version recommendations
    st.subheader("Version Strategy")
    version_streams = df.groupby(['is_sped_up', 'is_jersey', 'is_remix'])['streams'].mean().reset_index()
    best_version = "Sped Up" if version_streams.loc[version_streams['is_sped_up'] == 1, 'streams'].max() > \
                   version_streams.loc[version_streams['is_jersey'] == 1, 'streams'].max() else "Jersey"
    
    st.markdown(f"**Recommended Version Type: {best_version}**")
    st.dataframe(version_streams.sort_values('streams', ascending=False))
    
    # Optimal parameters
    st.subheader("Optimal Release Parameters")
    optimal_params = {
        'Release Month': datetime(2023, best_month, 1).strftime('%B'),
        'Version Type': best_version,
        'Track Position': 1,
        'Market Coverage': 185,
        'Album Type': 'single',
        'Explicit Content': True
    }
    
    st.json(optimal_params)
    st.download_button(
        label="Download Optimization Report",
        data=pd.DataFrame(optimal_params.items()).to_csv(index=False),
        file_name="gwamz_optimal_release_strategy.csv",
        mime="text/csv"
    )

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Gwamz Music Analytics Dashboard • Data updated monthly</p>
    </div>
""", unsafe_allow_html=True)