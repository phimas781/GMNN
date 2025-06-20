from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from datetime import datetime

app = FastAPI()

# Load models
with open('gwamz_streams_model.pkl', 'rb') as f:
    streams_model = pickle.load(f)
with open('gwamz_popularity_model.pkl', 'rb') as f:
    pop_model = pickle.load(f)

class SongInput(BaseModel):
    release_date: str
    album_type: str
    total_tracks_in_album: int
    available_markets_count: int
    track_number: int
    disc_number: int
    explicit: bool
    version_type: str

@app.post("/predict")
async def predict(song: SongInput):
    # Reference values (would normally come from database)
    ref_values = {
        'artist_followers': 7937,
        'artist_popularity': 41,
        'avg_streams': 1000000,
        'avg_popularity': 40,
        'first_release': '2021-04-29'
    }
    
    # Prepare input
    release_date = datetime.strptime(song.release_date, '%Y-%m-%d')
    days_since_first = (release_date - datetime.strptime(ref_values['first_release'], '%Y-%m-%d')).days
    
    input_data = {
        'artist_followers': ref_values['artist_followers'],
        'artist_popularity': ref_values['artist_popularity'],
        'album_type_single': 1 if song.album_type == 'single' else 0,
        'release_year': release_date.year,
        'total_tracks_in_album': song.total_tracks_in_album,
        'available_markets_count': song.available_markets_count,
        'track_number': song.track_number,
        'disc_number': song.disc_number,
        'explicit': int(song.explicit),
        'release_month': release_date.month,
        'release_dayofweek': release_date.weekday(),
        'is_sped_up': 1 if song.version_type == "Sped Up" else 0,
        'is_jersey': 1 if song.version_type == "Jersey" else 0,
        'is_remix': 1 if song.version_type == "Remix" else 0,
        'is_instrumental': 1 if song.version_type == "Instrumental" else 0,
        'album_version_count': song.total_tracks_in_album,
        'days_since_first_release': days_since_first,
        'track_sequence': song.track_number,
        'rolling_avg_streams': ref_values['avg_streams'],
        'rolling_avg_popularity': ref_values['avg_popularity']
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns are present
    expected_columns = streams_model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    
    # Predict
    streams_pred = streams_model.predict(input_df)[0]
    pop_pred = pop_model.predict(input_df)[0]
    
    return {
        "predicted_streams": int(streams_pred),
        "predicted_popularity": int(pop_pred),
        "days_since_first_release": days_since_first
    }

# Run with: uvicorn api:app --reload