import pandas as pd
import os
from fuzzywuzzy import process
from pathlib import Path

def normalize_name(name):
    """Normalize names for better matching."""
    return ''.join(e for e in name if e.isalnum()).lower()

# Load Spotify features
spotify_features_df = pd.read_csv("spotify_playlist_features.csv")
spotify_features_df['normalized_name'] = spotify_features_df['track_name'].apply(normalize_name)

# Get local song files
local_files = [Path(f) for f in os.listdir('../songs') if f.endswith('.wav')]
print(local_files)
local_song_names = [f.stem for f in local_files]
normalized_local_names = [normalize_name(name) for name in local_song_names]

# Match local songs to Spotify tracks
matches = {local_name: process.extractOne(normalized_local_names[i], spotify_features_df['normalized_name'])
           for i, local_name in enumerate(local_song_names)}

# Generate a DataFrame from matches
matched_data = []
for local_name, (matched_name, score, index) in matches.items():
    if score > 85:  # threshold for match quality
        track_data = spotify_features_df.iloc[index].to_dict()
        track_data['local_path'] = local_files[local_song_names.index(local_name)]
        matched_data.append(track_data)

matched_df = pd.DataFrame(matched_data)

# Save the matched data to a CSV for later use
matched_df.to_csv("matched_song_data.csv", index=False)
print("Matched song data saved to 'matched_song_data.csv'")
