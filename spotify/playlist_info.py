import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import pandas as pd
import json


load_dotenv()


def setup_spotify():
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri="https://hci.csail.mit.edu/",
            scope="playlist-read-private",
        )
    )


def get_playlist_tracks(spotify, playlist_id):
    results = spotify.playlist_tracks(playlist_id)
    tracks = results["items"]
    while results["next"]:
        results = spotify.next(results)
        tracks.extend(results["items"])
    return tracks


def get_audio_features(spotify, track_ids):
    features = spotify.audio_features(track_ids)
    return features


def get_audio_analysis(spotify, track_ids):
    analyses = {id: spotify.audio_analysis(id) for id in track_ids}
    return analyses


def analyze_playlist(spotify, playlist_id):
    track_details = get_playlist_tracks(spotify, playlist_id)
    track_ids = [track["track"]["id"] for track in track_details if track["track"]]
    features = get_audio_features(spotify, track_ids)
    analyses = get_audio_analysis(spotify, track_ids)
    return features, analyses


spotify_client = setup_spotify()
playlist_id = os.getenv("SPOTIFY_PLAYLIST_ID")
features, analyses = analyze_playlist(spotify_client, playlist_id)


# save features as CSV
features_df = pd.DataFrame(features)
features_df.to_csv("spotify_playlist_features.csv", index=False)
print("Saved features to spotify_playlist_features.csv")

# save analyses as JSON
analyses_json = json.dumps(analyses, indent=2)
with open("spotify_analysis.json", "w") as json_file:
    json_file.write(analyses_json)

print("Saved features to spotify_playlist_analyses.json")
