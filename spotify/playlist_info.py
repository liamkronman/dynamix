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

def get_track_details(tracks):
    track_ids = []
    track_names = []
    track_artists = []
    for item in tracks:
        track = item['track']
        track_ids.append(track['id'])
        track_names.append(track['name'])
        track_artists.append(track['artists'][0]['name'])  # Just the first artist for simplicity
    return track_ids, track_names, track_artists

def get_audio_features(spotify, track_ids):
    features = spotify.audio_features(track_ids)
    return features



def get_audio_analysis(spotify, track_ids):
    analyses = {id: spotify.audio_analysis(id) for id in track_ids}
    return analyses


def analyze_playlist(spotify, playlist_id):
    tracks = get_playlist_tracks(spotify, playlist_id)
    track_ids, track_names, track_artists = get_track_details(tracks)
    features = get_audio_features(spotify, track_ids)
    # Combine features with track names and artists
    for feature, name, artist in zip(features, track_names, track_artists):
        feature['track_name'] = name
        feature['artist'] = artist
    return features


spotify_client = setup_spotify()
playlist_id = os.getenv("SPOTIFY_PLAYLIST_ID")
features = analyze_playlist(spotify_client, playlist_id)


# save features as CSV
features_df = pd.DataFrame(features)
features_df.to_csv("spotify_playlist_features.csv", index=False)
print("Saved features to spotify_playlist_features.csv")

# # save analyses as JSON
# analyses_json = json.dumps(analyses, indent=2)
# with open("spotify_analysis.json", "w") as json_file:
#     json_file.write(analyses_json)

# print("Saved features to spotify_playlist_analyses.json")
