# backend.py
from flask import Flask, request, jsonify, Response, send_file
import time
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import asyncio
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig
import os
from dotenv import load_dotenv
from flask_socketio import SocketIO


load_dotenv()


from transitions.transition import (
    read_wav,
    calculate_transition_timing,
    gradual_high_pass_blend_transition,
    calculate_8bar_starts,
)


# Breakdown of Hume AI emotions
POSTIVE_EMOTIONS = set(
    [
        "Admiration",
        "Adoration",
        "Aesthetic Appreciation",
        "Amusement",
        "Awe",
        "Contentment",
        "Ecstasy",
        "Entrancement",
        "Excitement",
        "Interest",
        "Joy",
        "Love",
        "Nostalgia",
        "Relief",
        "Romance",
        "Satisfaction",
        "Desire",
        "Surprise (positive)",
        "Sympathy",
        "Triumph",
    ]
)
NEGATIVE_EMOTIONS = set(
    [
        "Anger",
        "Anxiety",
        "Awkwardness",
        "Boredom",
        "Confusion",
        "Contempt",
        "Embarrassment",
        "Empathic Pain",
        "Fear",
        "Guilt",
        "Horror",
        "Pain",
        "Sadness",
        "Shame",
        "Surprise (negative)",
        "Tiredness",
    ]
)
NEUTRAL_EMOTIONS = set(
    [
        "Calmness",
        "Concentration",
        "Contemplation",
        "Craving",
        "Determination",
        "Disappointment",
        "Disgust",
        "Distress",
        "Doubt",
        "Envy",
        "Pride",
        "Realization",
    ]
)


app = Flask(__name__)

CORS(
    app,
    resources={r"/process_frame": {"origins": "http://localhost:3000"}},
)

client = HumeStreamClient(os.getenv("HUME_API_KEY"))
config = FaceConfig(identify_faces=True)

df = pd.read_csv("./spotify/good_matched_song_data.csv")
offset = 0

streaming = True  # Flag to control streaming
file_position = 0  # Variable to track the file position


# HELPER
def base64_to_image(base64_string):
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)

    # Write the decoded image data to a file
    with open("test.png", "wb") as file:
        file.write(image_data)


async def get_prediction():
    async with client.connect([config]) as socket:
        result = await socket.send_file("test.png")
        return result


def get_bboxes(prediction):
    bboxes = []
    if not "predictions" in prediction["face"]:
        return []
    for p in prediction["face"]["predictions"]:
        emotions = p["emotions"]
        top_emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)[:3]
        left, top, width, height = (
            int(p["bbox"]["x"]),
            int(p["bbox"]["y"]),
            int(p["bbox"]["w"]),
            int(p["bbox"]["h"]),
        )
        bboxes.append(
            {
                "top": top,
                "left": left,
                "width": width,
                "height": height,
                "top_emotions": top_emotions,
            }
        )

    return bboxes


def calculate_sentiment(pred):
    if "predictions" in pred["face"]:
        pos, neg = 0, 0

        for p in pred["face"]["predictions"]:
            emotions = p["emotions"]
            top_emotion = sorted(emotions, key=lambda x: x["score"], reverse=True)[0]
            if top_emotion["name"] in POSTIVE_EMOTIONS:
                pos += 1
            else:
                neg += 1

        # Determine which sentiment category has the highest count
        max_count = max(abs(pos), abs(neg))
        if max_count == pos:
            return 1
        else:
            return -1
    else:
        return None


# SONG HELPER
def select_song(current_song=None, switch_type=None, history=None):
    """Selects a song based on the current song and switch type, avoiding recent history and the current song."""
    if history is None:
        history = []

    df_filtered = df[
        ~df["track_name"].isin(history)
    ]  # Exclude all songs in history from the options

    if current_song is None or switch_type == "switch":
        # Randomly choose from songs not in history
        song = (
            df_filtered.sample().iloc[0]
            if not df_filtered.empty
            else df.sample().iloc[0]
        )
    elif switch_type == "same":
        # Filter songs to find the closest match not including the current song
        if current_song is not None:
            df_filtered = df_filtered[
                df_filtered["track_name"] != current_song["track_name"]
            ]  # Also exclude current song
            suitable_songs = df_filtered[
                (
                    df_filtered["tempo"].between(
                        current_song["tempo"] - 5, current_song["tempo"] + 5
                    )
                )
                & (df_filtered["danceability"] >= current_song["danceability"])
                & (df_filtered["energy"] >= current_song["energy"])
            ]
            song = (
                suitable_songs.nsmallest(1, "tempo").iloc[0]
                if not suitable_songs.empty
                else current_song
            )
        else:
            song = (
                df_filtered.sample().iloc[0]
                if not df_filtered.empty
                else df.sample().iloc[0]
            )
    else:
        song = (
            df_filtered.sample().iloc[0]
            if not df_filtered.empty
            else df.sample().iloc[0]
        )  # Fallback to random selection

    print(
        f"Selected Song: {song['track_name']} - Command: {switch_type}"
    )  # Debug statement
    return song


# def play_song(song):
#     """Play a song using Pygame."""
#     path = "../songs/" + song["local_path"]
#     if pygame.mixer.music.get_busy():
#         pygame.mixer.music.stop()
#     pygame.mixer.music.load(path)
#     pygame.mixer.music.play()
#     print(f"Playing: {path}")  # Debug statement

sentiment_history = []
song_history = []
current_song = select_song()
file_path = "./songs/" + current_song["local_path"]
can_transition = True


def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    global offset
    print(f"Offset: {offset / 1000.0} seconds")
    # current_pos = pygame.mixer.music.get_pos()  # Get current position in milliseconds
    current_pos = 0
    print(f"Current Position: {current_pos / 1000.0} seconds")
    print(f"Tempo: {current_song['tempo']} -> {next_song['tempo']}")
    bpm1 = current_song["tempo"]
    track_length_ms = len(read_wav("./songs/" + current_song["local_path"]))
    drop_ms = (
        current_song["beat_drop_s"] * 1000
    )  # Convert beat drop time to milliseconds
    start_times = calculate_8bar_starts(bpm1, track_length_ms, drop_ms)

    if "beat_drop_s" in next_song:
        beat_drop_ms_next_song = next_song["beat_drop_s"] * 1000
        print(f"Next Song Beat Drop: {beat_drop_ms_next_song / 1000.0} seconds")
        next_start = next((time for time in start_times if time > current_pos), None)
        transition_duration_ms = calculate_transition_timing(bpm1, 8, 4)
        # Assuming transition function is set to handle two audio segments
        transitioned_track, _ = gradual_high_pass_blend_transition(
            read_wav("./songs/" + current_song["local_path"]),
            read_wav("./songs/" + next_song["local_path"]),
            next_start + transition_duration_ms,
            beat_drop_ms_next_song,
            current_song["tempo"],
            next_song["tempo"],
        )

        # Export and play the transitioned track
        # current_pos = pygame.mixer.music.get_pos()
        current_pos = 0
        # pygame.mixer.music.load(
        #     transitioned_track[current_pos + offset :].export(format="wav")
        # )
        # pygame.mixer.music.play()
        offset = (
            beat_drop_ms_next_song - transition_duration_ms - (next_start - current_pos)
        )
        transition_track = transitioned_track[current_pos + offset :].export(
            format="wav"
        )

        response = change_audio_file(f'./songs/{next_song["local_path"]}')
        print(response)  # Print the response returned by change_audio_file
        global can_transition
        can_transition = False

    else:
        print("No beat drop info found, playing next song immediately.")
        # play_song(next_song)


def determine_transition(sentiment_history):
    state = ""
    global current_song
    # determine if we need to switch
    if not can_transition:
        return
    if sum(sentiment_history) < -10:  # means sentiment history is all negative
        print("emergency transition")
        transition_command = "switch"  # or whatever logic you have
        # Perform transition logic
        next_song = select_song(current_song, transition_command, song_history)
        handle_transition(current_song, next_song)
        current_song = next_song
        song_history.append(current_song["track_name"])
    else:
        print("continue")
        transition_command = "same"  # or whatever logic you have


# ROUTES
@app.route("/process_frame", methods=["POST"])
async def process_frame():
    data = request.get_json()
    image_src = data["imageSrc"]
    base64_to_image(image_src)
    prediction = await get_prediction()
    bboxes = get_bboxes(prediction)
    if "predictions" in prediction["face"]:
        sentiment_score = calculate_sentiment(prediction)
        sentiment_history.append(sentiment_score)
        if len(sentiment_history) > 20:
            sentiment_history.pop(0)

        determine_transition(sentiment_history)

    return jsonify({"boundingBoxes": bboxes, "sentimentHistory": sentiment_history})


@app.route("/stream_audio")
def stream_audio():
    def generate_audio():
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                yield chunk
                global file_position
                file_position = f.tell()

    # Create a response object with the audio data generator
    return Response(generate_audio(), mimetype="audio/mpeg")


@app.route("/change_audio_file/<new_file_path>")
def change_audio_file(new_file_path):
    print("IN CHANGE AUGIO FILE ", new_file_path)
    global file_path
    file_path = new_file_path
    return "Audio file changed to: " + new_file_path


if __name__ == "__main__":
    app.run(port=8000, debug=True)
