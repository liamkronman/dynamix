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

import asyncio
import cv2
import base64
import requests
import websockets
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import pygame
from pydub import AudioSegment
from pygame import mixer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pygame.locals import *


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
    resources={
        r"/process_frame": {"origins": "http://localhost:3000"},
        r"/start_mixer": {"origins": "http://localhost:3000"},
        r"/quit_mixer": {"origins": "http://localhost:3000"},
        r"/pause_mixer": {"origins": "http://localhost:3000"},
        r"/unpause_mixer": {"origins": "http://localhost:3000"},
    },
)

client = HumeStreamClient(os.getenv("HUME_API_KEY"))
config = FaceConfig(identify_faces=True)


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
    global POSTIVE_EMOTIONS
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


df = pd.read_csv("./spotify/good_matched_song_data.csv")

# initialize the positive feedback
if "sentiment_score" not in df.columns:
    df["sentiment_score"] = 0

features = ["danceability", "energy", "tempo", "loudness", "valence"]
df_features_scaled = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Function to calculate similarity matrix
def calculate_similarity(df):
    return cosine_similarity(df[features])


similarity_matrix = calculate_similarity(df_features_scaled)
# print(f"Song similarity matrix: {similarity_matrix}")

emotional_scores = []

transitioning = False
transition_start = -1  # tracks the start of the transition
transition_end = -1  # tracks the end of the transition


def calculate_sentiment(pred):
    global emotional_scores
    pos, neg = 0, 0
    for p in pred["face"]["predictions"]:
        emotions = p["emotions"]
        for e in emotions:
            if e["name"] in POSTIVE_EMOTIONS:
                pos += e["score"]
            elif e["name"] in NEGATIVE_EMOTIONS:
                neg += e["score"]
    emotional_scores.append(pos - neg)


def reset_emotional_scores():
    global emotional_scores
    emotional_scores = []


def select_song(current_song=None, switch_type=None, history=None):
    """Selects a song based on the current song and switch type, avoiding recent history and the current song."""
    global similarity_matrix, df

    if history is None:
        history = []

    mask = ~df["track_name"].isin(history)
    # print(df["sentiment_score"].values)

    aggregate_scores = np.dot(similarity_matrix, df["sentiment_score"].values)

    filtered_scores = aggregate_scores[mask]

    # add the minimum score to all scores to make them positive
    filtered_scores += abs(filtered_scores.min())

    if filtered_scores.sum() == 0:
        probabilities = np.ones(len(filtered_scores)) / len(filtered_scores)
    else:
        probabilities = filtered_scores / filtered_scores.sum()
    # print(probabilities)

    selected_index = np.random.choice(df[mask].index, p=probabilities)

    song = df.loc[selected_index]

    print(
        f"Selected Song: {song['track_name']} - Command: {switch_type}"
    )  # Debug statement
    return song


def play_song(song):
    """Play a song using Pygame."""
    path = "./songs/" + song["local_path"]
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"Playing: {path}")  # Debug statement


offset = 0
current_pos = 0  # Get current position in milliseconds


def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    global offset
    global transition_start
    global transition_end
    global current_pos
    print(f"Offset: {offset / 1000.0} seconds")
    current_pos = pygame.mixer.music.get_pos()  # Get current position in milliseconds
    true_pos = current_pos + offset
    print(f"Current Position: {current_pos / 1000.0} seconds")
    print(f"Tempo: {current_song['tempo']} -> {next_song['tempo']}")
    bpm1 = current_song["tempo"]
    track_length_ms = len(read_wav("./songs/" + current_song["local_path"]))
    drop_ms = (
        current_song["beat_drop_s"] * 1000
    )  # Convert beat drop time to milliseconds
    start_times = calculate_8bar_starts(bpm1, track_length_ms, drop_ms)

    if "beat_drop_s" in next_song:
        if true_pos >= current_song["beat_drop_s"] * 1000:
            beat_drop_ms_next_song = next_song["beat_drop_s"] * 1000
            print(f"Next Song Beat Drop: {beat_drop_ms_next_song / 1000.0} seconds")
            next_start = next((time for time in start_times if time > true_pos), None)
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
            current_pos = pygame.mixer.music.get_pos()
            true_pos = current_pos + offset
            pygame.mixer.music.load(transitioned_track[true_pos:].export(format="wav"))
            pygame.mixer.music.play()
            transition_start = next_start - true_pos
            transition_end = transition_duration_ms + transition_start
            offset = (
                beat_drop_ms_next_song
                - transition_duration_ms
                - (next_start - true_pos)
            )
    else:
        print("No beat drop info found, playing next song immediately.")
        play_song(next_song)


def calculate_headcount(pred):
    if "predictions" in pred["face"]:
        return len(pred["face"]["predictions"])
    else:
        return 0


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
        return 0


on = False
sentiment_history = []
num_frames = 20
sentiment_history = []
headcount_history = []
history = []
current_mood = "Neutral"  # Default mood
decided_transition = False


# ROUTES
@app.route("/start_mixer", methods=["POST"])
def start_mixer():
    global current_song, history, on
    pygame.init()
    pygame.mixer.init()
    on = True
    current_song = select_song()
    play_song(current_song)
    history.append(current_song["track_name"])
    return jsonify({"message": "started mixer!"})


@app.route("/quit_mixer", methods=["POST"])
def end_mixer():
    global on
    on = False
    pygame.quit()
    return jsonify({"message": "quit mixer!"})


@app.route("/pause_mixer", methods=["POST"])
def pause_mixer():
    global on
    on = False
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
    return jsonify({"message": "paused mixer!"})


@app.route("/unpause_mixer", methods=["POST"])
def unpause_mixer():
    global on, sentiment_history
    on = True
    print("UNPAUSED MIXER")
    sentiment_history = num_frames * [0]
    pygame.mixer.music.unpause()
    return jsonify({"message": "unpaused mixer!"})


@app.route("/process_frame", methods=["POST"])
async def process_frame():
    global sentiment_history, current_song, num_frames, current_pos, offset, transitioning, transition_start, transition_end, decided_transition

    data = request.get_json()
    image_src = data["imageSrc"]
    base64_to_image(image_src)
    prediction = await get_prediction()
    bboxes = get_bboxes(prediction)

    if not on:
        print("not processing frames")
        return jsonify({"boundingBoxes": bboxes})
    else:
        print("processing frames")

    current_pos = pygame.mixer.music.get_pos()
    if transition_start <= current_pos <= transition_end:
        transitioning = True
        sentiment_history = [1] * num_frames
        decided_transition = False
    else:
        transitioning = False

    if "predictions" in prediction["face"]:
        sentiment_score = calculate_sentiment(prediction)
        sentiment_history.append(sentiment_score)
        if len(sentiment_history) > 20:
            sentiment_history.pop(0)

    # Update sentiment and determine song transitions
    sentiment_score = calculate_sentiment(prediction)
    # add sentiment score to "sentiment_score" column in df for current song
    df.loc[
        df["track_name"] == current_song["track_name"], "sentiment_score"
    ] += sentiment_score
    sentiment_history.append(sentiment_score)
    current_mood = "Positive" if sentiment_score == 1 else "Negative"

    if len(sentiment_history) > num_frames:
        sentiment_history.pop(0)
        if (
            not decided_transition
            and sum(sentiment_history) < -10
            # or current_headcount <= headcount_history[0] // 2
        ):  # Negative mood detected
            # only initiate a transition if the beat drop has passed
            # print("HERE ", current_pos, offset, transition_start, transition_end)
            print("SENTIMENT HISTORY THAT TRIGGERS A TRANSITION ", sentiment_history)
            if current_pos + offset >= current_song["beat_drop_s"] * 1000:
                decided_transition = True
                transition_command = "switch"
                next_song = select_song(current_song, transition_command, history)
                handle_transition(current_song, next_song)
                current_song = next_song
                history.append(current_song["track_name"])
                sentiment_history = [1] * num_frames

        else:
            # print("Continue playing")
            # print(sentiment_history)
            pass

    return jsonify({"boundingBoxes": bboxes, "sentimentHistory": sentiment_history})


if __name__ == "__main__":
    app.run(port=8000)
