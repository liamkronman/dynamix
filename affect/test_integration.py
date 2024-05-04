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

df = pd.read_csv("../spotify/good_matched_song_data.csv")
pygame.init()
pygame.mixer.init()

import random

import sys

sys.path.append("..")
from transitions.transition import (
    read_wav,
    calculate_transition_timing,
    gradual_high_pass_blend_transition,
    calculate_8bar_starts,
)


load_dotenv()

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

# initialize the positive feedback
if 'positive_feedback_count' not in df.columns:
    df['positive_feedback_count'] = 0

features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']
df[features] = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Function to calculate similarity matrix
def calculate_similarity(df):
    return cosine_similarity(df[features])

def update_positive_feedback(track_id, feedback_events_count):
    decay_factor = 0.9  # Adjust this based on desired rate of decay
    df.loc[df['id'] != track_id, 'positive_feedback_count'] *= decay_factor
    df.loc[df['id'] == track_id, 'positive_feedback_count'] += 1


similarity_matrix = calculate_similarity(df)

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


def play_song(song):
    """Play a song using Pygame."""
    path = "../songs/" + song["local_path"]
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"Playing: {path}")  # Debug statement


offset = 0


def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    global offset
    print(f"Offset: {offset / 1000.0} seconds")
    current_pos = pygame.mixer.music.get_pos()  # Get current position in milliseconds
    print(f"Current Position: {current_pos / 1000.0} seconds")
    print(f"Tempo: {current_song['tempo']} -> {next_song['tempo']}")
    bpm1 = current_song["tempo"]
    track_length_ms = len(read_wav("../songs/" + current_song["local_path"]))
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
            read_wav("../songs/" + current_song["local_path"]),
            read_wav("../songs/" + next_song["local_path"]),
            next_start + transition_duration_ms,
            beat_drop_ms_next_song,
            current_song["tempo"],
            next_song["tempo"],
        )

        # Export and play the transitioned track
        current_pos = pygame.mixer.music.get_pos()
        pygame.mixer.music.load(
            transitioned_track[current_pos + offset :].export(format="wav")
        )
        pygame.mixer.music.play()
        offset = (
            beat_drop_ms_next_song - transition_duration_ms - (next_start - current_pos)
        )
    else:
        print("No beat drop info found, playing next song immediately.")
        play_song(next_song)


async def connect_to_websocket():
    uri = "wss://api.hume.ai/v0/stream/models"
    extra_headers = {"X-Hume-Api-Key": os.getenv("HUME_API_KEY")}

    async with websockets.connect(uri, extra_headers=extra_headers) as websocket:
        await read_frames_and_call_api(websocket, "/")


def encode_data(filepath: Path) -> str:
    with Path(filepath).open("rb") as fp:
        bytes_data = base64.b64encode(fp.read())
        encoded_data = bytes_data.decode("utf-8")
        return encoded_data


def encode_frame(frame) -> str:
    # Convert the frame to bytes
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()

    # Encode the bytes in base64 format
    encoded_data = base64.b64encode(frame_bytes).decode("utf-8")

    return encoded_data


def draw_on_frame(frame, pred):
    # Convert frame to numpy array

    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add bounding boxes
    for p in pred:
        x, y, w, h = (
            int(p["bbox"]["x"]),
            int(p["bbox"]["y"]),
            int(p["bbox"]["w"]),
            int(p["bbox"]["h"]),
        )

        # Draw bounding box
        cv2.rectangle(frame_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add top three emotions
        emotions = p["emotions"]
        top_three = sorted(emotions, key=lambda x: x["score"], reverse=True)[:3]
        labels = [f"{e['name']}: {round(e['score'], 2)}" for e in top_three]
        i = 0
        for l in labels:
            cv2.putText(
                frame_np,
                l,
                (x, y - 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            i += 1

    # Convert the modified numpy array back to a frame
    modified_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    return modified_frame


def annotate_sentiment_score(frame, sentiment_score, sentiment_history):
    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sentiment_score_map = {-1: "negative", 0: "neutral", 1: "positive"}
    print(sentiment_score_map[sentiment_score])
    cv2.putText(
        frame_np,
        sentiment_score_map[sentiment_score],
        (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame_np,
        ", ".join([sentiment_score_map[score] for score in sentiment_history]),
        (0, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    modified_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    return modified_frame


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


async def read_frames_and_call_api(websocket, path):
    cap = cv2.VideoCapture(0)

    # running window of sentiment from last 20 frames
    num_frames = 20
    sentiment_history = []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    print(fps)

    history = []
    current_song = select_song()
    play_song(current_song)
    history.append(current_song["track_name"])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame_count % fps * 5 > 0:
        #     frame_count += 1
        #     # Display the frame
        #     cv2.imshow("frame", frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        #     continue

        payload = {
            "models": {"face": {"identify_faces": True}},
            "data": encode_frame(frame),
        }

        await websocket.send(json.dumps(payload))

        message = await websocket.recv()
        pred = json.loads(message)

        # update sentiment tracker
        sentiment_score = calculate_sentiment(pred)  # 1 for positive, -1 for negative
        if sentiment_score:
            sentiment_history.append(sentiment_score)

        # Keep sentiment history within the specified number of frames
        if len(sentiment_history) > num_frames:
            sentiment_history.pop(0)

        # determine if we need to switch
        if sum(sentiment_history) < -10:  # means sentiment history is all negative
            print("emergency transition")
            transition_command = "switch"  # or whatever logic you have
            # Perform transition logic
            next_song = select_song(current_song, transition_command, history)
            handle_transition(current_song, next_song)
            current_song = next_song
            history.append(current_song["track_name"])
            for _ in range(20):
                sentiment_history.append(1)
            if len(history) > 10:
                history.pop(0)
        else:
            print("continue")
            transition_command = "same"  # or whatever logic you have

        if "predictions" in pred["face"]:
            annotated_frame = draw_on_frame(frame, pred["face"]["predictions"])
            annotated_frame = annotate_sentiment_score(
                annotated_frame, sentiment_score, sentiment_history
            )
        else:
            annotated_frame = frame

        frame_count += 1

        # Display the frame
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


asyncio.get_event_loop().run_until_complete(connect_to_websocket())
