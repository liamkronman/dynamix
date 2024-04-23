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

df = pd.read_csv("../spotify/matched_song_data.csv")
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


def select_song(current_song=None, switch_type=None, history=None):
    """Selects a song based on the current song and switch type, avoiding recent history."""
    if history is None:
        history = []

    if switch_type == "switch" or current_song is None:
        # Filter out recently played songs if possible
        available_songs = df[~df["track_name"].isin(history)]
        song = (
            available_songs.sample().iloc[0]
            if not available_songs.empty
            else df.sample().iloc[0]
        )
    elif switch_type == "same":
        # Ensure current_song is defined to avoid KeyError
        if current_song is not None:
            suitable_songs = df[
                (
                    df["tempo"].between(
                        current_song["tempo"] - 5, current_song["tempo"] + 5
                    )
                )
                & (df["danceability"] >= current_song["danceability"])
                & (df["energy"] >= current_song["energy"])
                & (~df["track_name"].isin(history))
                & (df["track_name"] != current_song["track_name"])
            ]
            song = (
                suitable_songs.sample().iloc[0]
                if not suitable_songs.empty
                else current_song
            )
        else:
            song = df.sample().iloc[0]  # Default selection if no current song
    else:
        song = df.sample().iloc[0]  # Fallback to random selection

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


def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    current_pos = pygame.mixer.music.get_pos() / 1000.0  # convert ms to seconds
    start_times = calculate_8bar_starts(
        current_song["tempo"], len(current_song["local_path"]), current_pos
    )
    if start_times:
        transition_point = min(start_times, key=lambda x: abs(x - current_pos))
        transition_duration_ms = calculate_transition_timing(
            current_song["tempo"], 8, 4
        )
        transitioned_track, _ = gradual_high_pass_blend_transition(
            current_song,
            next_song,
            transition_point + transition_duration_ms,
            next_song["beat_drop_ms"],
            current_song["tempo"],
            next_song["tempo"],
        )
        pygame.mixer.music.load(transitioned_track.export(format="wav"))
        pygame.mixer.music.play()
    else:
        print("No valid transition points found, playing next song immediately.")
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
        total = 0
        pos, neg, neutral = 0, 0, 0

        for p in pred["face"]["predictions"]:
            emotions = p["emotions"]
            top_emotion = sorted(emotions, key=lambda x: x["score"], reverse=True)[0]
            if top_emotion["name"] in NEGATIVE_EMOTIONS:
                neg += 1
            elif top_emotion["name"] in POSTIVE_EMOTIONS:
                pos += 1
            else:
                neutral += 0

        # Determine which sentiment category has the highest count
        max_count = max(pos, neg, neutral)
        if max_count == pos:
            return 1
        elif max_count == neg:
            return -1
        else:
            return 0

    else:
        return None


async def read_frames_and_call_api(websocket, path):
    cap = cv2.VideoCapture(0)

    # running window of sentiment from last 20 frames
    num_frames = 20
    sentiment_history = []

    history = []
    current_song = select_song()
    play_song(current_song)
    history.append(current_song["track_name"])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        payload = {
            "models": {"face": {"identify_faces": True}},
            "data": encode_frame(frame),
        }

        await websocket.send(json.dumps(payload))

        message = await websocket.recv()
        pred = json.loads(message)

        # update sentiment tracker
        sentiment_score = calculate_sentiment(
            pred
        )  # 1 for positive, 0 for negative, -1 for negative
        if sentiment_score:
            sentiment_history.append(sentiment_score)

        # Keep sentiment history within the specified number of frames
        if len(sentiment_history) > num_frames:
            sentiment_history.pop(0)

        # determine if we need to switch
        if sum(sentiment_history) < -20:  # means sentiment history is all negative
            print("emergency transition")
            transition_command = "switch"  # or whatever logic you have
            # Perform transition logic
            next_song = select_song(current_song, transition_command, history)
            handle_transition(current_song, next_song)
            current_song = next_song
            history.append(current_song["track_name"])
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

        # Display the frame
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


asyncio.get_event_loop().run_until_complete(connect_to_websocket())
