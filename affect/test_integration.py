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

# Load the custom font


df = pd.read_csv("../spotify/good_matched_song_data.csv")
pygame.init()
pygame.mixer.init()
font = pygame.font.Font(None, 36)
font_path = "../Press_Start_2P/PressStart2P-Regular.ttf"
title_font = pygame.font.Font(font_path, 64)

width, height = 1120, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Dynamix: Read the Room")

stream_width, stream_height = 800, 500
stream_x, stream_y = (width - stream_width) // 2, (height - stream_height) // 2
stream_rect = pygame.Rect(stream_x, stream_y, stream_width, stream_height)

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
if "positive_feedback_count" not in df.columns:
    df["positive_feedback_count"] = 0

def update_song_sentiments(current_song_id, sentiment_score):
    """Update the sentiment score of the song based on the current session's feedback."""
    if current_song_id in df.index:
        df.loc[current_song_id, 'play_count'] += 1
        # Calculate a rolling average of sentiment scores
        current_average = df.loc[current_song_id, 'sentiment_score']
        new_average = (current_average * (df.loc[current_song_id, 'play_count'] - 1) + sentiment_score) / df.loc[current_song_id, 'play_count']
        df.loc[current_song_id, 'sentiment_score'] = new_average

features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']
# df[features] = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # don't change TEMPO!

# Function to calculate similarity matrix
def calculate_similarity(df):
    return cosine_similarity(df[features])


def update_positive_feedback(track_id, feedback_events_count):
    decay_factor = 0.9  # Adjust this based on desired rate of decay
    df.loc[df["id"] != track_id, "positive_feedback_count"] *= decay_factor
    df.loc[df["id"] == track_id, "positive_feedback_count"] += 1


similarity_matrix = calculate_similarity(df)

emotional_scores = []

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
    global emotional_scores
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
    true_pos = current_pos + offset
    print(f"Current Position: {current_pos / 1000.0} seconds")
    print(f"Tempo: {current_song['tempo']} -> {next_song['tempo']}")
    bpm1 = current_song["tempo"]
    track_length_ms = len(read_wav("../songs/" + current_song["local_path"]))
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
                read_wav("../songs/" + current_song["local_path"]),
                read_wav("../songs/" + next_song["local_path"]),
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
            offset = (
                beat_drop_ms_next_song - transition_duration_ms - (next_start - true_pos)
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
        return None


async def read_frames_and_call_api(websocket, path):
    cap = cv2.VideoCapture(0)

    # Initial setup
    num_frames = 20
    sentiment_history = []
    headcount_history = []
    history = []
    current_song = select_song()
    play_song(current_song)
    history.append(current_song["track_name"])
    current_mood = "Neutral"  # Default mood

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Process frame for display
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                original_width, original_height = frame.shape[1], frame.shape[0]
                frame_rgb = np.rot90(frame_rgb)
                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                frame_surface = pygame.transform.scale(
                    frame_surface, (stream_width, stream_height)
                )

                # Fill the background and blit the frame
                screen.fill((0, 0, 0))
                screen.blit(frame_surface, (stream_x, stream_y))

                # Prepare frame data for sending
                encoded_frame = encode_frame(frame)
                payload = {
                    "models": {"face": {"identify_faces": True}},
                    "data": encoded_frame,
                }
                await websocket.send(json.dumps(payload))
                message = await websocket.recv()
                pred = json.loads(message)

                # Update headcount
                current_headcount = calculate_headcount(pred)
                headcount_history.append(current_headcount)
                if len(headcount_history) > num_frames:
                    headcount_history.pop(0)

                # Update sentiment and determine song transitions
                sentiment_score = calculate_sentiment(pred)
                if sentiment_score:
                    sentiment_history.append(sentiment_score)
                    current_mood = "Positive" if sentiment_score == 1 else "Negative"

                if len(sentiment_history) > num_frames:
                    sentiment_history.pop(0)

                if (
                    sum(sentiment_history) < -10
                    or current_headcount <= headcount_history[0] // 2
                ):  # Negative mood detected
                    # only initiate a transition if the beat drop has passed
                    if (
                        pygame.mixer.music.get_pos()
                        >= current_song["beat_drop_s"] * 1000
                    ):
                        print("Emergency transition")
                        transition_command = "switch"
                        next_song = select_song(
                            current_song, transition_command, history
                        )
                        handle_transition(current_song, next_song)
                        current_song = next_song
                        history.append(current_song["track_name"])
                        for _ in range(20):
                            sentiment_history.append(1)
                        if len(history) > 10:
                            history.pop(0)
                else:
                    print("Continue playing")

                if "predictions" in pred["face"]:
                    for p in pred["face"]["predictions"]:
                        draw_bounding_boxes_and_labels(
                            p, frame_surface, original_width, original_height
                        )

                # Display current song and mood
                display_song_and_mood(current_song, current_mood, current_headcount)

                # Update the display
                pygame.display.flip()

                if pygame.event.get(pygame.QUIT):
                    break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


def display_song_and_mood(current_song, current_mood, current_headcount):
    # Display song info and mood
    current_time = pygame.mixer.music.get_pos() // 1000
    song_info = f'Song: {current_song["track_name"]} - {current_time+offset//1000}s'
    mood_info = f"Mood: {current_mood}"
    headcount_info = f"Headcount: {current_headcount}"

    title_text = title_font.render("Dynamix", True, pygame.Color("white"))
    song_text = font.render(song_info, True, pygame.Color("white"))
    mood_text = font.render(mood_info, True, pygame.Color("white"))
    headcount_text = font.render(headcount_info, True, pygame.Color("white"))

    screen.blit(title_text, (10, 10))
    screen.blit(song_text, (10, 90))
    screen.blit(mood_text, (10, 130))
    screen.blit(headcount_text, (10, 160))


def draw_bounding_boxes_and_labels(
    prediction, surface, original_width, original_height
):
    # Calculate scaling factors
    scale_x = stream_width / original_width
    scale_y = stream_height / original_height

    # Adjust bounding box coordinates
    x = int(prediction["bbox"]["x"] * scale_x) + stream_x
    y = int(prediction["bbox"]["y"] * scale_y) + stream_y
    w = int(prediction["bbox"]["w"] * scale_x)
    h = int(prediction["bbox"]["h"] * scale_y)

    # Determine the top emotion and set color based on its type
    top_emotion = max(prediction["emotions"], key=lambda e: e["score"])
    if top_emotion["name"] in POSTIVE_EMOTIONS:
        color = (0, 255, 0)  # Green for positive
    elif top_emotion["name"] in NEGATIVE_EMOTIONS:
        color = (255, 0, 0)  # Red for negative
    else:
        color = (128, 128, 128)  # Gray for neutral

    # Draw bounding box
    pygame.draw.rect(screen, color, (x, y, w, h), 2)

    # Display the top three emotions with scores
    emotions = prediction["emotions"]
    top_three = sorted(
        sorted(emotions, key=lambda e: e["score"], reverse=True)[:3],
        key=lambda e: e["score"],
    )
    for i, emotion in enumerate(top_three):
        label = f"{emotion['name']}: {round(emotion['score'], 2)}"
        label_surface = font.render(label, True, color)
        screen.blit(label_surface, (x, y - (i + 1) * 20))

    # Display person id
    id = font.render(prediction["face_id"], True, color)
    screen.blit(id, (x, y - (i + 3) * 20))


if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(connect_to_websocket())
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pygame.quit()  # Ensure Pygame quits properly
