import pandas as pd
import pygame
from pydub import AudioSegment
from transition import read_wav, calculate_transition_timing, gradual_high_pass_blend_transition, calculate_8bar_starts
from threading import Thread
import asyncio
import cv2
import sys
import os

# Add directory to sys.path for stream.py imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../affect'))
from stream import connect_to_websocket, encode_frame, calculate_sentiment

# Load song features and setup Pygame
df = pd.read_csv("../spotify/matched_song_data.csv")
pygame.init()
pygame.mixer.init()

# Define global variable for current song and history
current_song = None
history = []

def select_song(switch_type=None):
    global current_song
    if switch_type == 'switch' or current_song is None:
        # Filter out recently played songs
        available_songs = df[~df['track_name'].isin(history)]
        current_song = available_songs.sample().iloc[0] if not available_songs.empty else df.sample().iloc[0]
    else:
        # Find a song with similar mood and close BPM
        suitable_songs = df[(df['tempo'].between(current_song['tempo'] - 5, current_song['tempo'] + 5)) &
                            (df['danceability'] >= current_song['danceability']) &
                            (df['energy'] >= current_song['energy']) &
                            (~df['track_name'].isin(history))]
        current_song = suitable_songs.sample().iloc[0] if not suitable_songs.empty else current_song
    return current_song

def play_song():
    global current_song
    path = "../songs/" + current_song['local_path']
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    history.append(current_song['track_name'])
    if len(history) > 10:
        history.pop(0)

def pygame_thread():
    print("thread started")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

def sentiment_analysis_loop():
    async def sentiment_control():
        async with connect_to_websocket() as websocket:
            while True:
                sentiment_score = await calculate_sentiment(websocket)  # Assume this returns the sentiment score
                if sentiment_score < 0:
                    select_song('switch')
                else:
                    select_song('same')
                # play_song()
    
    asyncio.run(sentiment_control())

if __name__ == "__main__":
    # Start Pygame in a separate thread
    pygame_thread = Thread(target=pygame_thread)
    pygame_thread.start()
    
    # Run the sentiment analysis loop
    sentiment_analysis_loop()
