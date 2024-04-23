import pandas as pd
import pygame
from pydub import AudioSegment
from transition import read_wav, calculate_transition_timing, gradual_high_pass_blend_transition, calculate_8bar_starts
import random

# Load song features and setup Pygame
df = pd.read_csv("../spotify/matched_song_data.csv")
pygame.init()
pygame.mixer.init()

def select_song(current_song=None, switch_type=None, history=None):
    """Selects a song based on the current song and switch type, avoiding recent history."""
    if history is None:
        history = []

    if current_song is None or switch_type == 'switch':
        # Filter out recently played songs if possible
        available_songs = df[~df['track_name'].isin(history)]
        return available_songs.sample().iloc[0] if not available_songs.empty else df.sample().iloc[0]
    elif switch_type == 'same':
        # Find the closest song not in history and not the current song
        if current_song.name in df.index:
            df_temp = df.copy()
            df_temp['similarity_score'] = (
                abs(df_temp['tempo'] - current_song['tempo']) +
                abs(df_temp['danceability'] - current_song['danceability']) +
                abs(df_temp['energy'] - current_song['energy'])
            )
            suitable_songs = df_temp[(~df_temp['track_name'].isin(history)) & (df_temp['track_name'] != current_song['track_name'])]
            return suitable_songs.loc[suitable_songs['similarity_score'].idxmin()] if not suitable_songs.empty else current_song
        else:
            return df.sample().iloc[0]

def play_song(song):
    """Play a song using Pygame."""
    path = "../songs/" + song['local_path']
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    current_pos = pygame.mixer.music.get_pos() / 1000.0  # convert ms to seconds
    start_times = calculate_8bar_starts(current_song['tempo'], len(current_song['local_path']), current_pos)
    if start_times:
        transition_point = min(start_times, key=lambda x: abs(x - current_pos))
        transition_duration_ms = calculate_transition_timing(current_song['tempo'], 8, 4)
        transitioned_track, _ = gradual_high_pass_blend_transition(current_song, next_song, transition_point + transition_duration_ms, next_song['beat_drop_ms'], current_song['tempo'], next_song['tempo'])
        pygame.mixer.music.load(transitioned_track.export(format="wav"))
        pygame.mixer.music.play()
    else:
        print("No valid transition points found, playing next song immediately.")
        play_song(next_song)

history = []
current_song = select_song()
play_song(current_song)
history.append(current_song['track_name'])

running = True
while running:
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            running = False
        elif event.type is pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                command = random.choice(['switch', 'same'])  # This simulates a command based on external triggers
                print(f"Command received: {command}")
                next_song = select_song(current_song, command, history)
                print(f"Switching to {next_song['track_name']} by {next_song['artist']}")
                handle_transition(current_song, next_song)
                current_song = next_song
                history.append(current_song['track_name'])
                if len(history) > 10:
                    history.pop(0)  # limit history size

pygame.quit()
