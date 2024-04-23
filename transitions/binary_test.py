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

    if switch_type == 'switch' or current_song is None:
        # Filter out recently played songs if possible
        available_songs = df[~df['track_name'].isin(history)]
        song = available_songs.sample().iloc[0] if not available_songs.empty else df.sample().iloc[0]
    elif switch_type == 'same':
        # Ensure current_song is defined to avoid KeyError
        if current_song is not None:
            suitable_songs = df[(df['tempo'].between(current_song['tempo'] - 5, current_song['tempo'] + 5)) &
                                (df['danceability'] >= current_song['danceability']) &
                                (df['energy'] >= current_song['energy']) &
                                (~df['track_name'].isin(history)) &
                                (df['track_name'] != current_song['track_name'])]
            song = suitable_songs.sample().iloc[0] if not suitable_songs.empty else current_song
        else:
            song = df.sample().iloc[0]  # Default selection if no current song
    else:
        song = df.sample().iloc[0]  # Fallback to random selection

    print(f"Selected Song: {song['track_name']} - Command: {switch_type}")  # Debug statement
    return song

def play_song(song):
    """Play a song using Pygame."""
    path = "../songs/" + song['local_path']
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"Playing: {path}")  # Debug statement

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
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("Space key pressed.")  # Debug statement
                command = random.choice(['switch', 'same'])
                print(f"Command: {command}")
                next_song = select_song(current_song, command, history)
                # handle_transition(current_song, next_song)
                current_song = next_song
                history.append(current_song['track_name'])
                if len(history) > 10:
                    history.pop(0)

pygame.quit()
