import pandas as pd
import pygame
from pydub import AudioSegment
from transition import read_wav, calculate_transition_timing, gradual_high_pass_blend_transition, calculate_8bar_starts
import random

# Load song features and setup Pygame
df = pd.read_csv("../spotify/good_matched_song_data.csv")
pygame.init()
pygame.mixer.init()

def select_song(current_song=None, switch_type=None, history=None):
    """Selects a song based on the current song and switch type, avoiding recent history and the current song."""
    if history is None:
        history = []

    df_filtered = df[~df['track_name'].isin(history)]  # Exclude all songs in history from the options

    if current_song is None or switch_type == 'switch':
        # Randomly choose from songs not in history
        song = df_filtered.sample().iloc[0] if not df_filtered.empty else df.sample().iloc[0]
    elif switch_type == 'same':
        # Filter songs to find the closest match not including the current song
        if current_song is not None:
            df_filtered = df_filtered[df_filtered['track_name'] != current_song['track_name']]  # Also exclude current song
            suitable_songs = df_filtered[
                (df_filtered['tempo'].between(current_song['tempo'] - 5, current_song['tempo'] + 5)) &
                (df_filtered['danceability'] >= current_song['danceability']) &
                (df_filtered['energy'] >= current_song['energy'])
            ]
            song = suitable_songs.nsmallest(1, 'tempo').iloc[0] if not suitable_songs.empty else current_song
        else:
            song = df_filtered.sample().iloc[0] if not df_filtered.empty else df.sample().iloc[0]
    else:
        song = df_filtered.sample().iloc[0] if not df_filtered.empty else df.sample().iloc[0]  # Fallback to random selection

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

offset = 0

def handle_transition(current_song, next_song):
    """Handle transition from current song to next song, considering the current play position."""
    global offset
    print(f"Offset: {offset / 1000.0} seconds")
    current_pos = pygame.mixer.music.get_pos()  # Get current position in milliseconds
    print(f"Current Position: {current_pos / 1000.0} seconds")
    print(f"Tempo: {current_song['tempo']} -> {next_song['tempo']}")
    bpm1 = current_song['tempo']
    track_length_ms = len(read_wav("../songs/" + current_song['local_path']))
    drop_ms = current_song['beat_drop_s'] * 1000  # Convert beat drop time to milliseconds
    start_times = calculate_8bar_starts(bpm1, track_length_ms, drop_ms)

    if 'beat_drop_s' in next_song:
        beat_drop_ms_next_song = next_song['beat_drop_s'] * 1000
        print(f"Next Song Beat Drop: {beat_drop_ms_next_song / 1000.0} seconds")
        next_start = next((time for time in start_times if time > current_pos), None)
        transition_duration_ms = calculate_transition_timing(bpm1, 8, 4)
        # Assuming transition function is set to handle two audio segments
        transitioned_track, _ = gradual_high_pass_blend_transition(
            read_wav("../songs/" + current_song['local_path']),
            read_wav("../songs/" + next_song['local_path']),
            next_start+transition_duration_ms,
            beat_drop_ms_next_song,
            current_song['tempo'],
            next_song['tempo']
        )
        
        # Export and play the transitioned track
        temp_path = f"temp_transitioned_track.wav"
        current_pos = pygame.mixer.music.get_pos()
        pygame.mixer.music.load(transitioned_track[current_pos+offset:].export(format="wav"))
        pygame.mixer.music.play()
        offset = beat_drop_ms_next_song - transition_duration_ms
    else:
        print("No beat drop info found, playing next song immediately.")
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
                handle_transition(current_song, next_song)
                current_song = next_song
                history.append(current_song['track_name'])
                if len(history) > 10:
                    history.pop(0)

pygame.quit()
