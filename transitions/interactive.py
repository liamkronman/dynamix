import pygame
from pydub import AudioSegment
from transition import read_wav, calculate_transition_timing, gradual_high_pass_blend_transition, calculate_8bar_starts

def setup_pygame():
    """Initialize Pygame and return the display surface."""
    pygame.init()
    size = (600, 400)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Transition Trigger")
    pygame.font.init()
    return screen

def format_time(ms):
    """Format milliseconds into minutes:seconds."""
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000*60)) % 60)
    return f'{minutes:02}:{seconds:02}'

def main(track1_path, track2_path, beat_drop_track1_s, beat_drop_track2_s, bpm1, bpm2):
    track1 = read_wav(track1_path)
    track2 = read_wav(track2_path)
    drop_ms = beat_drop_track1_s * 1000
    beat_drop_track2_ms = beat_drop_track2_s * 1000
    track_length_ms = len(track1)
    start_times = calculate_8bar_starts(bpm1, track_length_ms, drop_ms)

    print("8-bar section starts:", start_times)

    screen = setup_pygame()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)  # Create a font object with default font and size 36
    info_font = pygame.font.Font(None, 24)  # Smaller font for info messages

    # Prepare the first track
    pygame.mixer.music.load(track1_path)
    pygame.mixer.music.play(-1)  # Loop indefinitely

    running = True
    transition_triggered = False
    time_offset = 0  # Initialize the time offset

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not transition_triggered:
                    current_pos = pygame.mixer.music.get_pos()  # Current position in milliseconds
                    next_start = next((time for time in start_times if time > current_pos), None)
                    if next_start is not None:
                        transition_duration_ms = calculate_transition_timing(bpm1, 8, 4)
                        transitioned_track, _ = gradual_high_pass_blend_transition(track1, track2, next_start+transition_duration_ms, beat_drop_track2_ms, bpm1, bpm2)
                        # Prepare and play the transitioned track
                        # Get current playback position
                        current_pos = pygame.mixer.music.get_pos()  # milliseconds
                        # Create new audio from the second track
                        new_audio = transitioned_track[current_pos:]
                        time_offset += current_pos  # Add current playback position to the offset
                        pygame.mixer.music.load(new_audio.export(format="wav"))
                        pygame.mixer.music.play()
                        transition_triggered = True

        screen.fill((0, 0, 0))
        current_time = pygame.mixer.music.get_pos() + time_offset
        time_text = format_time(current_time)  # Format the current time
        text_surface = font.render(time_text, True, (255, 255, 255))  # Render the text
        screen.blit(text_surface, (10, 10))  # Draw the text on the screen at position (10, 10)

        if not transition_triggered:
            message = "Hit space to start transition"
            message_surface = info_font.render(message, True, (255, 255, 255))
            screen.blit(message_surface, (screen.get_width() / 2 - message_surface.get_width() / 2, screen.get_height() / 2))
        elif current_time < next_start+transition_duration_ms:
            countdown_text = f"Transitioning..."
            countdown_surface = info_font.render(countdown_text, True, (255, 255, 255))
            screen.blit(countdown_surface, (screen.get_width() / 2 - countdown_surface.get_width() / 2, screen.get_height() / 2))
        else:
            message = "Transition complete!"
            message_surface = info_font.render(message, True, (255, 255, 255))
            screen.blit(message_surface, (screen.get_width() / 2 - message_surface.get_width() / 2, screen.get_height() / 2))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    # main("../songs/toxic.wav", "../songs/clarity.wav", 15.3, 39.3, 143, 128)
    # main("../songs/die_young.wav", "../songs/clarity.wav", 37.5, 39.3, 128, 128)
    main("../songs/clarity.wav", "../songs/die_young.wav", 39.3, 37.5, 128, 128)
