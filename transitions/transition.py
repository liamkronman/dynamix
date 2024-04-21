import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter

def read_wav(file_path):
    """Reads a WAV file and returns it as a PyDub audio segment."""
    return AudioSegment.from_wav(file_path)

def extract_samples(audio_segment, downsample_rate=100):
    """Extracts samples from an audio segment for plotting, with optional downsampling."""
    samples = np.array(audio_segment.get_array_of_samples())
    return samples[::downsample_rate] if downsample_rate > 1 else samples

def calculate_transition_timing(bpm, measures, beats_per_measure):
    """Calculates the duration of the transition in milliseconds based on BPM."""
    beat_duration_ms = (60 / bpm) * 1000
    return int(measures * beats_per_measure * beat_duration_ms)

def calculate_8bar_starts(bpm, track_length_ms, drop_ms):
    """Calculate the start times of all 8-bar sections in the track."""
    bar_duration_ms = (60 / bpm) * 4 * 1000  # Duration of 4 beats (1 bar)
    section_duration_ms = bar_duration_ms * 8  # Duration of 8 bars
    start_times = [drop_ms - (i * section_duration_ms) for i in range(int(drop_ms / section_duration_ms) + 1)]
    start_times += [drop_ms + (i * section_duration_ms) for i in range(1, int((track_length_ms - drop_ms) / section_duration_ms) + 1)]
    return sorted([time for time in start_times if 0 <= time <= track_length_ms])

def plot_detailed_waveforms(track1_samples, track2_samples, filtered_samples, final_samples, transition_start_ms, transition_duration_ms, sample_rate, downsample_rate):
    """Plots detailed waveforms including the filtered and final combined tracks."""
    # Ensure that the length of the time axis matches the length of the downsampled samples
    downsampled_length = lambda samples: (len(samples) // downsample_rate) * downsample_rate  # Ensure divisibility
    time_axis = lambda samples: np.linspace(0, downsampled_length(samples) / sample_rate, num=downsampled_length(samples) // downsample_rate)

    titles = ['Original Track 1', 'Original Track 2', 'Filtered Track 1', 'Final Combined Track']
    plt.figure(figsize=(20, 16))

    for i, samples in enumerate([track1_samples, track2_samples, filtered_samples, final_samples]):
        downsampled_samples = samples[:downsampled_length(samples):downsample_rate]  # Trim to a divisible length before downsampling
        
        plt.subplot(4, 1, i+1)
        plt.plot(time_axis(samples), downsampled_samples, label=titles[i])
        plt.axvline(x=transition_start_ms / 1000, color='red', linestyle='--', label=f"Transition Start @ {transition_start_ms / 1000}s")
        plt.title(titles[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()
    
def gradual_high_pass_blend_transition(track1, track2, start_ms1, start_ms2, transition_duration_ms):
    # Points of the high-pass filter start and blend start
    filter_start_ms1 = start_ms1 - transition_duration_ms
    blend_start_ms2 = start_ms2 - (transition_duration_ms // 2)

    # Segments before the filter and blend
    pre_filter = track1[:filter_start_ms1]

    # Segments that will be processed
    filter_segment = track1[filter_start_ms1:start_ms1]
    blend_segment = track2[blend_start_ms2:start_ms2]

    post_transition = track2[start_ms2:]

    # Prepare segments for transition
    steps = 100
    step_ms = transition_duration_ms // steps
    filtered_segment = AudioSegment.silent(duration=0)
    
    # Define the number of blending steps
    blend_steps = steps // 2
    blend_step_ms = (transition_duration_ms // 2) // blend_steps

    # Create an initial silent segment for the incoming blend
    incoming_blend = AudioSegment.silent(duration=blend_step_ms * blend_steps)

    max_low_pass_freq = 5000  # The initial low-pass filter frequency

    min_low_pass_freq = 16000

    # Apply the high-pass filter and volume increase gradually
    for i in range(steps):
        current_ms = i * step_ms
        cutoff = 100 + (i / steps) * (5000 - 100)  # Gradually increase cutoff frequency

        # Apply high-pass filter to each step segment of track1
        step_segment = filter_segment[current_ms:current_ms + step_ms]
        filtered_step = high_pass_filter(step_segment, cutoff)
        filtered_segment += filtered_step

        # Gradually increase volume for the incoming segment of track2
        if i >= steps // 2:
            blend_index = i - (steps // 2)
            blend_current_ms = blend_index * blend_step_ms
            incoming_segment = blend_segment[blend_current_ms:blend_current_ms + blend_step_ms]

            # Calculate the fade-in factor as a ratio (0.0 to 1.0)
            fade_in_factor = blend_index / blend_steps
            
            # Convert the fade-in factor to a gain value in decibels
            # The dB change for a full volume track would be 0 dB,
            # and for silence it would be negative infinity, but we can use -120 dB as a practical "silence" level.
            # Here we interpolate between -infinity dB (silence) and 0 dB (full volume)
            volume_increase_db = 20 * np.log10(fade_in_factor) if fade_in_factor > 0 else -120
    
            # Apply a low-pass filter that gradually increases in frequency to normal
            current_low_pass_freq = max_low_pass_freq - (max_low_pass_freq - min_low_pass_freq) * (blend_index / blend_steps)  # Gradually approach 500 Hz
            incoming_segment = low_pass_filter(incoming_segment, current_low_pass_freq)
            incoming_segment = incoming_segment + volume_increase_db
            
            # Overlay the incoming segment onto the incoming_blend segment
            incoming_blend = incoming_blend.overlay(incoming_segment, position=blend_current_ms)

    # Combine filtered and incoming blend segments
    filtered_segment = filtered_segment.overlay(incoming_blend, position=transition_duration_ms // 2)

    # Combine all segments into the final track
    final_track = pre_filter + filtered_segment + post_transition
    return final_track, filtered_segment


def main(track1_path, track2_path, beat_drop_track1_s, beat_drop_track2_s, bpm, measures, beats_per_measure, transition):
    track1 = read_wav(track1_path)
    track2 = read_wav(track2_path)
    transition_duration_ms = calculate_transition_timing(bpm, measures, beats_per_measure)
    beat_drop_track1_ms = beat_drop_track1_s * 1000
    beat_drop_track2_ms = beat_drop_track2_s * 1000

    if transition == "blend1":
        transitioned_track, filtered_segment = gradual_high_pass_blend_transition(track1, track2, beat_drop_track1_ms, beat_drop_track2_ms, transition_duration_ms)
        transitioned_track.export("gradual_high_pass_blend_transition_output.wav", format="wav")
        print("Gradual high-pass blend transition completed and file exported.")
    else:
        print("Invalid transition type specified.")
        return

    track1_samples = extract_samples(track1, 100)
    track2_samples = extract_samples(track2, 100)
    filtered_samples = extract_samples(filtered_segment, 100)
    final_samples = extract_samples(transitioned_track, 100)

    plot_detailed_waveforms(
        track1_samples, 
        track2_samples, 
        filtered_samples, 
        final_samples, 
        beat_drop_track1_ms - transition_duration_ms, 
        transition_duration_ms, 
        track1.frame_rate, 
        downsample_rate=441
    )

    print("Transition completed and file exported.")

if __name__ == "__main__":
    main("../songs/clarity.wav", "../songs/die_young.wav", 39.3, 37.5, 128, 8, 4, "blend1")
