import numpy as np
import librosa
import ffmpeg
import os
import matplotlib.pyplot as plt
from scipy import signal
import argparse
from tqdm import tqdm
import tempfile
import subprocess
from pydub import AudioSegment
from pydub.silence import detect_silence

def extract_audio_from_mp4(input_file, temp_audio_file):
    """Extract audio from an MP4 file to a temporary WAV file for analysis"""
    print(f"Extracting audio from {input_file}...")
    (
        ffmpeg
        .input(input_file)
        .output(temp_audio_file, acodec='pcm_s16le', ar='44100', ac=2)
        .run(quiet=True, overwrite_output=True)
    )
    return temp_audio_file

def detect_song_boundaries(audio_file, num_songs, min_silence_len=1000, silence_thresh=-40, 
                           min_song_duration=30000, plot=False):
    """
    Detect song boundaries in a live set using silence detection and energy analysis
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    num_songs : int
        Number of songs expected in the recording
    min_silence_len : int
        Minimum length of silence (in ms) to consider as a potential boundary
    silence_thresh : int
        Threshold (in dB) below which to consider as silence
    min_song_duration : int
        Minimum duration of a song in milliseconds (default: 30000 = 30 seconds)
    plot : bool
        Whether to display analysis plots
    
    Returns:
    --------
    tuple
        (song_boundaries, talking_segments)
        song_boundaries: List of (start, end) times for each song in milliseconds
        talking_segments: List of (start, end) times for talking segments in milliseconds
    """
    print("Loading audio file for analysis...")
    # Load using pydub for silence detection
    audio = AudioSegment.from_file(audio_file)
    total_duration = len(audio)
    
    # Detect silent sections
    print("Detecting silence...")
    silence_sections = detect_silence(audio, 
                                    min_silence_len=min_silence_len, 
                                    silence_thresh=silence_thresh)
    
    # Convert silence sections to milliseconds
    silence_sections = [(start, end) for start, end in silence_sections]
    
    # Load audio using librosa for energy analysis
    y, sr = librosa.load(audio_file, sr=None)
    
    # Calculate the RMS energy
    print("Calculating energy profile...")
    frame_length = int(sr * 0.5)  # 0.5 second frames
    hop_length = int(sr * 0.1)    # 0.1 second hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate the time for each RMS frame
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    times_ms = times * 1000  # Convert to milliseconds
    
    # Smooth the RMS curve to reduce noise
    window_size = 21
    rms_smooth = signal.savgol_filter(rms, window_size, 3)
    
    # Trim beginning and end: Find the first and last significant energy points
    threshold = 0.1 * np.max(rms_smooth)  # 10% of max energy as threshold
    significant_energy_indices = np.where(rms_smooth > threshold)[0]
    
    if len(significant_energy_indices) > 0:
        first_energy_idx = significant_energy_indices[0]
        last_energy_idx = significant_energy_indices[-1]
        
        # Convert to milliseconds
        first_energy_time = times_ms[first_energy_idx]
        last_energy_time = times_ms[last_energy_idx]
        
        # Add a small buffer
        first_energy_time = max(0, first_energy_time - 1000)  # 1 second buffer at start
        last_energy_time = min(total_duration, last_energy_time + 1000)  # 1 second buffer at end
    else:
        # Fallback if no significant energy found
        first_energy_time = 0
        last_energy_time = total_duration
    
    # Find the longest silence sections
    long_silences = []
    for start, end in silence_sections:
        duration = end - start
        # Skip silences at the very beginning or end
        if start < first_energy_time + min_silence_len or end > last_energy_time - min_silence_len:
            continue
        
        if duration > min_silence_len * 2:  # Consider longer silences as significant
            long_silences.append((start, end, duration))
    
    # Sort by duration (longest first)
    long_silences.sort(key=lambda x: x[2], reverse=True)
    
    # Select the top N-1 longest silence sections as potential song boundaries
    # where N is the number of songs
    potential_boundaries = []
    if len(long_silences) >= num_songs - 1:
        potential_boundaries = [(start, end) for start, end, _ in long_silences[:num_songs-1]]
    else:
        # If not enough long silences, supplement with energy minima
        # Find local minima in the smoothed RMS
        rel_min_idx = signal.argrelextrema(rms_smooth, np.less, order=50)[0]
        
        # Filter out minima at the very beginning or end
        rel_min_idx = [idx for idx in rel_min_idx if 
                      times_ms[idx] > first_energy_time + min_silence_len and 
                      times_ms[idx] < last_energy_time - min_silence_len]
        
        # Convert frame indices to milliseconds
        min_times_ms = [times_ms[idx] for idx in rel_min_idx]
        
        # Sort minima by their RMS energy value
        sorted_mins = sorted([(time_ms, rms_smooth[idx]) for time_ms, idx in zip(min_times_ms, rel_min_idx)], 
                           key=lambda x: x[1])
        
        # Take enough lowest energy points to reach num_songs-1 boundaries
        needed = num_songs - 1 - len(long_silences)
        additional_points = [time_ms for time_ms, _ in sorted_mins[:needed]]
        
        # Convert single points to small ranges for consistency
        additional_boundaries = [(point - 100, point + 100) for point in additional_points]
        
        # Combine with long silences
        potential_boundaries.extend(additional_boundaries)
    
    # Sort boundaries by start time
    potential_boundaries.sort(key=lambda x: x[0])
    
    # Now define song segments based on these boundaries
    # Start from first significant energy point
    song_boundaries = []
    start_time = first_energy_time
    
    for i, (boundary_start, boundary_end) in enumerate(potential_boundaries):
        # End of current song
        song_end = boundary_start
        
        # Skip if the segment would be too short
        if song_end - start_time < min_song_duration:
            # If this would make a song too short, skip this boundary
            continue
        
        song_boundaries.append((start_time, song_end))
        
        # Start of next song
        start_time = boundary_end
    
    # Add the last song if it's long enough
    if last_energy_time - start_time >= min_song_duration:
        song_boundaries.append((start_time, last_energy_time))
    
    # If we have too few songs, try to split longer segments
    if len(song_boundaries) < num_songs:
        print(f"Warning: Only identified {len(song_boundaries)} song boundaries. "
              f"Expected {num_songs}. Some songs may not have clear boundaries.")
    
    # If we have too many songs, merge the shortest ones
    while len(song_boundaries) > num_songs:
        # Calculate song durations
        song_durations = [(end - start, i) for i, (start, end) in enumerate(song_boundaries)]
        
        # Sort by duration
        song_durations.sort(key=lambda x: x[0])
        
        # Get index of shortest song
        shortest_idx = song_durations[0][1]
        
        # Merge with adjacent song (prefer the shorter adjacent if possible)
        if shortest_idx == 0:
            # Merge with next song
            next_start, next_end = song_boundaries[1]
            song_boundaries[0] = (song_boundaries[0][0], next_end)
            song_boundaries.pop(1)
        elif shortest_idx == len(song_boundaries) - 1:
            # Merge with previous song
            prev_start, prev_end = song_boundaries[shortest_idx - 1]
            song_boundaries[shortest_idx - 1] = (prev_start, song_boundaries[shortest_idx][1])
            song_boundaries.pop(shortest_idx)
        else:
            # Merge with shorter adjacent
            prev_duration = song_boundaries[shortest_idx][0] - song_boundaries[shortest_idx-1][0]
            next_duration = song_boundaries[shortest_idx+1][1] - song_boundaries[shortest_idx+1][0]
            
            if prev_duration < next_duration:
                # Merge with previous
                song_boundaries[shortest_idx-1] = (song_boundaries[shortest_idx-1][0], 
                                                 song_boundaries[shortest_idx][1])
            else:
                # Merge with next
                song_boundaries[shortest_idx] = (song_boundaries[shortest_idx][0], 
                                              song_boundaries[shortest_idx+1][1])
                song_boundaries.pop(shortest_idx+1)
    
    # Identify talking segments (the boundaries themselves)
    talking_segments = []
    for i in range(len(song_boundaries) - 1):
        first_song_end = song_boundaries[i][1]
        second_song_start = song_boundaries[i+1][0]
        
        # Only consider it a talking segment if it's significant
        if second_song_start - first_song_end > min_silence_len * 2:
            talking_segments.append((first_song_end, second_song_start))
    
    # Plot the RMS and boundaries if requested
    if plot:
        fig = plt.figure(figsize=(15, 5))
        plt.plot(times, rms, alpha=0.5, label='RMS Energy')
        plt.plot(times, rms_smooth, label='Smoothed RMS')
        
        # Plot trim points
        plt.axvline(x=first_energy_time/1000, color='purple', linestyle='--', label='First Energy')
        plt.axvline(x=last_energy_time/1000, color='purple', linestyle='--', label='Last Energy')
        
        # Plot song boundaries
        for start, end in song_boundaries:
            plt.axvspan(start/1000, end/1000, alpha=0.2, color='green')
        
        # Plot talking segments
        for start, end in talking_segments:
            plt.axvspan(start/1000, end/1000, alpha=0.3, color='red')
        
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.title('Audio Energy and Song Boundaries')
        plt.legend()
        fig.savefig(audio_file + '+_audio_energy.png')
        plt.close(fig)
    
    return song_boundaries, talking_segments

def split_mp4(input_file, output_prefix, song_boundaries, talking_segments, 
              output_format="mp4", include_talking=True):
    """
    Split an MP4 file into separate files based on detected song boundaries
    
    Parameters:
    -----------
    input_file : str
        Path to the input MP4 file
    output_prefix : str
        Prefix for output filenames
    song_boundaries : list
        List of (start, end) times for each song in milliseconds
    talking_segments : list
        List of (start, end) times for talking segments in milliseconds
    output_format : str
        Output format (mp4 or aac)
    include_talking : bool
        Whether to include talking segments as separate files
    """
    print(f"Splitting {input_file} into {len(song_boundaries)} songs...")
    
    # Determine codec based on output format
    if output_format.lower() == "aac":
        codec = "aac"
        ext = "aac"
    else:
        codec = "copy"  # Use copy for mp4 to avoid re-encoding video
        ext = "mp4"
    
    # Split songs
    for i, (start, end) in enumerate(song_boundaries):
        output_file = f"{output_prefix}song_{i+1}.{ext}"
        
        # Convert milliseconds to seconds for ffmpeg
        start_sec = start / 1000
        duration_sec = (end - start) / 1000
        
        print(f"Extracting song {i+1}: {start_sec:.2f}s to {start_sec + duration_sec:.2f}s " + 
              f"(Duration: {duration_sec:.2f} seconds)")
        
        try:
            # Use ffmpeg to extract the segment
            if output_format.lower() == "mp4":
                (
                    ffmpeg
                    .input(input_file, ss=start_sec, t=duration_sec)
                    .output(output_file, c=codec)
                    .run(quiet=True, overwrite_output=True)
                )
            else:  # AAC format
                (
                    ffmpeg
                    .input(input_file, ss=start_sec, t=duration_sec)
                    .output(output_file, c="aac")
                    .run(quiet=True, overwrite_output=True)
                )
            
            print(f"Saved {output_file}")
        
        except ffmpeg.Error as e:
            print(f"Error extracting song {i+1}: {e.stderr.decode()}")
    
    # Split talking segments if requested
    if include_talking:
        for i, (start, end) in enumerate(talking_segments):
            output_file = f"{output_prefix}talking_{i+1}.{ext}"
            
            # Convert milliseconds to seconds for ffmpeg
            start_sec = start / 1000
            duration_sec = (end - start) / 1000
            
            print(f"Extracting talking segment {i+1}: {start_sec:.2f}s to {start_sec + duration_sec:.2f}s " + 
                  f"(Duration: {duration_sec:.2f} seconds)")
            
            try:
                # Use ffmpeg to extract the segment
                if output_format.lower() == "mp4":
                    (
                        ffmpeg
                        .input(input_file, ss=start_sec, t=duration_sec)
                        .output(output_file, c=codec)
                        .run(quiet=True, overwrite_output=True)
                    )
                else:  # AAC format
                    (
                        ffmpeg
                        .input(input_file, ss=start_sec, t=duration_sec)
                        .output(output_file, c="aac")
                        .run(quiet=True, overwrite_output=True)
                    )
                
                print(f"Saved {output_file}")
            
            except ffmpeg.Error as e:
                print(f"Error extracting talking segment {i+1}: {e.stderr.decode()}")

def process_live_set(input_file, num_songs, output_prefix="", output_format="mp4", 
                     min_silence_len=1500, silence_thresh=-40, min_song_duration=30000,
                     include_talking=True, plot=False):
    """
    Process a live set MP4 recording - detect song boundaries and split into individual files
    
    Parameters:
    -----------
    input_file : str
        Path to the input MP4 file
    num_songs : int
        Number of songs in the live set
    output_prefix : str
        Prefix for output filenames
    output_format : str
        Output format (mp4 or aac)
    min_silence_len : int
        Minimum length of silence (in ms) to consider as a potential boundary
    silence_thresh : int
        Threshold (in dB) below which to consider as silence
    min_song_duration : int
        Minimum duration of a song in milliseconds (default: 30000 = 30 seconds)
    include_talking : bool
        Whether to include talking segments as separate files
    plot : bool
        Whether to display analysis plots
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio for analysis
        temp_audio_file = os.path.join(temp_dir, "audio_extract.wav")
        extract_audio_from_mp4(input_file, temp_audio_file)
        
        # Detect song boundaries
        song_boundaries, talking_segments = detect_song_boundaries(
            temp_audio_file, num_songs, min_silence_len, silence_thresh, 
            min_song_duration, plot)
        
        # Split the MP4 file
        split_mp4(input_file, output_prefix, song_boundaries, talking_segments, 
                  output_format, include_talking)
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an MP4 recording of a live music set into individual songs")
    
    parser.add_argument("input_file", help="Path to the input MP4 file")
    parser.add_argument("num_songs", type=int, help="Number of songs in the recording")
    parser.add_argument("--output-prefix", default="", help="Prefix for output filenames")
    parser.add_argument("--output-format", default="mp4", choices=["mp4", "aac"], 
                        help="Output format (mp4 or aac)")
    parser.add_argument("--min-silence", type=int, default=1500, 
                        help="Minimum silence length (ms) to consider as a boundary")
    parser.add_argument("--silence-threshold", type=int, default=-40, 
                        help="Silence threshold (dB)")
    parser.add_argument("--min-song-duration", type=int, default=30000,
                        help="Minimum song duration in milliseconds (default: 30000 = 30 seconds)")
    parser.add_argument("--include-talking", action="store_true", 
                        help="Include talking segments as separate files")
    parser.add_argument("--plot", action="store_true", 
                        help="Display analysis plots")
    
    args = parser.parse_args()
    
    process_live_set(
        args.input_file, 
        args.num_songs,
        args.output_prefix,
        args.output_format,
        args.min_silence,
        args.silence_threshold,
        args.min_song_duration,
        args.include_talking,
        args.plot
    )
