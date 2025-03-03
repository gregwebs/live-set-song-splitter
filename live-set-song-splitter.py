import argparse
import numpy as np
import librosa
import os
import logging
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_from_mp4(input_file, output_audio=None):
    """Extract audio from an MP4 file"""
    logging.info(f"Extracting audio from {input_file}")
    video = VideoFileClip(input_file)
    if output_audio:
        video.audio.write_audiofile(output_audio, codec='pcm_s16le')
        return output_audio
    else:
        # Return as numpy array
        return np.array(video.audio.to_soundarray())

def detect_song_boundaries(audio_file, expected_songs, min_song_length_sec=30, 
                          silence_threshold=-50, min_silence_length_sec=3, 
                          max_silence_length_sec=30, plot=False):
    """
    Detect song boundaries in an audio file.
    
    Parameters:
    - audio_file: Path to the audio file
    - expected_songs: Number of songs expected in the recording
    - min_song_length_sec: Minimum duration of a song in seconds
    - silence_threshold: Threshold in dB to consider as silence
    - min_silence_length_sec: Minimum silence duration to consider a boundary
    - max_silence_length_sec: Maximum silence duration to include in a song
    - plot: Whether to plot the audio energy for visualization
    
    Returns:
    - list of (start, end) tuples for each song
    - list of (start, end) tuples for each talking segment
    """
    logging.info(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    logging.info(f"Audio duration: {duration:.2f} seconds")
    
    # Compute the short-time energy in dB
    frame_length = int(sr * 0.025)  # 25ms frame
    hop_length = int(sr * 0.010)    # 10ms hop
    
    # Calculate energy in dB for each frame
    S = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Smooth the energy curve
    S_db_smooth = median_filter(S_db, size=15)
    
    # Find silence regions
    silence_mask = S_db_smooth < silence_threshold
    
    # Convert frame indices to time
    frame_times = librosa.frames_to_time(np.arange(len(S_db_smooth)), sr=sr, hop_length=hop_length)
    
    # Find continuous regions of silence
    silence_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_silence in enumerate(silence_mask):
        if is_silence and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = frame_times[i]
        elif not is_silence and in_silence:
            # End of silence
            in_silence = False
            silence_end = frame_times[i]
            silence_duration = silence_end - silence_start
            
            # Only consider silences longer than min_silence_length_sec
            if silence_duration >= min_silence_length_sec:
                silence_regions.append((silence_start, silence_end))
    
    # Handle the case where the file ends during silence
    if in_silence:
        silence_regions.append((silence_start, duration))
    
    logging.info(f"Found {len(silence_regions)} potential song boundaries")
    
    # Trim beginning and end silences
    active_audio_start = 0
    active_audio_end = duration
    
    if silence_regions and silence_regions[0][0] == 0:
        active_audio_start = silence_regions[0][1]
        silence_regions.pop(0)
    
    if silence_regions and silence_regions[-1][1] == duration:
        active_audio_end = silence_regions[-1][0]
        silence_regions.pop(-1)
    
    logging.info(f"Active audio: {active_audio_start:.2f}s to {active_audio_end:.2f}s")
    
    # If we have more silence regions than expected song boundaries, 
    # select the most significant ones
    if len(silence_regions) > expected_songs - 1:
        # Sort by silence duration
        sorted_silences = sorted(silence_regions, 
                                key=lambda x: x[1] - x[0], 
                                reverse=True)
        
        # Take the top N longest silences
        significant_silences = sorted_silences[:expected_songs - 1]
        # Sort back by time
        silence_regions = sorted(significant_silences, key=lambda x: x[0])
        
        logging.info(f"Selected {len(silence_regions)} most significant silences as song boundaries")
    
    # Create song boundaries from silence regions
    song_boundaries = []
    last_end = active_audio_start
    
    for silence_start, silence_end in silence_regions:
        # Check if this silence is a "talking" segment (very long silence)
        if silence_end - silence_start > max_silence_length_sec:
            # End the current song at the start of the talking
            song_boundaries.append((last_end, silence_start))
            # Mark this as a talking segment (to be separated later)
            song_boundaries.append(("talking", silence_start, silence_end))
            last_end = silence_end
        else:
            # For shorter silences, end the song at the middle of the silence
            mid_silence = (silence_start + silence_end) / 2
            song_boundaries.append((last_end, mid_silence))
            last_end = mid_silence
    
    # Add the final song
    song_boundaries.append((last_end, active_audio_end))
    
    # Filter out any songs that are too short
    min_song_length = min_song_length_sec  # in seconds
    filtered_boundaries = []
    talk_segments = []
    
    for boundary in song_boundaries:
        if boundary[0] == "talking":
            talk_segments.append((boundary[1], boundary[2]))
        elif boundary[1] - boundary[0] >= min_song_length:
            filtered_boundaries.append(boundary)
    
    logging.info(f"Identified {len(filtered_boundaries)} songs and {len(talk_segments)} talking segments")
    
    # Plotting for visualization
    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(frame_times, S_db_smooth)
        plt.axhline(y=silence_threshold, color='r', linestyle='-', label=f'Threshold ({silence_threshold} dB)')
        
        # Plot song boundaries
        for start, end in filtered_boundaries:
            plt.axvline(x=start, color='g', linestyle='--')
            plt.axvline(x=end, color='r', linestyle='--')
        
        # Plot talking segments
        for start, end in talk_segments:
            plt.axvspan(start, end, alpha=0.2, color='yellow')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (dB)')
        plt.title('Audio Energy and Detected Song Boundaries')
        plt.legend()
        plt.tight_layout()
        plt.savefig('song_detection.png')
        logging.info("Saved detection visualization to song_detection.png")
    
    return filtered_boundaries, talk_segments

def extract_segments(input_file, output_dir, song_boundaries, talk_segments, is_video=True):
    """
    Extract audio segments based on detected boundaries
    
    Parameters:
    - input_file: Path to the input file
    - output_dir: Directory to save output files
    - song_boundaries: List of (start, end) tuples for songs
    - talk_segments: List of (start, end) tuples for talking segments
    - is_video: Whether the input is an MP4 video file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if is_video:
        video = VideoFileClip(input_file)
        
        # Extract songs
        for i, (start, end) in enumerate(song_boundaries):
            output_file = os.path.join(output_dir, f"song_{i+1}.mp4")
            logging.info(f"Extracting song {i+1} ({start:.2f}s - {end:.2f}s) to {output_file}")
            
            segment = video.subclip(start, end)
            segment.write_videofile(output_file, audio_codec='aac')
        
        # Extract talking segments
        for i, (start, end) in enumerate(talk_segments):
            output_file = os.path.join(output_dir, f"talking_{i+1}.mp4")
            logging.info(f"Extracting talking segment {i+1} ({start:.2f}s - {end:.2f}s) to {output_file}")
            
            segment = video.subclip(start, end)
            segment.write_videofile(output_file, audio_codec='aac')
        
        video.close()
    else:
        # Handle audio-only files
        y, sr = librosa.load(input_file, sr=None)
        
        # Extract songs
        for i, (start, end) in enumerate(song_boundaries):
            output_file = os.path.join(output_dir, f"song_{i+1}.wav")
            logging.info(f"Extracting song {i+1} ({start:.2f}s - {end:.2f}s) to {output_file}")
            
            start_frame = int(start * sr)
            end_frame = int(end * sr)
            sf.write(output_file, y[start_frame:end_frame], sr)
        
        # Extract talking segments
        for i, (start, end) in enumerate(talk_segments):
            output_file = os.path.join(output_dir, f"talking_{i+1}.wav")
            logging.info(f"Extracting talking segment {i+1} ({start:.2f}s - {end:.2f}s) to {output_file}")
            
            start_frame = int(start * sr)
            end_frame = int(end * sr)
            sf.write(output_file, y[start_frame:end_frame], sr)

def main():
    parser = argparse.ArgumentParser(description='Split a live music recording into individual songs')
    parser.add_argument('input_file', help='Path to the input MP4 file')
    parser.add_argument('output_dir', help='Directory to save extracted songs')
    parser.add_argument('--num-songs', type=int, required=True, help='Expected number of songs')
    parser.add_argument('--min-song-length', type=int, default=30, 
                        help='Minimum song length in seconds (default: 30)')
    parser.add_argument('--silence-threshold', type=float, default=-50, 
                        help='Silence threshold in dB (default: -50)')
    parser.add_argument('--min-silence', type=float, default=3, 
                        help='Minimum silence duration for song boundary in seconds (default: 3)')
    parser.add_argument('--max-silence', type=float, default=30, 
                        help='Maximum silence to include in song in seconds (default: 30)')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plot')
    parser.add_argument('--temp-audio', help='Path to save temporary audio file (optional)')
    
    args = parser.parse_args()
    
    # Check if input is a video file
    is_video = args.input_file.lower().endswith('.mp4')
    
    # For MP4 files, extract audio first for analysis
    if is_video:
        if args.temp_audio:
            temp_audio = args.temp_audio
        else:
            temp_audio = 'temp_audio.wav'
        extract_audio_from_mp4(args.input_file, temp_audio)
        audio_file = temp_audio
    else:
        audio_file = args.input_file
    
    # Detect song boundaries
    song_boundaries, talk_segments = detect_song_boundaries(
        audio_file, 
        args.num_songs,
        min_song_length_sec=args.min_song_length,
        silence_threshold=args.silence_threshold,
        min_silence_length_sec=args.min_silence,
        max_silence_length_sec=args.max_silence,
        plot=args.plot
    )
    
    # Extract segments
    extract_segments(args.input_file, args.output_dir, song_boundaries, talk_segments, is_video)
    
    # Clean up temporary file
    if is_video and not args.temp_audio and os.path.exists('temp_audio.wav'):
        os.remove('temp_audio.wav')
    
    logging.info(f"Successfully split {args.input_file} into {len(song_boundaries)} songs and {len(talk_segments)} talk segments")

if __name__ == "__main__":
    main()