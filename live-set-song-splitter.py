import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

def split_live_set(audio_file, num_songs, output_prefix="song_", plot=False):
    """
    Split a live audio recording into individual song files.
    
    Parameters:
    -----------
    audio_file : str
        Path to the input audio file
    num_songs : int
        Number of songs in the live set
    output_prefix : str
        Prefix for output filenames
    plot : bool
        Whether to display analysis plots
    """
    print(f"Loading audio file: {audio_file}")
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Calculate the RMS energy over time
    frame_length = int(sr * 0.5)  # 0.5 second frames
    hop_length = int(sr * 0.1)    # 0.1 second hop
    
    # Get the RMS energy
    print("Calculating RMS energy...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate the time for each RMS frame
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    # Smoothing the RMS curve to reduce noise
    print("Smoothing energy curve...")
    window_size = 21
    rms_smooth = signal.savgol_filter(rms, window_size, 3)
    
    # Find local minima in the smoothed RMS (these are potential song boundaries)
    print("Identifying potential song boundaries...")
    rel_min_idx = signal.argrelextrema(rms_smooth, np.less, order=50)[0]
    
    # If we have too many minima, select the ones with lowest energy
    if len(rel_min_idx) > num_songs - 1:
        # Sort the minima by their RMS energy value
        sorted_mins = sorted([(idx, rms_smooth[idx]) for idx in rel_min_idx], 
                             key=lambda x: x[1])
        
        # Take the num_songs-1 lowest energy points
        rel_min_idx = np.array([idx for idx, _ in sorted_mins[:num_songs-1]])
        rel_min_idx.sort()  # Sort back into chronological order
    
    # Convert frame indices to sample indices
    split_points = [0]  # Start of the file
    for idx in rel_min_idx:
        sample_idx = idx * hop_length
        split_points.append(sample_idx)
    split_points.append(len(y))  # End of the file
    
    # Plot the RMS and the split points if requested
    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(times, rms, alpha=0.5, label='RMS Energy')
        plt.plot(times, rms_smooth, label='Smoothed RMS')
        
        # Plot the split points
        split_times = [t * hop_length / sr for t in rel_min_idx]
        plt.vlines(split_times, 0, max(rms), color='r', linestyle='--', label='Split Points')
        
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.title('Audio Energy and Split Points')
        plt.legend()
        plt.show()
    
    # Save each song as a separate file
    print(f"Splitting into {num_songs} songs...")
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i+1]
        
        # Extract the segment
        segment = y[start:end]
        
        # Save to a file
        output_file = f"{output_prefix}{i+1}.wav"
        sf.write(output_file, segment, sr)
        
        duration = (end - start) / sr
        print(f"Saved {output_file} - Duration: {duration:.2f} seconds")
    
    print("Splitting complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a live audio recording into individual songs.")
    parser.add_argument("audio_file", help="Path to the input audio file")
    parser.add_argument("num_songs", type=int, help="Number of songs in the recording")
    parser.add_argument("--output-prefix", default="song_", help="Prefix for output filenames")
    parser.add_argument("--plot", action="store_true", help="Display analysis plots")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.audio_file):
        print("file does not exist " + args.audio_file)
        os._exit(1)

    split_live_set(args.audio_file, args.num_songs, args.output_prefix, args.plot)