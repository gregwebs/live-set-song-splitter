# live-set-song-splitter
Split a live performance set into individual songs

## Overview
This tool analyzes audio files to detect silence between songs in a live recording and splits the file into separate song tracks.

## Requirements
- Rust (with Cargo)
- FFmpeg (for audio analysis)
- MP4Box (for file splitting)

## Usage
```bash
cargo run -- <input_file> <num_songs>
```

Where:
- `input_file` is the path to your audio/video file
- `num_songs` is the expected number of songs in the recording

## How it works
1. Extracts and analyzes audio waveform data
2. Detects silence regions using energy thresholds
3. Identifies song boundaries
4. Uses MP4Box to split the original file into individual song files
