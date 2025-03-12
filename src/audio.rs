use std::io::{BufReader, Read};
use std::fs::File;
use crate::ffmpeg::create_ffmpeg_command;

const SAMPLE_RATE: u32 = 44100;
const WINDOW_SIZE: usize = 4096;
const HOP_SIZE: usize = 1024;
const MIN_SILENCE_DURATION: f64 = 2.0; // Seconds of silence to detect a gap
const MIN_SONG_DURATION: f64 = 30.0; // Minimum song length in seconds
const ENERGY_THRESHOLD: f64 = 0.005; // Threshold for audio energy detection (lowered for better sensitivity)


#[derive(Clone, Debug)]
pub struct AudioSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub is_song: bool,
}

// const MAX_GAP_DURATION: f64 = 15.0; // Seconds - gaps longer than this are considered "talking" segments

pub fn extract_audio_waveform(input_file: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Create a temporary WAV file
    let temp_wav = "temp_audio.wav";

    // Extract audio to WAV using FFmpeg
    let status = create_ffmpeg_command()
        .args(&[
            "-i",
            input_file,
            "-vn", // No video
            "-acodec",
            "pcm_s16le", // PCM signed 16-bit little-endian
            "-ar",
            &SAMPLE_RATE.to_string(), // Sample rate
            "-ac",
            "1",  // Mono channel
            "-y", // Overwrite output file
            temp_wav,
        ])
        .status()?;

    if !status.success() {
        return Err("Failed to extract audio to WAV".into());
    }

    // Read the WAV file
    let file = File::open(temp_wav)?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // WAV header is 44 bytes, then PCM data follows
    let pcm_data = &buffer[44..];

    // Convert bytes to samples (16-bit signed integers)
    let mut samples = Vec::new();
    for i in 0..(pcm_data.len() / 2) {
        let sample = i16::from_le_bytes([pcm_data[i * 2], pcm_data[i * 2 + 1]]);
        samples.push(sample as f32 / 32768.0); // Normalize to [-1.0, 1.0]
    }

    // Clean up
    std::fs::remove_file(temp_wav)?;

    Ok(samples)
}

pub fn analyze_audio(
    samples: &[f32],
    expected_songs: usize,
    total_duration: f64,
) -> Result<Vec<AudioSegment>, Box<dyn std::error::Error>> {
    println!("Analyzing audio to detect songs and gaps...");

    // Calculate energy over time
    let energy_profile = calculate_energy_profile(samples);

    // Adaptive threshold based on the audio content
    let mean_energy: f64 = energy_profile.iter().sum::<f64>() / energy_profile.len() as f64;
    let adaptive_threshold = mean_energy * 0.25; // 25% of mean energy
    let threshold = adaptive_threshold
        .min(ENERGY_THRESHOLD)
        .max(ENERGY_THRESHOLD * 0.1);

    println!("Using energy threshold: {:.6}", threshold);

    // Find silence points using the adaptive threshold
    let silence_points = find_silence_points(&energy_profile, threshold);

    println!("Found {} potential silence points", silence_points.len());

    // Convert silence points to segments
    let mut segments = create_segments(silence_points, expected_songs, total_duration);

    // If we still don't have enough segments, try force splitting
    if segments.iter().filter(|s| s.is_song).count() < expected_songs {
        println!(
            "Not enough songs detected, forcing split into {} segments",
            expected_songs
        );
        segments = force_split_into_songs(total_duration, expected_songs);
    }

    Ok(segments)
}

fn calculate_energy_profile(samples: &[f32]) -> Vec<f64> {
    let mut energy_profile = Vec::new();

    // Calculate RMS energy for each window with hop_size step
    for window_start in (0..samples.len()).step_by(HOP_SIZE) {
        if window_start + WINDOW_SIZE > samples.len() {
            break;
        }

        // Calculate RMS for this window
        let sum_squared: f32 = samples[window_start..(window_start + WINDOW_SIZE)]
            .iter()
            .map(|&s| s * s)
            .sum();

        let rms = (sum_squared / WINDOW_SIZE as f32).sqrt();
        energy_profile.push(rms as f64);
    }

    // Apply a simple moving average to smooth the energy profile
    let window_size = (SAMPLE_RATE as usize / HOP_SIZE) / 2; // ~0.5 second window
    let mut smoothed_profile = Vec::with_capacity(energy_profile.len());

    for i in 0..energy_profile.len() {
        let start = if i < window_size { 0 } else { i - window_size };
        let end = std::cmp::min(i + window_size + 1, energy_profile.len());
        let avg = energy_profile[start..end].iter().sum::<f64>() / (end - start) as f64;
        smoothed_profile.push(avg);
    }

    smoothed_profile
}

fn find_silence_points(energy_profile: &[f64], threshold: f64) -> Vec<usize> {
    let mut silence_points = Vec::new();
    let frames_per_second = SAMPLE_RATE as f64 / HOP_SIZE as f64;
    let min_silence_frames = (MIN_SILENCE_DURATION * frames_per_second) as usize;

    let mut silence_start = None;
    let mut silence_length = 0;

    // Find silence spans
    for (i, &energy) in energy_profile.iter().enumerate() {
        if energy < threshold {
            // Low energy detected (silence)
            if silence_start.is_none() {
                silence_start = Some(i);
            }
            silence_length += 1;
        } else {
            // Energy above threshold (sound)
            if let Some(start) = silence_start {
                if silence_length >= min_silence_frames {
                    // We found a silence span that's long enough
                    let midpoint = start + silence_length / 2;
                    silence_points.push(midpoint);
                    println!(
                        "Silence detected at {:.2}s (length: {:.2}s)",
                        midpoint as f64 / frames_per_second,
                        silence_length as f64 / frames_per_second
                    );
                }
                silence_start = None;
                silence_length = 0;
            }
        }
    }

    // Check if we ended with silence
    if let Some(start) = silence_start {
        if silence_length >= min_silence_frames {
            let midpoint = start + silence_length / 2;
            silence_points.push(midpoint);
            println!(
                "Final silence detected at {:.2}s (length: {:.2}s)",
                midpoint as f64 / frames_per_second,
                silence_length as f64 / frames_per_second
            );
        }
    }

    silence_points
}

fn create_segments(
    silence_points: Vec<usize>,
    expected_songs: usize,
    total_duration: f64,
) -> Vec<AudioSegment> {
    let mut segments = Vec::new();
    let frames_per_second = SAMPLE_RATE as f64 / HOP_SIZE as f64;
    // let max_gap_frames = (MAX_GAP_DURATION * frames_per_second) as usize;
    // let min_song_frames = (MIN_SONG_DURATION * frames_per_second) as usize;

    // If no silence detected, try equal division
    if silence_points.is_empty() {
        return force_split_into_songs(total_duration, expected_songs);
    }

    // Create potential splitting points from silence
    let mut split_points = Vec::new();
    split_points.push(0.0); // Start of the file

    for point in silence_points {
        let time = point as f64 / frames_per_second;
        split_points.push(time);
    }

    split_points.push(total_duration); // End of the file

    // Create segments between split points
    for i in 0..split_points.len() - 1 {
        let start_time = split_points[i];
        let end_time = split_points[i + 1];
        let duration = end_time - start_time;

        // Only consider as a song if it's long enough
        let is_song = duration >= MIN_SONG_DURATION;

        segments.push(AudioSegment {
            start_time,
            end_time,
            is_song,
        });
    }

    // Post-process: merge short segments that are probably not songs
    let mut i = 0;
    while i < segments.len() - 1 {
        let current_duration = segments[i].end_time - segments[i].start_time;
        let next_duration = segments[i + 1].end_time - segments[i + 1].start_time;

        // If both current and next segments are short, merge them
        if current_duration < MIN_SONG_DURATION && next_duration < MIN_SONG_DURATION {
            segments[i].end_time = segments[i + 1].end_time;
            segments[i].is_song = (current_duration + next_duration) >= MIN_SONG_DURATION;
            segments.remove(i + 1);
            // Don't increment i, so we check newly merged segment again
        } else {
            i += 1;
        }
    }

    // Count how many segments are currently marked as songs
    let song_count = segments.iter().filter(|s| s.is_song).count();
    println!("After initial analysis: {} potential songs", song_count);

    // Adjust segments to match expected song count if needed
    if song_count != expected_songs {
        adjust_segments_for_song_count(&mut segments, expected_songs);
    }

    segments
}

fn force_split_into_songs(total_duration: f64, expected_songs: usize) -> Vec<AudioSegment> {
    println!("Force splitting into {} equal segments", expected_songs);
    let mut segments = Vec::new();

    // Calculate segment duration
    let segment_duration = total_duration / expected_songs as f64;

    // Create equal segments
    for i in 0..expected_songs {
        let start_time = i as f64 * segment_duration;
        let end_time = (i + 1) as f64 * segment_duration;

        segments.push(AudioSegment {
            start_time,
            end_time,
            is_song: true,
        });
    }

    segments
}

fn adjust_segments_for_song_count(segments: &mut Vec<AudioSegment>, expected_songs: usize) {
    // Count how many segments are currently marked as songs
    let song_count = segments.iter().filter(|s| s.is_song).count();

    println!(
        "Adjusting segments: have {}, need {}",
        song_count, expected_songs
    );

    if song_count == expected_songs {
        return; // We have the right number of songs
    }

    if song_count < expected_songs {
        // We need to convert some non-song segments to songs
        // Start with the longest non-song segments
        let mut non_songs: Vec<(usize, f64)> = segments
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_song)
            .map(|(i, s)| (i, s.end_time - s.start_time))
            .collect();

        if non_songs.is_empty() {
            // If there are no non-song segments, we need to split existing songs
            let mut new_segments = Vec::new();
            for segment in segments.iter() {
                let duration = segment.end_time - segment.start_time;
                if duration > MIN_SONG_DURATION * 2.0 {
                    // This song is long enough to split in two
                    let mid_point = segment.start_time + duration / 2.0;
                    new_segments.push(AudioSegment {
                        start_time: segment.start_time,
                        end_time: mid_point,
                        is_song: true,
                    });
                    new_segments.push(AudioSegment {
                        start_time: mid_point,
                        end_time: segment.end_time,
                        is_song: true,
                    });
                } else {
                    new_segments.push(segment.clone());
                }

                // Stop if we have enough songs
                if new_segments.iter().filter(|s| s.is_song).count() >= expected_songs {
                    break;
                }
            }

            *segments = new_segments;
            return;
        }

        non_songs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (index, _) in non_songs.iter().take(expected_songs - song_count) {
            segments[*index].is_song = true;
        }
    } else {
        // We have too many songs, merge the shortest ones
        let mut songs: Vec<(usize, f64)> = segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_song)
            .map(|(i, s)| (i, s.end_time - s.start_time))
            .collect();

        songs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (index, _) in songs.iter().take(song_count - expected_songs) {
            segments[*index].is_song = false;
        }
    }
}