use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::Command;

const SAMPLE_RATE: u32 = 44100;
const WINDOW_SIZE: usize = 4096;
const HOP_SIZE: usize = 1024;
const MIN_SILENCE_DURATION: f64 = 2.0; // Seconds of silence to detect a gap
const MIN_SONG_DURATION: f64 = 30.0; // Minimum song length in seconds
const ENERGY_THRESHOLD: f64 = 0.005; // Threshold for audio energy detection (lowered for better sensitivity)

// const MAX_GAP_DURATION: f64 = 15.0; // Seconds - gaps longer than this are considered "talking" segments

#[derive(Clone, Debug)]
struct AudioSegment {
    start_time: f64,
    end_time: f64,
    is_song: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Song {
    title: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct SetList {
    artist: String,
    #[serde(rename = "setList")]
    set_list: Vec<Song>,
}

#[derive(Debug, Clone)]
struct VideoInfo {
    // Basic information
    duration: f64,
    #[allow(dead_code)]
    framerate: f64,
    start_time: f64,
    
    // Keyframe information
    keyframe_timestamps: Vec<f64>,
    avg_keyframe_interval: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_mp4_file> <setlist.json>", args[0]);
        eprintln!("Example JSON: {{\"artist\": \"Artist Name\", \"setList\": [{{\"title\": \"Song Title 1\"}}, {{\"title\": \"Song Title 2\"}}]}}");
        return Ok(());
    }

    let input_file = &args[1];
    let setlist_path = &args[2];

    // Parse the JSON setlist file
    let setlist_file = File::open(setlist_path)?;
    let setlist_reader = BufReader::new(setlist_file);
    let setlist: SetList = serde_json::from_reader(setlist_reader)?;

    let num_songs = setlist.set_list.len();

    println!("Analyzing file: {}", input_file);
    println!("Artist: {}", setlist.artist);
    println!("Expected number of songs: {}", num_songs);
    println!("Songs:");
    for (i, song) in setlist.set_list.iter().enumerate() {
        println!("  {}. {}", i + 1, song.title);
    }

    // Get all video information at once
    let video_info = get_video_info(input_file)?;
    println!("Total duration: {:.2} seconds", video_info.duration);

    // First try to detect song boundaries using text overlays
    println!("Attempting to detect song boundaries using text overlays...");
    let mut segments = detect_song_boundaries_from_text(input_file, &setlist.artist, &setlist.set_list, &video_info)?;
    for segment in &segments {
        println!("Segment: {:?}", segment);
    }
    
    // If text detection didn't find enough songs, fall back to audio analysis
    if segments.iter().filter(|s| s.is_song).count() < num_songs {
        println!("Text overlay detection didn't find all songs. Falling back to audio analysis...");
        
        // Extract audio waveform data using FFmpeg
        println!("Extracting audio waveform...");
        let audio_data = extract_audio_waveform(input_file)?;
        
        // Analyze audio to find segments
        segments = analyze_audio(&audio_data, num_songs, video_info.duration)?;
    }

    println!("Found {} segments", segments.len());
    for (i, segment) in segments.iter().enumerate() {
        println!(
            "Segment {}: {:.2}s to {:.2}s ({:.2}s) - {}",
            i + 1,
            segment.start_time,
            segment.end_time,
            segment.end_time - segment.start_time,
            if segment.is_song { "SONG" } else { "gap" }
        );
    }

    // Process each detected segment
    process_segments(input_file, &segments, &setlist.set_list)?;

    println!("Audio splitting complete!");
    Ok(())
}

fn get_video_info(input_file: &str) -> Result<VideoInfo, Box<dyn std::error::Error>> {
    println!("Analyzing video file metadata...");
    
    // Get basic video information in one call
    let basic_info_output = Command::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate:format=duration,start_time",
            "-of", "json",
            input_file,
        ])
        .output()?;

    if !basic_info_output.status.success() {
        return Err("Failed to get video information".into());
    }

    let info_json = String::from_utf8(basic_info_output.stdout)?;
    let info: serde_json::Value = serde_json::from_str(&info_json)?;
    
    // Extract duration
    let duration = info["format"]["duration"].as_str()
        .ok_or("Missing duration")?
        .parse::<f64>()?;
        
    // Extract start time
    let start_time = info["format"]["start_time"].as_str()
        .unwrap_or("0")
        .parse::<f64>()
        .unwrap_or(0.0);
        
    // Extract framerate
    let fps_str = info["streams"][0]["r_frame_rate"].as_str().ok_or("Missing framerate")?;
    let mut fps = 25.0; // Default fallback value
    if let Some((num, den)) = fps_str.split_once('/') {
        if let (Ok(n), Ok(d)) = (num.parse::<f64>(), den.parse::<f64>()) {
            if d > 0.0 {
                fps = n / d;
            }
        }
    }
    
    println!("Video duration: {}s, start time: {}s, framerate: {:.2} fps", 
             duration, start_time, fps);
    
    // Get keyframe timestamps
    println!("Extracting keyframe information...");
    let keyframe_data = Command::new("ffprobe")
        .args(&[
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=print_section=0",
            input_file,
        ])
        .output()?;
    
    let keyframe_data_str = String::from_utf8(keyframe_data.stdout)?;
    
    // Parse keyframe timestamps - format is "pts_time,flags" where flags contains "K" for keyframes
    let keyframe_timestamps: Vec<f64> = keyframe_data_str
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 && parts[1].contains('K') {
                // This is a keyframe, parse the timestamp
                parts[0].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect();
    
    println!("Found {} keyframes", keyframe_timestamps.len());
    
    // Calculate average keyframe interval
    let avg_keyframe_interval = if keyframe_timestamps.len() >= 2 {
        let interval = (keyframe_timestamps[keyframe_timestamps.len() - 1] - keyframe_timestamps[0]) / 
                      (keyframe_timestamps.len() - 1) as f64;
        println!("Average keyframe interval: {:.2}s", interval);
        interval
    } else {
        // Default interval is 2 seconds
        println!("Could not calculate keyframe interval, using default (2s)");
        2.0
    };
    
    Ok(VideoInfo {
        duration,
        framerate: fps,
        start_time,
        keyframe_timestamps,
        avg_keyframe_interval,
    })
}

fn extract_audio_waveform(input_file: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Create a temporary WAV file
    let temp_wav = "temp_audio.wav";

    // Extract audio to WAV using FFmpeg
    let status = Command::new("ffmpeg")
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

fn analyze_audio(
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

fn process_segments(
    input_file: &str,
    segments: &[AudioSegment],
    songs: &[Song],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing {} segments...", segments.len());

    let mut song_counter = 0;
    let mut gap_counter = 0;

    for segment in segments.iter() {
        if !segment.is_song {
            // Optionally process gaps
            gap_counter += 1;
            // let output_file = format!("gap_{:02}.mp4", gap_counter);

            println!(
                "ignoring gap {}: {:.2}s to {:.2}s",
                gap_counter, segment.start_time, segment.end_time
            );

            // extract_segment(input_file, &output_file, segment.start_time, segment.end_time)?;
            continue;
        }

        // Process song
        song_counter += 1;

        // Check if we have a song title for this segment
        let song_title = if song_counter <= songs.len() {
            &songs[song_counter - 1].title
        } else {
            // Fallback if we have more segments than songs
            println!("Warning: More song segments detected than provided in setlist. Using default naming.");
            &format!("song_{}", song_counter)
        };

        // Create a safe filename from the song title
        let safe_title = sanitize_filename(song_title);
        let output_file = format!("{}.mp4", safe_title);

        println!(
            "Extracting song {}: \"{}\" - {:.2}s to {:.2}s (duration: {:.2}s)",
            song_counter,
            song_title,
            segment.start_time,
            segment.end_time,
            segment.end_time - segment.start_time
        );

        extract_segment(
            input_file,
            &output_file,
            segment.start_time,
            segment.end_time,
        )?;
    }

    println!(
        "Successfully extracted {} songs and {} gaps",
        song_counter, gap_counter
    );
    Ok(())
}

fn detect_song_boundaries_from_text(
    input_file: &str,
    artist: &str,
    songs: &[Song],
    video_info: &VideoInfo,
) -> Result<Vec<AudioSegment>, Box<dyn std::error::Error>> {
    let total_duration = video_info.duration;
    let artist_cmp = artist.to_lowercase();
    // Create a temporary directory for frames
    let temp_dir = "temp_frames";
    if Path::new(temp_dir).exists() {
        fs::remove_dir_all(temp_dir)?;
    }
    fs::create_dir(temp_dir)?;

    let mut sorted_songs: Vec<Song> = songs.to_vec().iter().map(|song| Song { title: song.title.to_lowercase() }).collect();
    println!("songs: {:?}", songs);
    // sorted_songs.clone_from_slice(songs);
    sorted_songs.sort_by(|a, b| a.title.len().partial_cmp(&b.title.len()).unwrap().reverse());

    println!("Extracting keyframes for song title detection...");
    
    // Extract keyframes with potential text overlays - using keyframes for better timestamp accuracy
    let status = Command::new("ffmpeg")
        .args(&[
            "-skip_frame", "nokey",     // Only process keyframes
            "-i", input_file,
            "-c:v", "png",
            "-vsync", "0",              // Use original timestamps
            "-qscale:v", "31",          // Quality setting
            "-vf", "scale=400:200,crop=iw/1.5:ih/5:0:160",  // Focus on the text area
            &format!("{}/%d.png", temp_dir),  // Use sequential numbering
        ])
        .status()?;
        
    println!("Keyframes extracted successfully.");

    if !status.success() {
        return Err("Failed to extract frames".into());
    }

    // Get list of extracted frames
    let mut frames = fs::read_dir(temp_dir)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().map_or(false, |ext| ext == "png")
        })
        .collect::<Vec<_>>();

    println!("Extracted {} frames, analyzing for text...", frames.len());

    // Map to store detected song start times
    let mut song_start_times = Vec::new();

    // Process each frame to detect text
    frames.sort_by(|a, b| a.path().cmp(&b.path()));
    for frame_entry in frames {
        let frame_path = frame_entry.path();
        let frame_name = frame_path.file_name().unwrap().to_string_lossy();
        
        // Extract frame number to calculate timestamp
        let frame_num = frame_name
            .strip_suffix(".png")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        
        // Create a temporary output file for tesseract
        let out_txt = format!("{}/{}", temp_dir, frame_num);
        
        // Run tesseract OCR on the frame
        let status = Command::new("tesseract")
            .args(&[
                frame_path.to_str().unwrap(),
                &out_txt,
                "--psm",
                "11",
            ])
            .stderr(std::process::Stdio::null()) // Silence stderr output
            .status()?;
            
        if !status.success() {
            println!("Warning: Tesseract failed on frame {}", frame_name);
            continue;
        }
        
        // Read the OCR result
        let out_txt_path = format!("{}.txt", out_txt);
        if let Ok(text) = fs::read_to_string(&out_txt_path) {
            // Use common parsing function
            let parsed = match parse_tesseract_output(&text, &artist_cmp) {
                Some(result) => result,
                None => continue
            };
            
            let (lines, overlay) = parsed;
            
            // Format text for display
            let filtered_text = if overlay {
                lines[1..].to_vec().join("\n")
            } else {
                lines.to_vec().join("\n")
            };
            
            if overlay {
                println!("Frame {}: Detected overlay: '{}'", frame_name, filtered_text);
            } else {
                // println!("Frame {}: Detected text: '{}'", frame_name, detected_text);
            }
            
            // Check if the detected text matches any song title
            for song in &sorted_songs {
                let song_title = &song.title;
                
                // Use the common matching logic
                if matches_song_title(&lines, song_title, overlay) {
                    // Get accurate timestamp for this frame using the video info
                    let timestamp = get_frame_timestamp(video_info, frame_num);
                    
                    println!("Match found! '{}' matches song '{}' at {}s (frame {})", 
                            filtered_text, song.title, timestamp, frame_num);
                    
                    // Extract additional frames around this timestamp for more accurate boundary detection
                    let refined_timestamp = refine_song_start_time(input_file, temp_dir, &artist_cmp, song_title, timestamp)?;
                    
                    // Use the refined timestamp if available, otherwise use the original
                    let final_timestamp = if refined_timestamp > 0.0 && refined_timestamp < timestamp {
                        println!("Refined start time for '{}': {}s (was {}s)", 
                                song_title, refined_timestamp, timestamp);
                        refined_timestamp
                    } else {
                        timestamp
                    };
                    
                    song_start_times.push((song.title.clone(), final_timestamp));
                    break;
                }
            }
        }
    }

    // Sort song start times by timestamp
    song_start_times.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Create segments from detected song boundaries
    let mut segments = Vec::new();
    
    if song_start_times.is_empty() {
        println!("No song titles detected in frames. Will fall back to audio analysis.");
        return Ok(Vec::new());
    }
    
    println!("Detected {} song boundaries from text overlays", song_start_times.len());
    
    // Create segments from the detected song start times
    for i in 0..song_start_times.len() {
        let start_time = song_start_times[i].1;
        let end_time = if i < song_start_times.len() - 1 {
            song_start_times[i + 1].1
        } else {
            total_duration
        };
        
        segments.push(AudioSegment {
            start_time,
            end_time,
            is_song: true,
        });
    }
    
    // If we have a gap at the beginning, add it as a non-song segment
    if !segments.is_empty() && segments[0].start_time > 0.0 {
        segments.insert(0, AudioSegment {
            start_time: 0.0,
            end_time: segments[0].start_time,
            is_song: false,
        });
    }
    
    // Clean up temporary files
    // fs::remove_dir_all(temp_dir)?;
    
    Ok(segments)
}

fn parse_tesseract_output(text: &str, artist: &str) -> Option<(Vec<String>, bool)> {
    let detected_text = text.trim().to_lowercase();
    
    // Skip if empty or too short
    if detected_text.len() < 4 {
        return None;
    }

    // Filter out empty lines
    let lines: Vec<String> = detected_text.lines()
        .filter(|line| line.trim().len() > 0)
        .map(|line| line.to_string())
        .collect();
        
    if lines.is_empty() {
        return None;
    }
    
    // Check if this is an overlay with artist at the top
    let is_overlay = !artist.is_empty() && lines[0].trim() == artist.to_lowercase();
    
    Some((lines, is_overlay))
}

fn matches_song_title(lines: &[String], song_title: &str, is_overlay: bool) -> bool {
    for line in lines {
        let line_lower = line.trim().to_lowercase();
        let title_lower = song_title.to_lowercase();
        
        // Check for partial match (if line contains song title or vice versa with overlay)
        if line_lower.contains(&title_lower) || (is_overlay && title_lower.contains(&line_lower)) {
            return true;
        }
    }
    false
}

fn refine_song_start_time(
    input_file: &str,
    temp_dir: &str,
    artist: &str,
    song_title: &str,
    initial_timestamp: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    println!("Refining start time for '{}' (initially at {}s)...", song_title, initial_timestamp);
    
    // Define the time window to look before the detected timestamp
    let look_back_seconds = 3.0;
    let start_time = if initial_timestamp > look_back_seconds {
        initial_timestamp - look_back_seconds
    } else {
        0.0
    };
    
    // Create a subdirectory for the refined frames
    let refined_dir = format!("{}/refined_{}", temp_dir, song_title.replace(" ", "_"));
    if !Path::new(&refined_dir).exists() {
        fs::create_dir(&refined_dir)?;
    }
    
    // Extract more frequent frames within the time window (4 frames per second)
    println!("Extracting additional frames from {}s to {}s at 4 fps...", start_time, initial_timestamp);
    let status = Command::new("ffmpeg")
        .args(&[
            "-ss", &format!("{}", start_time),
            "-to", &format!("{}", initial_timestamp),
            "-i", input_file,
            "-c:v", "png",
            "-vf", "fps=4,scale=400:200,crop=iw/1.5:ih/5:0:160", // Exactly 4 frames per second
            "-qscale:v", "31",          // Quality setting
            &format!("{}/%d.png", refined_dir), // Sequential numbering starting from 1
        ])
        .status()?;
    
    if !status.success() {
        println!("Failed to extract refined frames");
        return Ok(0.0);
    }
    
    // Read the refined frames and analyze them
    let mut frames = fs::read_dir(&refined_dir)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().map_or(false, |ext| ext == "png")
        })
        .collect::<Vec<_>>();
    
    frames.sort_by(|a, b| a.path().cmp(&b.path()));
    println!("Analyzing {} refined frames for song title '{}'", frames.len(), song_title);
    
    let mut earliest_match = 0.0;
    let frame_count = frames.len() as f64; // Total number of frames for interpolation
    
    // Process each refined frame
    for frame_entry in frames {
        let frame_path = frame_entry.path();
        let frame_name = frame_path.file_name().unwrap().to_string_lossy();
        
        // Extract frame number
        let frame_num = frame_name
            .strip_suffix(".png")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        
        // Create a temporary output file for tesseract
        let out_txt = format!("{}/{}", refined_dir, frame_num);
        
        // Run tesseract OCR on the frame
        let status = Command::new("tesseract")
            .args(&[
                frame_path.to_str().unwrap(),
                &out_txt,
                "--psm",
                "11",
            ])
            .stderr(std::process::Stdio::null())
            .status()?;
            
        if !status.success() {
            continue;
        }
        
        // Read the OCR result
        let out_txt_path = format!("{}.txt", out_txt);
        if let Ok(text) = fs::read_to_string(&out_txt_path) {
            let parsed = match parse_tesseract_output(&text, artist) {
                Some(result) => result,
                None => continue
            };
            
            let (lines, overlay) = parsed;
            
            // If we see the artist overlay that's good enough.
            // On the initial fade in we might be able to see the artist name but not the song title.
            if overlay || matches_song_title(&lines, song_title, overlay) {
                // Calculate exact timestamp for this frame
                // Since we're using fps=4, each frame is exactly 0.25 seconds
                // Frame numbers start at 1, so frame 1 = start_time, frame 2 = start_time + 0.25, etc.
                let fps = 4.0; // Matches the fps value in the ffmpeg command
                let frame_time = start_time + ((frame_num - 1) as f64 / fps);
                
                println!("Earlier match for '{}' at {}s (frame {}/{}, +{}s from start)", 
                        song_title, frame_time, frame_num, frame_count, 
                        (frame_num - 1) as f64 / fps);
                
                if earliest_match == 0.0 || frame_time < earliest_match {
                    earliest_match = frame_time;
                }
            }
        }
    }
    
    // Return the earliest match if found, otherwise 0.0
    if earliest_match > 0.0 {
        println!("Successfully refined start time for '{}' from {}s to {}s (-{:.2}s)", 
                song_title, initial_timestamp, earliest_match, initial_timestamp - earliest_match);
    } else {
        println!("Could not find earlier boundary for '{}', keeping original timestamp: {}s", 
                song_title, initial_timestamp);
    }
    Ok(earliest_match)
}

fn get_frame_timestamp(video_info: &VideoInfo, frame_num: usize) -> f64 {
    // If we have enough keyframes, map the frame number to the correct keyframe
    if !video_info.keyframe_timestamps.is_empty() {
        // Frame numbers are 1-indexed in our extraction, but array is 0-indexed
        let index = if frame_num > 0 { frame_num - 1 } else { 0 };
        
        if index < video_info.keyframe_timestamps.len() {
            println!("Using direct keyframe timestamp: {}s for frame {}", 
                    video_info.keyframe_timestamps[index], frame_num);
            return video_info.keyframe_timestamps[index];
        } else {
            println!("Frame number {} exceeds available keyframes {}, using estimation", 
                    frame_num, video_info.keyframe_timestamps.len());
        }
    }
    
    // Fallback - estimate timestamp based on keyframe interval
    let estimated_timestamp = video_info.start_time + (frame_num as f64 * video_info.avg_keyframe_interval);
    println!("Estimated timestamp for frame {}: {:.2}s", frame_num, estimated_timestamp);
    
    estimated_timestamp
}

fn sanitize_filename(input: &str) -> String {
    // Replace characters that are problematic in filenames
    let mut sanitized = input.replace(
        &['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0'][..],
        "_",
    );

    // Trim leading/trailing whitespace and dots
    sanitized = sanitized.trim().trim_matches('.').to_string();

    // If the name is empty after sanitization, provide a default
    if sanitized.is_empty() {
        sanitized = "untitled".to_string();
    }

    sanitized
}

fn extract_segment(
    input_file: &str,
    output_file: &str,
    start_time: f64,
    end_time: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // let duration = end_time - start_time;

    // Use MP4Box for segment extraction
    let status = Command::new("MP4Box")
        .args(&[
            "-splitx",
            &format!("{:.3}:{:.3}", start_time, end_time),
            "-out",
            output_file,
            input_file,
        ])
        .status()?;

    if !status.success() {
        return Err(format!("Failed to extract segment to {}", output_file).into());
    }

    Ok(())
}

// This is really slow!
// Perhaps because it re-encodes the audio?
fn _extract_segment_ffmpeg(
    input_file: &str,
    output_file: &str,
    start_time: f64,
    end_time: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args(&[
            "-i",
            input_file,
            "-ss",
            &format!("{:.3}", start_time),
            "-to",
            &format!("{:.3}", end_time),
            "-c:a",
            "aac", // AAC audio codec
            "-b:a",
            "192k", // Bitrate
            "-vn",  // No video
            "-y",   // Overwrite output file
            output_file,
        ])
        .status()?;

    if !status.success() {
        return Err(format!("Failed to extract segment to {}", output_file).into());
    }

    Ok(())
}

// MP4Box -udta TRACKID:type=name:str="Name of My Track" file.mp4
// MP4Box -udta 3:type=name -udta 3:type=name:str="Director Commentary" file.mp4
// -time [tkID=]DAY/MONTH/YEAR-H:M:S: set movie or track creation time
// -mtime tkID=DAY/MONTH/YEAR-H:M:S: set media creation time
// tags: -tags name=value:tag2=value https://wiki.gpac.io/MP4Box/mp4box-other-opts/#tagging-support

// Look for song titles overlayed on the video
// 1) extracting frames
// qscale might only work with jpg
// ffmpeg -skip_frame nokey -i Faye\ Webster：\ Tiny\ Desk\ Concert.mp4 -c:v png -vsync 0 -qscale:v 31 'vout/faye_frame%05d.png'
// 2) tesseract
//  usually shows up right at or after the start of song
// however, sometimes it shows up while they are doing a band introduction instead of the real song.

/*
ffmpeg -skip_frame nokey -i Faye\ Webster：\ Tiny\ Desk\ Concert.mp4 -c:v png -vsync 0 -qscale:v 31 -vf "scale=400:200,crop=iw/3:ih/5:0:160" 'vout/faye_frame%04d.png'
tesseract faye_frame0006.png out
cat out.txt
*/
