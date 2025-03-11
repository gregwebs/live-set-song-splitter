mod ocr;
use crate::ocr::{OcrParse, run_tesseract_ocr_parse, run_tesseract_ocr, parse_tesseract_output};
use stringmetrics::{levenshtein_weight, LevWeights};


use serde::{Deserialize, Serialize};
use std::collections::{HashSet};
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
    album: String,
    date: String,
    #[serde(rename = "setList")]
    set_list: Vec<Song>,
}

#[derive(Debug, Clone, Copy)]
struct FrameInfo {
    timestamp: f64,
    is_keyframe: bool,
}

#[derive(Debug, Clone)]
struct VideoInfo {
    // Basic information
    duration: f64,
    framerate: u32, // Integer frames per second

    // Frame information
    frames: Vec<FrameInfo>,
    keyframe_indices: Vec<usize>,
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
    let mut segments = detect_song_boundaries_from_text(
        input_file,
        &setlist.artist,
        &setlist.set_list,
        &video_info,
    )?;
    for segment in &segments {
        println!("Segment: {:?}", segment);
    }

    // TODO: cli option to choose between overlay and audio analysis
    let fallback_audio_analysis = false;

    // If text detection didn't find enough songs, fall back to audio analysis
    if segments.iter().filter(|s| s.is_song).count() < num_songs {
        let msg = "Text overlay detection didn't find all songs.";
        if !fallback_audio_analysis {
            return Err(msg.into());
        }
        println!("{} Falling back to audio analysis...", msg);

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

    // Extract year from date field
    let year = if !setlist.date.is_empty() {
        setlist.date.split('-').next().unwrap_or("").to_string()
    } else {
        String::new()
    };

    // Process each detected segment
    process_segments(
        input_file,
        &segments,
        &setlist.set_list,
        &setlist.artist,
        &setlist.album,
        &year,
    )?;

    println!("Audio splitting complete!");
    Ok(())
}

fn get_video_info(input_file: &str) -> Result<VideoInfo, Box<dyn std::error::Error>> {
    println!("Analyzing video file metadata...");

    // Get basic video information in one call
    let basic_info_output = create_ffprobe_command()
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate:format=duration,start_time",
            "-of",
            "json",
            input_file,
        ])
        .output()?;

    if !basic_info_output.status.success() {
        return Err("Failed to get video information".into());
    }

    let info_json = String::from_utf8(basic_info_output.stdout)?;
    let info: serde_json::Value = serde_json::from_str(&info_json)?;

    // Extract duration
    let duration = info["format"]["duration"]
        .as_str()
        .ok_or("Missing duration")?
        .parse::<f64>()?;

    // Extract start time
    let start_time = info["format"]["start_time"]
        .as_str()
        .unwrap_or("0")
        .parse::<f64>()
        .unwrap_or(0.0);
    if start_time != 0.0 {
        panic!("start time is not 0, this may cause issues with audio splitting.");
    }

    // Extract framerate
    let fps_str = info["streams"][0]["r_frame_rate"]
        .as_str()
        .ok_or("Missing framerate")?;
    let mut fps: u32 = 24; // Default fallback value
    if let Some((num, den)) = fps_str.split_once('/') {
        if let (Ok(n), Ok(d)) = (num.parse::<f64>(), den.parse::<f64>()) {
            if d > 0.0 {
                // Calculate framerate and round to nearest integer
                fps = (n / d).round() as u32;
            }
        }
    }

    println!(
        "Video duration: {}s, start time: {}s, framerate: {} fps",
        duration, start_time, fps
    );

    // Get all frame information in a single pass
    println!("Extracting all frame information...");
    let frame_data = create_ffprobe_command()
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=print_section=0",
            input_file,
        ])
        .output()?;

    let frame_data_str = String::from_utf8(frame_data.stdout)?;

    // Parse frame data - format is "pts_time,flags"
    let mut frames = Vec::new();
    let mut keyframe_indices = Vec::new();

    for (i, line) in frame_data_str.lines().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(timestamp) = parts[0].parse::<f64>() {
                let is_keyframe = parts[1].contains('K');

                // Add to frames collection
                frames.push(FrameInfo {
                    timestamp,
                    is_keyframe,
                });

                // If it's a keyframe, record its index
                if is_keyframe {
                    keyframe_indices.push(i);
                }
            }
        }
    }

    println!(
        "Found {} frames, including {} keyframes",
        frames.len(),
        keyframe_indices.len()
    );

    Ok(VideoInfo {
        duration,
        framerate: fps,
        frames,
        keyframe_indices,
    })
}

fn create_ffmpeg_command() -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(&["-hide_banner", "-loglevel", "warning"]);
    cmd.stdout(std::process::Stdio::null());
    cmd
}

fn create_ffprobe_command() -> Command {
    let mut cmd = Command::new("ffprobe");
    cmd.args(&["-hide_banner", "-loglevel", "warning"]);
    cmd
}

fn extract_audio_waveform(input_file: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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
    artist: &str,
    album: &str,
    year: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing {} segments...", segments.len());
    if segments.len() > songs.len() {
        return Err(format!(
            "Too many segments detected. {} segments but only {} songs provided.",
            segments.len(),
            songs.len()
        )
        .into());
    }

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

            // extract_segment(input_file, &output_file, segment.start_time, segment.end_time, None, None, None)?;
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
            Some(song_title),
            Some(artist),
            Some(album),
            Some(year),
            Some(song_counter), // Add song number as track metadata
        )?;
    }

    println!(
        "Successfully extracted {} songs and {} gaps",
        song_counter, gap_counter
    );
    Ok(())
}

fn frame_number_from_image_filename(frame_path: &std::path::PathBuf) -> usize {
    let frame_name = frame_path.file_name().unwrap().to_string_lossy();
    return frame_name
        .strip_suffix(".png")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
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

    let mut sorted_songs: Vec<Song> = songs
        .to_vec()
        .iter()
        .map(|song| Song {
            title: song.title.to_lowercase(),
        })
        .collect();
    // sorted_songs.clone_from_slice(songs);
    sorted_songs.sort_by(|a, b| a.title.len().partial_cmp(&b.title.len()).unwrap().reverse());

    println!("Extracting keyframes for song title detection...");

    // Extract keyframes with potential text overlays - using keyframes for better timestamp accuracy
    let status = create_ffmpeg_command()
        .args(&[
            "-skip_frame",
            "nokey", // Only process keyframes
            "-i",
            input_file,
            "-c:v",
            "png",
            "-vsync",
            "0", // Use original timestamps
            "-qscale:v",
            "31", // Quality setting
            "-vf",
            "scale=400:200,crop=iw/1.5:ih/5:0:160", // Focus on the text area
            &format!("{}/%d.png", temp_dir),        // Use sequential numbering
        ])
        .status()?;

    println!("Keyframes extracted successfully.");

    if !status.success() {
        return Err("Failed to extract frames".into());
    }

    // Get list of extracted frames
    let mut frames = fs::read_dir(temp_dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "png"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();

    println!("Extracted {} frames, analyzing for text...", frames.len());

    // Map to store detected song start times
    let mut song_start_times = Vec::new();
    let mut song_title_matched: HashSet<String> = HashSet::new();

    // Process each frame to detect text
    frames.sort_by(|a, b|
        frame_number_from_image_filename(a).cmp(&frame_number_from_image_filename(b))
    );
    for frame_path in frames {
        // Extract frame number to calculate timestamp
        let frame_num = frame_number_from_image_filename(&frame_path);

        let song_titles_to_match = &sorted_songs.iter().filter(|song|
            // skip already matched songs
            !song_title_matched.contains(&song.title)
        ).map(|song| &song.title).cloned().collect::<Vec<_>>();

        // Run tesseract OCR on the frame
        let parsed = run_tesseract_ocr_parse(frame_path.to_str().unwrap(), &artist_cmp, Some("11"))?;
        match parsed {
            Some(lo@(_, overlay)) => {
                let title_time= match_song_titles(input_file, temp_dir, &lo, song_titles_to_match, &artist_cmp, frame_num, video_info)?;
                if title_time.is_some() {
                    song_title_matched.insert(title_time.as_ref().unwrap().0.clone());
                    song_start_times.push(title_time.unwrap())
                } else if overlay {
                    // found some text but not a proper match
                    // try running tesseract with a different setting
                    let parsed2 = run_tesseract_ocr_parse(frame_path.to_str().unwrap(), &artist_cmp, None)?;
                    match parsed2 {
                        Some(lo) => {
                            let title_time= match_song_titles(input_file, temp_dir, &lo, song_titles_to_match, &artist_cmp, frame_num, video_info)?;
                            if title_time.is_some() {
                                song_title_matched.insert(title_time.as_ref().unwrap().0.clone());
                                song_start_times.push(title_time.unwrap())
                            }
                        }
                        None => { }
                    }
                }
            }
            None => { }
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

    println!(
        "Detected {} song boundaries from text overlays",
        song_start_times.len()
    );

    // Create segments from the detected song start times
    for i in 0..song_start_times.len() {
        let start_time = if i == 0 {
            // Always set the first song to start at 0 seconds
            0.0
        } else {
            song_start_times[i].1
        };

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

    // Note: No need to add a gap at the beginning since first song starts at 0.0

    // Clean up temporary files
    // fs::remove_dir_all(temp_dir)?;

    Ok(segments)
}

/// Normalize text by removing punctuation and spaces, keeping only alphanumeric characters.
/// Also converts all characters to lowercase for case-insensitive comparison.
fn normalize_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

fn matches_song_title(lines: &[String], song_title: &str, is_overlay: bool) -> Option<(String, u32)> {
    let title_normalized = normalize_text(song_title);
    let weights = LevWeights::new(2, 2, 1);
    let levenshtein_limit = 3;

    for line in lines {
        let line_normalized = normalize_text(line);
        
        // Check for exact or partial match
        if line_normalized.contains(&title_normalized) {
            return Some((line.clone(), 0))
        }
        if !is_overlay {
            continue
        }
        // If we have an overlay and no exact match was found, try fuzzy matching
        let lev = levenshtein_weight(&line_normalized, &title_normalized, levenshtein_limit + 10, &weights);
        // println!("levenshtein distance: {}. {}", lev, line);
        if lev <= levenshtein_limit {
            return Some((line.clone(), lev))
        }
        if title_normalized.starts_with(&line_normalized) {
            if (line_normalized.len() as f64 / title_normalized.len() as f64) >= 0.4 {
                return Some((line.clone(), (title_normalized.len() - line_normalized.len()) as u32))
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("Hello, World!"), "helloworld");
        assert_eq!(normalize_text("Test123"), "test123");
        assert_eq!(normalize_text("  Spaces   "), "spaces");
        assert_eq!(normalize_text("UPPERCASE"), "uppercase");
        assert_eq!(normalize_text("special-@#$-chars"), "specialchars");
    }

    #[test]
    fn test_matches_song_title() {
        // Test exact matches
        let lines = vec!["hello world".to_string(), "test song".to_string()];
        assert!(matches_song_title(&lines, "test song", false).is_some());
        
        // Test partial matches
        assert!(matches_song_title(&lines, "test", false).is_some());
        assert!(matches_song_title(&lines, "song", false).is_some());
        
        // Test case insensitivity
        assert!(matches_song_title(&lines, "TEST SONG", false).is_some());
        
        // Test with overlay
        assert!(matches_song_title(&lines, "hello world test", true).is_some());
        
        // Test fuzzy matching (only works with overlay flag)
        let ocr_lines = vec!["helo wrld".to_string()];  // OCR might miss letters
        assert!(!matches_song_title(&ocr_lines, "hello world", false).is_some());  // Should fail without overlay
        assert!(matches_song_title(&ocr_lines, "hello world", true).is_some());    // Should pass with overlay
        
        // Test non-matches
        let other_lines = vec!["completely different".to_string()];
        assert!(!matches_song_title(&other_lines, "test song", true).is_some());
    }
}

fn match_song_titles(
    input_file: &str,
    temp_dir: &str,
    ocr_parse: &OcrParse,
    song_titles_to_match: &Vec<String>,
    artist_cmp: &str,
    frame_num: usize,
    video_info: &VideoInfo,
) -> Result<Option<(String, f64)>, Box<dyn std::error::Error>> {
    let (lines, overlay) = ocr_parse;

    // Format text for display
    let filtered_text = if *overlay {
        lines[1..].to_vec().join("\n")
    } else {
        lines.to_vec().join("\n")
    };

    if *overlay {
        println!(
            "Frame {}: Detected overlay: '{}'",
            frame_num, filtered_text
        );
    } else {
        /*
        println!("Frame {}: Detected text: '{}'", frame_num, filtered_text);
        */
    }

    let mut best_match: Option<(String, (String, u32))> = None;
    for song_title in song_titles_to_match {
        match matches_song_title(&lines, song_title, *overlay) {
            None => {
                continue;
            }
            Some(matched@(_, lev_dist)) => {
                match best_match {
                    None => {
                        best_match = Some((song_title.to_string(), matched));
                    }
                    Some((_, (_, best_dist))) => {
                        if lev_dist < best_dist {
                            best_match = Some((song_title.to_string(), matched));
                        }
                    }
                }
            }
        }
    }
    match best_match {
        None => {
            Ok(None)
        }
        Some((song_title, (line, lev_dist))) => {
            println!(
                "Match found! '{}' matches song '{}' frame={} dist={}",
                line, &song_title, frame_num, lev_dist,
            );
            match timestamp_for_song(input_file, temp_dir, &artist_cmp, &song_title, frame_num, video_info) {
                Ok(timestamp) => {
                    return Ok(Some((song_title.to_string(), timestamp)));
                }
                Err(e) => Err(e)
            }
        }
    }
}

fn timestamp_for_song(
    input_file: &str,
    temp_dir: &str,
    artist_cmp: &str,
    song_title: &str,
    frame_num: usize,
    video_info: &VideoInfo,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Get accurate timestamp for this frame using the video info
    let frame_info = get_keyframe(video_info, frame_num);

    // Extract additional frames around this timestamp for more accurate boundary detection
    let refined_timestamp = refine_song_start_time(
        input_file,
        temp_dir,
        &artist_cmp,
        song_title,
        frame_num,
        video_info,
    )?;

    // Use the refined timestamp if available, otherwise use the original
    let final_timestamp =
        if refined_timestamp > 0.0 && refined_timestamp < frame_info.timestamp {
            refined_timestamp
        } else {
            frame_info.timestamp
        };
    return Ok(final_timestamp)
}

fn refine_song_start_time(
    input_file: &str,
    temp_dir: &str,
    artist: &str,
    song_title: &str,
    keyframe_num: usize,
    video_info: &VideoInfo,
) -> Result<f64, Box<dyn std::error::Error>> {
    let absolute_framenum = get_keyframe_absolute_framenum(video_info, keyframe_num);
    let keyframe_info = video_info.frames[absolute_framenum];
    let initial_timestamp = keyframe_info.timestamp;
    println!(
        "Refining start time for '{}' (initially at {}s)...",
        song_title, initial_timestamp
    );

    // Define the time window to look before the detected timestamp
    let look_back_seconds = 3;
    let start_time = if initial_timestamp > (look_back_seconds as f64) {
        initial_timestamp - (look_back_seconds as f64)
    } else {
        if initial_timestamp != 0.0 {
            panic!("Initial timestamp is less than look back seconds and not zero!")
        }
        0.0
    };

    // Create a subdirectory for the refined frames
    let refined_dir = format!("{}/refined_{}", temp_dir, song_title.replace(" ", "_"));
    if !Path::new(&refined_dir).exists() {
        fs::create_dir(&refined_dir)?;
    }

    // Extract frames at original video framerate for accuracy
    let fps = video_info.framerate;
    let status = create_ffmpeg_command()
        .args(&[
            "-ss",
            &format!("{}", start_time),
            "-to",
            &format!("{}", initial_timestamp),
            "-i",
            input_file,
            "-c:v",
            "png",
            "-vf",
            &format!("fps={},scale=400:200,crop=iw/1.5:ih/5:0:160", fps), // Use original video framerate
            "-qscale:v",
            "31",                               // Quality setting
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
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "png"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();

    frames.sort_by(|a, b|
        frame_number_from_image_filename(a).cmp(&frame_number_from_image_filename(b))
    );
    println!(
        "Analyzing {} refined frames for song title '{}' from {}s to {}s at {} fps",
        frames.len(),
        song_title,
        start_time, initial_timestamp, fps
    );

    let mut earliest_match: Option<usize> = None;

    let frame_count = frames.len();
    // Process each refined frame
    for frame_path in frames {
        // Extract frame number
        let frame_num = frame_number_from_image_filename(&frame_path);

        // Run tesseract OCR on the frame
        let text = run_tesseract_ocr(frame_path.to_str().unwrap(), Some("11"))?;
        let parsed = match parse_tesseract_output(&text, artist) {
            Some(result) => result,
            None => continue,
        };

        let (lines, overlay) = parsed;

        // If we see the artist overlay that's good enough.
        // On the initial fade in we might be able to see the artist name but not the song title.
        if overlay || matches_song_title(&lines, song_title, overlay).is_some() {
            if earliest_match.is_none() || frame_num < earliest_match.unwrap() {
                earliest_match = Some(frame_num);
            }
        }
    }

    // Return the earliest match if found, otherwise 0.0
    if let Some(earliest_match) = earliest_match {
        let subtracted_frame_num =
            frame_count as usize - earliest_match;
        let mut earliest_frame_num = absolute_framenum - subtracted_frame_num as usize;
        // TODO: detect the fade in itself instead of text
        // TODO: this should be a configureable fade in value
        // with a fade in we are always going to be late to detect the text
        // here we go back just one additional frame, but should probably go back more
        if earliest_frame_num > 1 {
            earliest_frame_num -= 1;
        }
        let frame = video_info.frames[earliest_frame_num];
        let new_time = frame.timestamp;
        println!(
            "Successfully refined start time for '{}' from {}s to {}s (-{:.2}s)",
            song_title,
            initial_timestamp,
            new_time,
            initial_timestamp - new_time
        );
        Ok(new_time)
    } else {
        println!(
            "Could not find earlier boundary for '{}', keeping original timestamp: {}s",
            song_title, initial_timestamp
        );
        return Ok(0.0);
    }
}


fn get_keyframe_absolute_framenum(video_info: &VideoInfo, frame_num: usize) -> usize {
    // If we have keyframe indices, map the frame number to the correct keyframe
    if video_info.keyframe_indices.is_empty() || video_info.frames.is_empty() {
        panic!("Keyframe indices and frames must be populated");
    }
    // Frame numbers are 1-indexed in our extraction, but array is 0-indexed
    let index = if frame_num > 0 { frame_num - 1 } else { 0 };

    // Check if this index happens to be in our keyframes list
    if index > video_info.keyframe_indices.len() {
        panic!(
            "Frame index {} is out of bounds for keyframe indices",
            index
        )
    }
    return video_info.keyframe_indices[index];
}

fn get_keyframe(video_info: &VideoInfo, frame_num: usize) -> FrameInfo {
    let index = get_keyframe_absolute_framenum(video_info, frame_num);
    // If it's a direct frame match, return that timestamp
    let frame = video_info.frames[index];
    let is_keyframe = video_info.frames[index].is_keyframe;
    /*
    println!(
        "Using direct frame timestamp: {}s for frame {} (keyframe: {})",
        frame.timestamp, frame_num, is_keyframe
    );
     */
    if !is_keyframe {
        panic!("Frame {} is not a keyframe", frame_num);
    }
    return frame;

    /*
    // Fallback - estimate timestamp based on frame interval
    let estimated_timestamp = video_info.start_time + (frame_num as f64 / video_info.framerate);

    // Find the closest actual frame timestamp
    if !video_info.frames.is_empty() {
        let exact_time = find_nearest_frame_timestamp(estimated_timestamp, &video_info.frames);
        panic!("Using closest frame timestamp for frame {}: {}s (estimate: {:.2}s)",
                frame_num, exact_time, estimated_timestamp);
        // return exact_time;
    }

    println!("Estimated timestamp for frame {}: {:.2}s", frame_num, estimated_timestamp);
    estimated_timestamp
    */
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

fn _extract_segment_mp4box(
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
fn extract_segment(
    input_file: &str,
    output_file: &str,
    start_time: f64,
    end_time: f64,
    song_title: Option<&str>,
    artist_name: Option<&str>,
    album_name: Option<&str>,
    year: Option<&str>,
    track_number: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = create_ffmpeg_command();

    cmd.args(&[
        "-i",
        input_file,
        "-ss",
        &format!("{:.3}", start_time),
        "-to",
        &format!("{:.3}", end_time),
    ]);

    // Add metadata if available
    if let Some(title) = song_title {
        cmd.args(&["-metadata", &format!("title={}", title)]);
    }

    if let Some(artist) = artist_name {
        cmd.args(&["-metadata", &format!("artist={}", artist)]);
    }

    if let Some(album) = album_name {
        cmd.args(&["-metadata", &format!("album={}", album)]);
    }

    if let Some(year_value) = year {
        if !year_value.is_empty() {
            cmd.args(&["-metadata", &format!("date={}", year_value)]);
        }
    }

    // Add track number metadata
    if let Some(track) = track_number {
        cmd.args(&["-metadata", &format!("track={}", track)]);
    }

    cmd.args(&[
        "-y", // Overwrite output file
        output_file,
    ]);

    let status = cmd.status()?;

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
