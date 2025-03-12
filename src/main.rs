mod ocr;
use crate::ocr::{
    matches_song_title, matches_song_title_weighted, weights_for_greedy_extractor,
    weights_for_stingy_extractor,
};
mod audio;
mod ffmpeg;
mod io;
use crate::io::{overwrite_dir, sanitize_filename};
mod video;
use crate::video::VideoInfo;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::BufReader;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Song {
    title: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SetMetaData {
    artist: String,
    album: Option<String>,
    date: Option<String>,
    show: Option<String>,
}

impl SetMetaData {
    fn year(&self) -> Option<String> {
        self.date
            .as_ref()
            .and_then(|date| date.split('-').next().map(|s| s.to_string()))
    }

    fn folder_name(&self) -> String {
        self.album.as_ref().unwrap_or(&self.artist).to_string()
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SetList {
    #[serde(flatten)]
    metadata: SetMetaData,
    #[serde(rename = "setList")]
    set_list: Vec<Song>,
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
    println!("Artist: {}", setlist.metadata.artist);
    println!("Expected number of songs: {}", num_songs);
    println!("Songs:");
    for (i, song) in setlist.set_list.iter().enumerate() {
        println!("  {}. {}", i + 1, song.title);
    }

    // Get all video information at once
    let video_info = VideoInfo::from_ffprobe_file(input_file)?;
    println!("Total duration: {:.2} seconds", video_info.duration);

    // First try to detect song boundaries using text overlays
    println!("Attempting to detect song boundaries using text overlays...");
    let mut segments = detect_song_boundaries_from_text(
        input_file,
        &setlist.metadata.artist,
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
        let audio_data = audio::extract_audio_waveform(input_file)?;

        // Analyze audio to find segments
        segments = audio::analyze_audio(&audio_data, num_songs, video_info.duration)?;
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

    let write_files = false;
    // Process each detected segment
    if write_files {
        process_segments(input_file, &segments, setlist)?;
    }

    println!("Audio splitting complete!");
    Ok(())
}

fn process_segments(
    input_file: &str,
    segments: &[audio::AudioSegment],
    concert: SetList,
) -> Result<(), Box<dyn std::error::Error>> {
    let songs = concert.set_list;
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

    let output_dir = concert.metadata.folder_name();
    overwrite_dir(&output_dir)?;
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
        let output_file = format!("{}/{}.mp4", &output_dir, safe_title);

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
            &concert.metadata,
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

const CROP_TO_TEXT: &str = "scale=400:200,crop=iw/1.5:ih/5:0:160";

fn detect_song_boundaries_from_text(
    input_file: &str,
    artist: &str,
    songs: &[Song],
    video_info: &VideoInfo,
) -> Result<Vec<audio::AudioSegment>, Box<dyn std::error::Error>> {
    let total_duration = video_info.duration;
    let artist_cmp = artist.to_lowercase();
    // Create a temporary directory for frames
    let temp_dir = "temp_frames";
    overwrite_dir(temp_dir)?;

    let mut sorted_songs: Vec<Song> = songs
        .to_vec()
        .iter()
        .map(|song| Song {
            title: song.title.to_lowercase(),
        })
        .collect();
    // sorted_songs.clone_from_slice(songs);
    sorted_songs.sort_by(|a, b| a.title.len().partial_cmp(&b.title.len()).unwrap().reverse());

    println!("Extracting frames every 2 seconds for song title detection...");

    let every_2_seconds = "fps=1,select='not(mod(t,2))'";

    // Extract frames every 2 seconds with potential text overlays
    let status = ffmpeg::create_ffmpeg_command()
        .args(&[
            "-i",
            input_file,
            "-c:v",
            "png",
            "-frame_pts",
            "1",
            "-fps_mode",
            "passthrough", // Use original timestamps (replaces -vsync 0)
            "-qscale:v",
            "31", // Quality setting
            "-vf",
            &format!("{},{}", every_2_seconds, CROP_TO_TEXT), // Extract 1 frame every 2 seconds, focus on the text area
            &format!("{}/%d.png", temp_dir),                  // Use sequential numbering
        ])
        .status()?;

    println!("Frames extracted every 2 seconds successfully.");

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
    frames.sort_by(|a, b| {
        frame_number_from_image_filename(a).cmp(&frame_number_from_image_filename(b))
    });
    for frame_path in frames {
        // Extract frame number to calculate timestamp
        let frame_num = frame_number_from_image_filename(&frame_path);

        let song_titles_to_match = &sorted_songs
            .iter()
            .filter(|song|
            // skip already matched songs
            !song_title_matched.contains(&song.title))
            .map(|song| &song.title)
            .cloned()
            .collect::<Vec<_>>();

        // Define an iterator for different PSM options
        let psm_options = [Some("11"), None, Some("6")].iter();

        // Iterate through PSM options until we find a match
        for &psm in psm_options {
            // Run tesseract OCR on the frame with current PSM option
            let parsed =
                ocr::run_tesseract_ocr_parse(frame_path.to_str().unwrap(), &artist_cmp, psm)?;

            if let Some(lo @ (_, overlay)) = parsed {
                let title_time = match_song_titles(
                    input_file,
                    temp_dir,
                    &lo,
                    song_titles_to_match,
                    &artist_cmp,
                    frame_num,
                    video_info,
                )?;

                if let Some(time_match) = title_time {
                    song_title_matched.insert(time_match.0.clone());
                    song_start_times.push(time_match);
                    break; // Found a match, no need to try other PSM options
                } else if !overlay {
                    // If no text overlay was detected, no point in trying other PSM options
                    break;
                }
                // If we found text overlay but no title match, continue to next PSM option
            } else {
                // If no text was detected at all, no point in trying other PSM options
                break;
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

        segments.push(audio::AudioSegment {
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

fn match_song_titles(
    input_file: &str,
    temp_dir: &str,
    ocr_parse: &ocr::OcrParse,
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
        println!("Frame {}: Detected overlay: '{}'", frame_num, filtered_text);
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
            Some(matched @ (_, lev_dist)) => match best_match {
                None => {
                    best_match = Some((song_title.to_string(), matched));
                }
                Some((_, (_, best_dist))) => {
                    if lev_dist < best_dist {
                        best_match = Some((song_title.to_string(), matched));
                    }
                }
            },
        }
    }
    match best_match {
        None => Ok(None),
        Some((song_title, (line, lev_dist))) => {
            println!(
                "Match found! '{}' matches song '{}' frame={} dist={}",
                line, &song_title, frame_num, lev_dist,
            );
            match timestamp_for_song(
                input_file,
                temp_dir,
                &artist_cmp,
                &song_title,
                frame_num,
                video_info,
            ) {
                Ok(timestamp) => {
                    return Ok(Some((song_title.to_string(), timestamp)));
                }
                Err(e) => Err(e),
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
    let final_timestamp = if refined_timestamp > 0.0 && refined_timestamp < (frame_num as f64) {
        refined_timestamp
    } else {
        frame_num as f64
    };
    return Ok(final_timestamp);
}

fn refine_song_start_time(
    input_file: &str,
    temp_dir: &str,
    artist: &str,
    song_title: &str,
    initial_frame_num: usize,
    video_info: &VideoInfo,
) -> Result<f64, Box<dyn std::error::Error>> {
    let initial_timestamp = initial_frame_num as f64;
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

    // move forward to the closest keyframe
    let (_, after_opt) = video_info.get_nearest_keyframes_by_time(initial_frame_num as f64);
    let (end_fram_num, end_timestamp) = if let Some(after_key_frame) = after_opt {
        (
            after_key_frame,
            video_info.frames[after_key_frame].timestamp,
        )
    } else {
        panic!("Could not find keyframe after initial timestamp!")
    };
    println!(
        "looking back from keyframe {} after {}",
        end_timestamp, initial_timestamp
    );

    // Create a subdirectory for the refined frames
    let refined_dir = format!("{}/refined_{}", temp_dir, song_title.replace(" ", "_"));
    overwrite_dir(&refined_dir)?;

    // Extract frames at original video framerate for accuracy
    let fps = video_info.framerate;
    let status = ffmpeg::create_ffmpeg_command()
        .args(&[
            "-ss",
            &format!("{}", start_time),
            "-to",
            &format!("{}", end_timestamp),
            "-i",
            input_file,
            "-c:v",
            "png",
            "-vf",
            &format!("fps={},{},{}", fps, CROP_TO_TEXT, ffmpeg::BLACK_AND_WHITE), // Use original video framerate
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

    println!(
        "Analyzing {} refined frames for song title '{}' from {}s to {}s at {} fps",
        frames.len(),
        song_title,
        start_time,
        end_timestamp,
        fps
    );

    let mut earliest_match: Option<usize> = None;

    frames.sort_by(|a, b| {
        frame_number_from_image_filename(a)
            .cmp(&frame_number_from_image_filename(b))
            .reverse()
    });
    let frame_count = frames.len();

    // Try different PSM options until we find a valid result
    let psm_options = [
        (weights_for_stingy_extractor(), Some("11")),
        (weights_for_stingy_extractor(), None),
        (weights_for_greedy_extractor(), Some("6")),
        (weights_for_greedy_extractor(), Some("12")),
        (weights_for_greedy_extractor(), Some("10")),
    ];

    // Process each refined frame
    for frame_path in frames {
        let frame_file = frame_path.to_str().unwrap();
        /*
        let mut frame_file = frame_path.to_str().unwrap().to_string();
        frame_file.push_str("bw");
        let status = Command::new("convert").arg("-monochrome").arg(&frame_path)
        .arg(&frame_file).status()?;
        if !status.success() {
            return Err(format!("Failed to convert file to bw {}", &frame_file).into());
        }
        */
        // Extract frame number
        let frame_num = frame_number_from_image_filename(&frame_path);

        let mut earliest_match_found = false;
        for (weights, psm) in &psm_options {
            let result = ocr::run_tesseract_ocr_parse(&frame_file, artist, *psm)?;
            match result {
                None => continue,
                Some(parsed) => {
                    let (lines, overlay) = parsed;
                    // If we see the artist overlay that's good enough.
                    // On the initial fade in we might be able to see the artist name but not the song title.
                    if overlay
                        || matches_song_title_weighted(&lines, song_title, overlay, &weights)
                            .is_some()
                    {
                        if earliest_match.is_none() || frame_num < earliest_match.unwrap() {
                            earliest_match = Some(frame_num);
                            earliest_match_found = true;
                        }
                    }
                }
            }
        }

        // If we go to an earlier time finding the match becomes harder, so break once we can no longer match
        // wait for earliest_match to be present because of the keyframe adjustment
        if earliest_match.is_some() && !earliest_match_found {
            break;
        }
    }
    // println!("earliest match frame {:?}/{}", earliest_match, frame_count);

    // Return the earliest match if found, otherwise 0.0
    if let Some(earliest_match) = earliest_match {
        let subtracted_frame_num = frame_count as usize - earliest_match;
        let earliest_frame_num = end_fram_num - subtracted_frame_num as usize;
        // We never detect the fade soon enough
        // So go back to the previous keyframe
        // This then allows for video splitting without re-encoding
        let frame = video_info.frames[earliest_frame_num];
        let (before, __) = video_info.get_nearest_keyframes_by_time(frame.timestamp);
        let new_time = video_info.frames[before].timestamp;
        /*
        if earliest_frame_num > 1 {
            earliest_frame_num -= 1;
        }
        let frame = video_info.frames[earliest_frame_num];
        let new_time = frame.timestamp;
        */
        println!(
            "Successfully refined start time for '{}' from {}s to {}s (-{:.2}s) frame {}",
            song_title,
            end_timestamp,
            new_time,
            end_timestamp - new_time,
            earliest_match,
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

// This is really slow because it re-encodes
// If we just want audio we should be able to avoid re-encoding
// For video we can't do precision splitting without re-encoding.
// It may be possible, but the video will stutter at least before and after the first and last keyframes if we don't re-encode.
// It is possible to only re-encode just the portion outside the keyframes and stitch it back together.
// https://superuser.com/questions/1850814/how-to-cut-a-video-with-ffmpeg-with-no-or-minimal-re-encoding
fn extract_segment(
    input_file: &str,
    output_file: &str,
    start_time: f64,
    end_time: f64,
    song_title: Option<&str>,
    concertdata: &SetMetaData,
    track_number: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = ffmpeg::create_ffmpeg_command();

    cmd.args(&[
        "-i",
        input_file,
        "-ss",
        &format!("{:.3}", start_time),
        "-to",
        &format!("{:.3}", end_time),
    ]);

    cmd.args(&["-metadata", &format!("artist={}", concertdata.artist)]);

    // Add metadata if available
    if let Some(title) = song_title {
        cmd.args(&["-metadata", &format!("title={}", title)]);
    }

    if let Some(ref album) = concertdata.album {
        cmd.args(&["-metadata", &format!("album={}", album)]);
    }

    if let Some(year_value) = concertdata.year() {
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
