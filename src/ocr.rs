use std::fs::{self};
use std::process::Command;

use stringmetrics::{levenshtein_weight, LevWeights};

pub type OcrParse = (Vec<String>, bool);

pub fn run_tesseract_ocr_parse(
    image_path: &str,
    artist_cmp: &str,
    psm: Option<&str>,
) -> Result<Option<OcrParse>, Box<dyn std::error::Error>> {
    let text = run_tesseract_ocr(image_path, psm)?;
    return match parse_tesseract_output(&text, &artist_cmp) {
        Some(result) => Ok(Some(result)),
        None => Ok(None),
    };
}

pub fn run_tesseract_ocr(
    image_path: &str,
    psm: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    // Run tesseract OCR on the image
    let mut cmd = Command::new("tesseract");
    cmd.arg(image_path).arg(image_path);

    // Add PSM option if specified
    if let Some(psm_value) = psm {
        cmd.args(&["--psm", psm_value]);
    }

    let output = cmd
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()?;

    if !output.status.success() {
        let error_message = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Tesseract OCR failed: {}", error_message).into());
    }

    // Read the OCR result from the output text file
    let out_txt_path = format!("{}.txt", image_path);
    let text = fs::read_to_string(&out_txt_path)?;

    Ok(text)
}

pub fn parse_tesseract_output(text: &str, artist: &str) -> Option<OcrParse> {
    let detected_text = text.trim();

    // Skip if empty or too short
    if detected_text.len() < 4 {
        return None;
    }

    // Filter out empty lines
    let lines: Vec<String> = detected_text
        .lines()
        .map(|line| line.trim())
        .filter(|line| line.len() > 0)
        .map(|line| line.to_string())
        .collect();

    if lines.is_empty() {
        return None;
    }

    let is_overlay = fuzzy_match_artist(&lines[0], artist);
    Some((lines, is_overlay))
}

fn fuzzy_match_artist(line_input: &str, artist_input: &str) -> bool {
    // Check if this is an overlay with artist at the top
    let line = line_input.to_lowercase().replace(" ", "");
    let artist = artist_input.to_lowercase().replace(" ", "");
    return !artist.is_empty() && !line.is_empty() && {
        // starts_with here allows tesseract to imagine extra characters at the end
        line.starts_with(&artist) ||
        // Check if the first line is a subset of the artist name
        // That should mean that tesseract missed the last few letters
        (artist.starts_with(&line) && ((line.len() as f64) / (artist.len() as f64) >= 0.7) ||
            { // Also allow tesseract to get the last few letters wrong
                let split_at = artist.chars().count() * 7 / 10;
                line.len() > split_at && {
                    let artist_start = line.chars().take(split_at).collect::<String>();
                    artist.starts_with(&artist_start)
                }
            }
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        assert!(fuzzy_match_artist("John Doe", "John Doe"));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(fuzzy_match_artist("JOHN DOE", "john doe"));
        assert!(fuzzy_match_artist("john doe", "JOHN DOE"));
    }

    #[test]
    fn test_space_handling() {
        assert!(fuzzy_match_artist("JohnDoe", "John Doe"));
        assert!(fuzzy_match_artist("John Doe", "JohnDoe"));
    }

    #[test]
    fn test_empty_artist() {
        assert!(!fuzzy_match_artist("John Doe", ""));
    }

    #[test]
    fn test_empty_line() {
        assert!(!fuzzy_match_artist("", "John Doe"));
    }

    #[test]
    fn test_partial_match_start() {
        assert!(fuzzy_match_artist("John Doe Extra", "John Doe"));
    }

    #[test]
    fn test_no_match() {
        assert!(!fuzzy_match_artist("Jane Smith", "John Doe"));
    }

    #[test]
    fn test_ratio_threshold() {
        assert!(!fuzzy_match_artist("johndo", "johndoe890"));
        assert!(fuzzy_match_artist("johndoe", "johndoe890"));
    }

    #[test]
    fn test_fuzzy_name() {
        assert!(fuzzy_match_artist("Megan Moror", "Megan Moroney"));
    }
}

/// Normalize text by removing punctuation and spaces, keeping only alphanumeric characters.
/// Also converts all characters to lowercase for case-insensitive comparison.
fn normalize_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

pub fn matches_song_title(
    lines: &[String],
    song_title: &str,
    is_overlay: bool,
) -> Option<(String, u32)> {
    let title_normalized = normalize_text(song_title);
    let weights = LevWeights::new(2, 2, 1);
    let levenshtein_limit = 3;

    for line in lines {
        let line_normalized = normalize_text(line);

        // Check for exact or partial match
        if line_normalized.contains(&title_normalized) {
            return Some((line.clone(), 0));
        }
        if !is_overlay {
            continue;
        }
        // If we have an overlay and no exact match was found, try fuzzy matching
        let lev = levenshtein_weight(
            &line_normalized,
            &title_normalized,
            levenshtein_limit + 10,
            &weights,
        );
        // println!("levenshtein distance: {}. {}. {}", lev, song_title, line);
        if lev <= levenshtein_limit {
            return Some((line.clone(), lev));
        }
        if title_normalized.starts_with(&line_normalized) {
            if (line_normalized.len() as f64 / title_normalized.len() as f64) >= 0.4 {
                return Some((
                    line.clone(),
                    (title_normalized.len() - line_normalized.len()) as u32,
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests_matches_song_title {
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
        let ocr_lines = vec!["helo wrld".to_string()]; // OCR might miss letters
        assert!(!matches_song_title(&ocr_lines, "hello world", false).is_some()); // Should fail without overlay
        assert!(matches_song_title(&ocr_lines, "hello world", true).is_some()); // Should pass with overlay

        // Test non-matches
        let other_lines = vec!["completely different".to_string()];
        assert!(!matches_song_title(&other_lines, "test song", true).is_some());
    }
}
