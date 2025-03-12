use std::fs;
use std::io;
use std::path::Path;

pub fn overwrite_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    if fs::exists(&path)? {
        fs::remove_dir_all(&path)?;
    }
    fs::create_dir(&path)
}

pub fn sanitize_filename(input: &str) -> String {
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
