use std::process::Command;

pub fn create_ffmpeg_command() -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(&["-hide_banner", "-loglevel", "warning"]);
    cmd.stdout(std::process::Stdio::null());
    cmd
}

pub fn create_ffprobe_command() -> Command {
    let mut cmd = Command::new("ffprobe");
    cmd.args(&["-hide_banner", "-loglevel", "warning"]);
    cmd
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
