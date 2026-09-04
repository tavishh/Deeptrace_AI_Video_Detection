import subprocess
import os

output_dir = "/kaggle/working/real_clips"
os.makedirs(output_dir, exist_ok=True)

failed = []
success = 0

for _, row in sampled.iterrows():
    youtube_id = row['youtube_id']
    start = int(row['time_start'])
    end = int(row['time_end'])
    label = row['label'].replace(' ', '_')
    clip_id = f"real_{label}_{youtube_id}"
    output_path = f"{output_dir}/{clip_id}.mp4"

    if os.path.exists(output_path):
        success += 1
        continue

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "--output", output_path,
        "--no-playlist",
        "--quiet",
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0 and os.path.exists(output_path):
        success += 1
        print(f"[ok] {clip_id}")
    else:
        failed.append(clip_id)
        print(f"[fail] {clip_id}")

print(f"\nDone. {success} succeeded, {len(failed)} failed.")
print(f"Failed: {failed}")