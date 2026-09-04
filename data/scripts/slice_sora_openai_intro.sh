#!/bin/bash
# slice_sora_openai_intro.sh
#
# Slices all 34 Sora scenes from OpenAI's official introduction video:
# https://www.youtube.com/watch?v=HK6y8DAPN_0
#
# Usage:
#   1. Download the full video first:
#      yt-dlp -o "data/raw/temp/sora_openai_intro.mp4" "https://www.youtube.com/watch?v=HK6y8DAPN_0"
#
#   2. Run this script from your project root:
#      bash data/scripts/slice_sora_openai_intro.sh

INPUT="data/raw/temp/sora_openai_intro.mp4.webm"
OUTPUT_DIR="data/raw/sora"

mkdir -p "$OUTPUT_DIR"

echo "Slicing 34 Sora scenes from $INPUT -> $OUTPUT_DIR"
echo ""

slice() {
    local start=$1
    local end=$2
    local name=$3
    local out="$OUTPUT_DIR/${name}.mp4"

    if [ -f "$out" ]; then
        echo "  [skip] $name already exists"
        return
    fi

    ffmpeg -i "$INPUT" -ss "$start" -to "$end" -c copy "$out" -loglevel error
    if [ $? -eq 0 ]; then
        echo "  [ok] $name ($start -> $end)"
    else
        echo "  [fail] $name"
    fi
}

# Chapter timestamps from YouTube description
# Format: slice "HH:MM:SS" "HH:MM:SS" "clip_name"

slice "00:00:09" "00:00:22" "sora_dancing_kangaroo"
slice "00:00:22" "00:00:43" "sora_snow_dogs"
slice "00:00:43" "00:00:55" "sora_river_birds"
slice "00:00:55" "00:01:08" "sora_petri_dish_pandas"
slice "00:01:08" "00:01:21" "sora_big_sur"
slice "00:01:21" "00:01:40" "sora_movie_trailer_astronaut"
slice "00:01:40" "00:01:57" "sora_coffee_pirates"
slice "00:01:57" "00:02:09" "sora_tokyo_snow"
slice "00:02:09" "00:02:30" "sora_cyberpunk_robot"
slice "00:02:30" "00:02:43" "sora_candle_monster"
slice "00:02:43" "00:03:04" "sora_the_offroader"
slice "00:03:04" "00:03:27" "sora_paper_origami"
slice "00:03:27" "00:03:38" "sora_nosy_cat"
slice "00:03:38" "00:03:51" "sora_woolly_mammoths"
slice "00:03:51" "00:04:14" "sora_lagos"
slice "00:04:14" "00:04:37" "sora_television_gallery"
slice "00:04:37" "00:04:59" "sora_cloud_reader"
slice "00:04:59" "00:05:11" "sora_miniature_construction"
slice "00:05:11" "00:05:38" "sora_gold_rush_aerial"
slice "00:05:38" "00:05:49" "sora_fairytale_furball"
slice "00:05:49" "00:06:12" "sora_amalfi_coast_aerial"
slice "00:06:12" "00:06:31" "sora_tokyo_tourist"
slice "00:06:31" "00:06:42" "sora_blossoming_flower"
slice "00:06:42" "00:07:05" "sora_art_museum"
slice "00:07:05" "00:07:28" "sora_solemn_gentleman"
slice "00:07:28" "00:07:47" "sora_eye_closeup"
slice "00:07:47" "00:07:58" "sora_chinese_new_year"
slice "00:07:58" "00:08:17" "sora_surfing_otter"
slice "00:08:17" "00:08:31" "sora_dalmatian_in_window"
slice "00:08:31" "00:08:42" "sora_tokyo_train"
slice "00:08:42" "00:08:53" "sora_zen_garden_gnome"
slice "00:08:53" "00:09:16" "sora_flock_paper_planes"
slice "00:09:16" "00:09:34" "sora_lost_lone_wolf"

echo ""
echo "Done. Clips saved to $OUTPUT_DIR"
echo "Next: run reencode.py on data/raw/sora -> data/reencoded/sora"
