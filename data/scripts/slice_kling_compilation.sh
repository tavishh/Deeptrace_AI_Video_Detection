#!/bin/bash
# slice_kling_compilation.sh
#
# Slices 20 Kling AI scenes from community compilation:
# https://www.youtube.com/watch?v=s5j_m68SC4U
# Source: "Compilation AI VIDEOS 2 - Multiples Short Clips Created with Kling AI"
# Attribution: Images via Midjourney, Animation via Kling AI (explicit in description)
#
# Usage:
#   1. Download the full video first:
#      yt-dlp -o "data/raw/temp/kling_compilation.%(ext)s" "https://www.youtube.com/watch?v=s5j_m68SC4U"
#
#   2. Run this script from your project root:
#      bash data/scripts/slice_kling_compilation.sh

INPUT=$(ls data/raw/temp/kling_compilation.* 2>/dev/null | head -1)
OUTPUT_DIR="data/raw/kling"

if [ -z "$INPUT" ]; then
    echo "Error: input file not found. Download it first:"
    echo "  yt-dlp -o \"data/raw/temp/kling_compilation.%(ext)s\" \"https://www.youtube.com/watch?v=s5j_m68SC4U\""
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Slicing 20 Kling scenes from $INPUT -> $OUTPUT_DIR"
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

slice "00:00:00" "00:00:10" "kling_girl_on_dinosaur"
slice "00:00:10" "00:00:20" "kling_woman_walking_in_forest"
slice "00:00:20" "00:00:30" "kling_woman_futuristic_city"
slice "00:00:30" "00:00:40" "kling_robot_tai_chi"
slice "00:00:40" "00:00:50" "kling_robot_tai_chi_on_fire"
slice "00:00:50" "00:01:00" "kling_woman_flying_vehicle"
slice "00:01:00" "00:01:10" "kling_woman_driving_vintage_car"
slice "00:01:10" "00:01:20" "kling_alien_flying_car"
slice "00:01:20" "00:01:30" "kling_astronaut_landing_desert"
slice "00:01:30" "00:01:40" "kling_woman_biker_dirt_road"
slice "00:01:40" "00:01:50" "kling_cat_as_king"
slice "00:01:50" "00:01:55" "kling_royal_cat_eating_feast"
slice "00:01:55" "00:02:00" "kling_fantasy_land_golden_egg"
slice "00:02:00" "00:02:10" "kling_king_kong_skyscraper"
slice "00:02:10" "00:02:20" "kling_back_to_future_car"
slice "00:02:20" "00:02:30" "kling_woman_coal_chamber"
slice "00:02:30" "00:02:40" "kling_astronauts_walking_forest"
slice "00:02:40" "00:02:50" "kling_girl_animals_fantasy_land"
slice "00:02:50" "00:03:00" "kling_aliens_unknown_planet"
slice "00:03:00" "00:03:10" "kling_girl_on_flying_bird"

echo ""
echo "Done. Clips saved to $OUTPUT_DIR"
echo "Next: run reencode.py on data/raw/kling -> data/reencoded/kling"