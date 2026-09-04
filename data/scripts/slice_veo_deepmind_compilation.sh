#!/bin/bash
# slice_veo_deepmind_compilation.sh
#
# Slices all 23 Veo scenes from Google DeepMind's official Veo demo compilation:
# https://www.youtube.com/watch?v=1L5OCzHuwCA
#
# Usage:
#   1. Download the full video first:
#      yt-dlp -o "data/raw/temp/veo_deepmind_compilation.%(ext)s" "https://www.youtube.com/watch?v=1L5OCzHuwCA"
#
#   2. Run this script from your project root:
#      bash data/scripts/slice_veo_deepmind_compilation.sh

INPUT=$(ls data/raw/temp/veo_deepmind_compilation.* 2>/dev/null | head -1)
OUTPUT_DIR="data/raw/veo"

if [ -z "$INPUT" ]; then
    echo "Error: input file not found. Download it first:"
    echo "  yt-dlp -o \"data/raw/temp/veo_deepmind_compilation.%(ext)s\" \"https://www.youtube.com/watch?v=1L5OCzHuwCA\""
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Slicing 23 Veo scenes from $INPUT -> $OUTPUT_DIR"
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
slice "00:00:01" "00:00:15" "veo_swimming_jellyfish"
slice "00:00:15" "00:00:23" "veo_lama_in_field"
slice "00:00:23" "00:00:30" "veo_night_cityscape"
slice "00:00:30" "00:00:39" "veo_lone_rider"
slice "00:00:39" "00:00:49" "veo_noir_alley"
slice "00:00:49" "00:00:55" "veo_mountain_dog"
slice "00:00:55" "00:01:03" "veo_lake_view"
slice "00:01:03" "00:01:11" "veo_lost_waterfall"
slice "00:01:11" "00:01:19" "veo_flying_through_space"
slice "00:01:19" "00:01:26" "veo_swimming_turtle"
slice "00:01:26" "00:01:35" "veo_lighthouse_cliff"
slice "00:01:35" "00:01:43" "veo_crochet_elephant"
slice "00:01:43" "00:01:51" "veo_colour_explosion"
slice "00:01:51" "00:01:59" "veo_blooming_flower"
slice "00:01:59" "00:02:07" "veo_northern_lights"
slice "00:02:07" "00:02:14" "veo_colorful_lizard"
slice "00:02:14" "00:02:23" "veo_film_noir_reading"
slice "00:02:23" "00:02:31" "veo_bubble_bath"
slice "00:02:31" "00:02:39" "veo_driving_through"
slice "00:02:39" "00:02:46" "veo_city_lights"
slice "00:02:46" "00:02:54" "veo_bbq"
slice "00:02:54" "00:03:03" "veo_hot_air_balloon"
slice "00:03:03" "00:03:15" "veo_paradise_island_and_dancing_lamas"

echo ""
echo "Done. Clips saved to $OUTPUT_DIR"
echo "Next: run reencode.py on data/raw/veo -> data/reencoded/veo"