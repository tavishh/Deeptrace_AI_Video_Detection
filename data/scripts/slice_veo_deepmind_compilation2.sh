#!/bin/bash
# slice_veo_deepmind_compilation2.sh
#
# Slices 18 Veo scenes from Google DeepMind's second official Veo demo compilation:
# https://www.youtube.com/watch?v=KIRh5ZnWZ5o
#
# Usage:
#   1. Download the full video first:
#      yt-dlp -o "data/raw/temp/veo_deepmind_compilation2.%(ext)s" "https://www.youtube.com/watch?v=KIRh5ZnWZ5o"
#
#   2. Run this script from your project root:
#      bash data/scripts/slice_veo_deepmind_compilation2.sh

INPUT=$(ls data/raw/temp/veo_deepmind_compilation2.* 2>/dev/null | head -1)
OUTPUT_DIR="data/raw/veo"

if [ -z "$INPUT" ]; then
    echo "Error: input file not found. Download it first:"
    echo "  yt-dlp -o \"data/raw/temp/veo_deepmind_compilation2.%(ext)s\" \"https://www.youtube.com/watch?v=KIRh5ZnWZ5o\""
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Slicing 18 Veo scenes from $INPUT -> $OUTPUT_DIR"
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

slice "00:00:06" "00:00:13" "veo_bird_in_forest"
slice "00:00:13" "00:00:22" "veo_blue_creature"
slice "00:00:22" "00:00:33" "veo_feathery_bird"
slice "00:00:33" "00:00:41" "veo_abstract_building"
slice "00:00:41" "00:00:49" "veo_floating_buildings"
slice "00:00:49" "00:00:59" "veo_gemstone_world"
slice "00:00:59" "00:01:07" "veo_underwater_painted_lady"
slice "00:01:07" "00:01:16" "veo_otter_in_stream"
slice "00:01:16" "00:01:24" "veo_silhouette_walking_into_light"
slice "00:01:24" "00:01:31" "veo_leopard_in_snow"
slice "00:01:31" "00:01:38" "veo_slow_motion_dog_balloon"
slice "00:01:38" "00:01:49" "veo_lady_sunglasses_busy_square"
slice "00:01:49" "00:01:57" "veo_dog_on_surfboard"
slice "00:01:57" "00:02:05" "veo_cat_on_skateboard"
slice "00:02:05" "00:02:13" "veo_cat_on_snowboard"
slice "00:02:13" "00:02:21" "veo_dog_with_sunglasses"
slice "00:02:21" "00:02:29" "veo_black_cat_top_hat"
slice "00:02:29" "00:02:40" "veo_colorful_dog"

echo ""
echo "Done. Clips saved to $OUTPUT_DIR"
echo "Next: run reencode.py on data/raw/veo -> data/reencoded/veo"