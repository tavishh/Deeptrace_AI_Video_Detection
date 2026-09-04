#!/bin/bash
# rename_kling_community.sh
#
# Renames community-sourced Kling clips to match project naming convention
# (kling_descriptor_format) and copies them into data/raw/kling/
#
# Usage:
#   bash data/scripts/rename_kling_community.sh /path/to/KlingAI

SOURCE_DIR="${1:-data/raw/kling}"
OUTPUT_DIR="data/raw/kling"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: source directory not found: $SOURCE_DIR"
    echo "Usage: bash data/scripts/rename_kling_community.sh /path/to/KlingAI"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Renaming and copying Kling community clips -> $OUTPUT_DIR"
echo ""

copy_clip() {
    local src="$SOURCE_DIR/$1"
    local dst="$OUTPUT_DIR/$2.mp4"

    if [ ! -f "$src" ]; then
        echo "  [missing] $1"
        return
    fi
    if [ -f "$dst" ]; then
        echo "  [skip] $2 already exists"
        return
    fi

    cp "$src" "$dst"
    echo "  [ok] $1 -> $2.mp4"
}

copy_clip "ancient_china_king_queen.mp4"        "kling_ancient_china_king_queen"
copy_clip "army_wife_death.mp4"                  "kling_army_wife_death"
copy_clip "cat_astronaut_in_space.mp4"           "kling_cat_astronaut_space"
copy_clip "cinematic_skateboarding_lens.mp4"     "kling_cinematic_skateboarding"
copy_clip "cyborg_in_space.mp4"                  "kling_cyborg_in_space"
copy_clip "deer_spotting_in_a_misty_forest.mp4"  "kling_deer_misty_forest"
copy_clip "distressed_woman_in_rain.mp4"         "kling_distressed_woman_rain"
copy_clip "driving_curved_night.mp4"             "kling_driving_curved_night"
copy_clip "family_watching_television.mp4"       "kling_family_watching_tv"
copy_clip "futuristic_hologram.mp4"              "kling_futuristic_hologram"
copy_clip "halo_soldiers_discovery.mp4"          "kling_halo_soldiers_discovery"
copy_clip "influencer_collagen_mask.mp4"         "kling_influencer_collagen_mask"
copy_clip "intergalactic_alien_fight.mp4"        "kling_intergalactic_alien_fight"
copy_clip "lone_explorer_desert.mp4"             "kling_lone_explorer_desert"
copy_clip "model_girl_in_music_video.mp4"        "kling_model_girl_music_video"
copy_clip "monster_alien_jungle_under_three_moon.mp4" "kling_monster_alien_jungle"
copy_clip "ninja_rooftop_fight.mp4"              "kling_ninja_rooftop_fight"
copy_clip "parasitical_creature.mp4"             "kling_parasitical_creature"
copy_clip "robocop_saves_minnesota.mp4"          "kling_robocop_saves_minnesota"
copy_clip "soldier_in_war.mp4"                   "kling_soldier_in_war"
copy_clip "sports_changing_room.mp4"             "kling_sports_changing_room"
copy_clip "vampire_dentist.mp4"                  "kling_vampire_dentist"
copy_clip "woman_in_distress_rain.mp4"           "kling_woman_in_distress_rain"

echo ""
echo "Done. Check $OUTPUT_DIR for renamed clips."