#!/usr/bin/env bash
# Batch TPM Data Generation Script

# Get the list of videos in the input directory
shopt -s nullglob
input_dir="/workspace/STRIVE_Datasets/RealWorld/videos"
output_base_dir="/workspace/tpm_dataset"
output_dir_prefix="realworld_"
for video_path in "$input_dir"/*.mp4; do
    video_filename=$(basename "$video_path")
    video_name="${video_filename%.*}"
    output_dir="${output_base_dir}/${output_dir_prefix}${video_name}"
    
    echo "Processing $video_path..."
    python code/scripts/run_tpm_data_gen_pipeline.py \
        --input "$video_path" \
        --output-dir "$output_dir" \
        --save-detected-tracks \
        --log-level INFO
done
