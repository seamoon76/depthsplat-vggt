#!/bin/bash

for scene_dir in */; do
    echo "Processing $scene_dir"

    scene_dir=${scene_dir%/}

    sparse_path="${scene_dir}/sparse"
    sparse_norm_path="${scene_dir}/sparse_norm"
    ref_gt="${scene_dir}/ref_images.txt"
    ref_norm="${scene_dir}/ref_images_norm.txt"
    output_gt="${scene_dir}/with_gt"
    output_with_norm="${scene_dir}/with_norm"
    output_norm_with_norm="${scene_dir}/norm_with_norm"

    mkdir -p "$output_gt"
    mkdir -p "$output_with_norm"
    mkdir -p "$output_norm_with_norm"

    # Case 1: sparse + ref_images.txt → with_gt
    if [ -d "$sparse_path" ] && [ -f "$ref_gt" ]; then
        echo "  Aligning sparse + ref_images.txt → with_gt"
        colmap model_aligner \
            --input_path "$sparse_path" \
            --output_path "$output_gt" \
            --ref_images_path "$ref_gt" \
            --ref_is_gps 0 \
            --alignment_type custom \
            --alignment_max_error 3.0
    else
        echo "  Skipping with_gt: missing sparse or ref_images.txt"
    fi

    # Case 2: sparse + ref_images_norm.txt → with_norm
    if [ -d "$sparse_path" ] && [ -f "$ref_norm" ]; then
        echo "  Aligning sparse + ref_images_norm.txt → with_norm"
        colmap model_aligner \
            --input_path "$sparse_path" \
            --output_path "$output_with_norm" \
            --ref_images_path "$ref_norm" \
            --ref_is_gps 0 \
            --alignment_type custom \
            --alignment_max_error 3.0
    else
        echo "  Skipping with_norm: missing sparse or ref_images_norm.txt"
    fi

    # Case 3: sparse_norm + ref_images_norm.txt → norm_with_norm
    if [ -d "$sparse_norm_path" ] && [ -f "$ref_norm" ]; then
        echo "  Aligning sparse_norm + ref_images_norm.txt → norm_with_norm"
        colmap model_aligner \
            --input_path "$sparse_norm_path" \
            --output_path "$output_norm_with_norm" \
            --ref_images_path "$ref_norm" \
            --ref_is_gps 0 \
            --alignment_type custom \
            --alignment_max_error 3.0
    else
        echo "  Skipping norm_with_norm: missing sparse_norm or ref_images_norm.txt"
    fi

done
