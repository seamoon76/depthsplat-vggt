#!/bin/bash

ROOT_DIR="/home/jiaysun/re10k_vggsfm"

for SCENE_DIR in "$ROOT_DIR"/*; do
    if [ -d "$SCENE_DIR" ]; then
        echo "Processing scene: $SCENE_DIR"
        python colmap2mvsnet.py --dense_folder "$SCENE_DIR"
    fi
done
