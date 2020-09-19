#!/usr/bin/env bash

set -e

for folder in train crops_fold0
do
  python "$PROJECT_ROOT"/gwd/colorization/generate.py \
    --img_pattern=/content/${folder}/*jpg \
    --weights_path=/content/dumps/权重/pix2pix_gen.pth \
    --output_root=/content/colored_${folder}
done
