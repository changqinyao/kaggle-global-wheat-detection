#!/usr/bin/env bash

set -e

for folder in train crops_fold0
do
  python "$PROJECT_ROOT"/gwd/colorization/generate.py \
    --img_pattern=/home/ubuntu/data/global-wheat-detection/${folder}/*jpg \
    --weights_path=/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/dumps/pix2pix_gen.pth \
    --output_root=/home/ubuntu/data/global-wheat-detection/colored_${folder}
done
