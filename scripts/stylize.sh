#!/usr/bin/env bash

set -e

for folder in SPIKE_images
do
  for i in {1..4}
  do
    python "$PROJECT_ROOT"/gwd/stylize/run.py \
      --content-dir=/home/ubuntu/data/SPIKE/SPIKE_Dataset/positive \
      --style-dir=/home/ubuntu/data/global-wheat-detection/test \
      --output-dir=/home/ubuntu/data/global-wheat-detection/stylized_${folder}_v${i}
  done
done
