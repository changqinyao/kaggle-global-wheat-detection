#!/usr/bin/env bash

set -e

for folder in train crops_fold0 SPIKE_images
do
  for i in 1 2 3 4
  do
    python "$PROJECT_ROOT"/gwd/stylize/run.py \
      --content-dir=/content/${folder} \
      --style-dir=/content/test \
      --output-dir=/content/stylized_${folder}_v${i}
  done
done
