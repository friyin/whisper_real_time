#!/bin/sh

datetime=$(date +%Y%M%d_%H%m%S)
outfile=out_${datetime}.txt
model=large-v2
energy_threshold=500

python transcribe_rt.py \
  --non_english \
  --model ${model} \
  --energy_threshold ${energy_threshold} \
  --output_file ${outfile}
