#!/bin/bash

# GPU_ID=${1}
# DISPLAY_ID=$((GPU_ID*10+1))

for SIZE in 10000 1000;
do

CONFIGS=(
    wgangp
    minibatch-stddev
)

for EXPNAME in ${CONFIGS[@]}
do

MODEL_CONSTRUCTOR="seeing.setting.load_proggan_ablation('${EXPNAME}')"

echo "Running $EXPNAME"

python -m seeing.samplegan \
      --model "${MODEL_CONSTRUCTOR}" \
      --outdir results/imagesample/${EXPNAME}/size_${SIZE} \
      --size ${SIZE} \

done

CONFIGS=(
    bedroom
    church
)

for EXPNAME in ${CONFIGS[@]}
do

MODEL_CONSTRUCTOR="seeing.setting.load_proggan('${EXPNAME}')"

echo "Running $EXPNAME"

python -m seeing.samplegan \
      --model "${MODEL_CONSTRUCTOR}" \
      --outdir results/imagesample/${EXPNAME}/size_${SIZE} \
      --size ${SIZE} \

done

done
