#!/bin/bash

MODEL=${1:-church}

# First, train an encoder.
python -m seeing.train_multilayer_inv --model=${MODEL}
for LAYER in $(seq 4); do
python -m seeing.train_onelayer_inv --invert_layer=${LAYER} --model=${MODEL}
done
python -m seeing.train_hybrid_inv --model=${MODEL}

# Wait for individual layers to be encoded
until [[ -f results/${MODEL}/invert_over5_resnet/done.txt && \
        -f results/${MODEL}/invert_layer_1_cse/done.txt &&   \
        -f results/${MODEL}/invert_layer_2_cse/done.txt &&   \
        -f results/${MODEL}/invert_layer_3_cse/done.txt &&   \
        -f results/${MODEL}/invert_layer_4_cse/done.txt ]]
do
    sleep 5
done


# And do 200 train, val and GAN-generated images.
for IMAGENUM in $(seq 0 200)
do

for SOURCE in train val gan
do

# python -m seeing.optimize_z_lbfgs \
python -m seeing.optimize_residuals \
    --image_number ${IMAGENUM} \
    --image_source ${SOURCE} \
    --model ${MODEL}

done

done
