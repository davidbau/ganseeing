#!/bin/bash

MODEL=${1:-church}

# First, train an encoder.
python -m ganpaint.train_multilayer_inv --model=${MODEL}
for LAYER in $(seq 4); do
python -m ganpaint.train_onelayer_inv --invert_layer=${LAYER} --model=${MODEL}
done
for BOTTOM in $(seq 5); do
python -m ganpaint.train_hybrid_bottom --model=${MODEL} --bottom ${BOTTOM}
done

