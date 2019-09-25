#!/bin/bash

BASELINE=datasets/lsun/church_outdoor_train
PGAN=results/imagesample/church/size_10000

echo 'Church training set to self'
python -m seeing.fsd \
    --histout results/fsd/fsd_church_selfcompare.png \
    --labelcount 30 --dpi 300 \
    --cachedir results/fsd/cache \
    ${BASELINE} ${BASELINE}

# python -m seeing.fsd ${PGAN} ${PGAN}
echo 'Church progressive gan'
python -m seeing.fsd \
    --histout results/fsd/fsd_pgan_church.png \
    --dpi 300 \
    --cachedir results/fsd/cache \
    ${BASELINE} ${PGAN}

BASELINE=datasets/lsun/bedroom_train
PGAN=results/imagesample/bedroom/size_10000

echo 'Bedroom training set to self'
python -m seeing.fsd \
    --histout results/fsd/fsd_bedroom_selfcompare.png \
    --labelcount 30 --dpi 300 \
    --cachedir results/fsd/cache \
    ${BASELINE} ${BASELINE}

echo 'Bedroom progressive gan'
python -m seeing.fsd \
    --histout results/fsd/fsd_pgan_bedroom.png \
    --dpi 300 \
    --cachedir results/fsd/cache \
    ${BASELINE} ${PGAN}


for ARCH in  \
    minibatch-stddev \
    wgangp \
    # stylegan \

do

BASELINE=datasets/lsun/bedroom_train
DIRNAME=results/imagesample/${ARCH}/size_10000

echo ${ARCH}
python -m seeing.fsd \
    --histout results/fsd/fsd_${ARCH}_bedroom.png \
    --dpi 300 \
    --cachedir results/fsd/cache \
    ${BASELINE} ${DIRNAME}

done


