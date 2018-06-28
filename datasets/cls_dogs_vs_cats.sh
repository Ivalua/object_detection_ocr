#!/bin/bash
SOURCE_DIR=$1
mkdir -p /sharedfiles/dogs_vs_cats/train/dogs
mkdir -p /sharedfiles/dogs_vs_cats/train/cats
mkdir -p /sharedfiles/dogs_vs_cats/validation/dogs
mkdir -p /sharedfiles/dogs_vs_cats/validation/cats

for i in $(seq 0 9999) ; do cp $SOURCE_DIR/dog.$i.jpg /sharedfiles/dogs_vs_cats/train/dogs/ ; done
for i in $(seq 0 9999) ; do cp $SOURCE_DIR/cat.$i.jpg /sharedfiles/dogs_vs_cats/train/cats/ ; done
for i in $(seq 10000 12499) ; do cp $SOURCE_DIR/cat.$i.jpg /sharedfiles/dogs_vs_cats/validation/cats/ ; done
for i in $(seq 10000 12499) ; do cp $SOURCE_DIR/dog.$i.jpg /sharedfiles/dogs_vs_cats/validation/dogs/ ; done
