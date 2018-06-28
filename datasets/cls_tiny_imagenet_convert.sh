#!/bin/bash
SOURCE_DIR=$1

# create class directories for validation dataset
for c in $SOURCE_DIR/train/*
do
  CLASS=`basename $c`
  echo $CLASS
  mkdir $SOURCE_DIR/val/$CLASS
done

# copy images at the class directory as asked by Keras Image Flow
IFS=$'\n'
for next in `cat $SOURCE_DIR/val/val_annotations.txt`
do
  next=`echo $next | tr -s ' '`
  echo "     $next"
  FILEPATH="$(cut -d'	' -f1 <<< $next)"
  LABEL="$(cut -d'	' -f2 <<< $next)"
  cp $SOURCE_DIR/val/images/$FILEPATH $SOURCE_DIR/val/$LABEL/$FILEPATH
done

exit 0
