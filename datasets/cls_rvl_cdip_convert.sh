#!/bin/bash
SOURCE_DIR=$1

classes=(letter
form
email
handwritten
advertisement
scientific\ report
scientific\ publication
specification
file\ folder
news\ article
budget
invoice
presentation
questionnaire
resume
memo)

for LABELFILE in $SOURCE_DIR/labels/*
do
  split=`basename ${LABELFILE%%.*}`
  echo "SPLIT: $split"
  TARGET=$SOURCE_DIR/$split
  mkdir -p $TARGET

  for c in "${classes[@]}"
  do
    mkdir $TARGET/${c// /_}
  done

  IFS=$'\n'
  for next in `cat $LABELFILE`
  do
    echo "     $next"
    FILEPATH="$(cut -d' ' -f1 <<< $next)"
    LABEL="$(cut -d' ' -f2 <<< $next)"
    mv $SOURCE_DIR/images/$FILEPATH $TARGET/${classes[$LABEL]// /_}/$(basename $FILEPATH)
  done
done
exit 0
