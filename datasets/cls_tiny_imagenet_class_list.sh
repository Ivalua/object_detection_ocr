#!/bin/bash
for next in `cat /sharedfiles/tiny-imagenet-200/wnids.txt`
do
  l="$l, '$next'"
  echo "     $next"
done
echo $l

exit 0
