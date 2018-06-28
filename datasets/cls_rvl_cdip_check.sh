#!/bin/bash 
ls /sharedfiles/rvl_cdip/train/ | wc -l

for split in train test val
do
	for d in /sharedfiles/rvl_cdip/$split/*
	do
		echo $d
		ls $d/ | wc -l
	done
done
echo "Train"
find /sharedfiles/rvl_cdip/train/ -type f | wc -l
echo "Val"
find /sharedfiles/rvl_cdip/val/ -type f | wc -l
echo "Test"
find /sharedfiles/rvl_cdip/test/ -type f | wc -l



