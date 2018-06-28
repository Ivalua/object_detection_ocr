#!/bin/bash
echo "Removing models from /sharedfiles"
rm /sharedfiles/models/*

echo "Removing datasets from /sharedfiles"
rm /sharedfiles/datasets/*

read -p "Remove TF logs? [y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing TF logs"
    rm -r ./Graph/*
fi
