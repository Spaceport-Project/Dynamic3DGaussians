#!/bin/bash

folder=$1

for folder in `find $folder -type d -mindepth 1 -maxdepth 1 | sort -V`;
do
    python remove_purplish_tint.py --path $folder &
    echo "python remove_purplish_tint.py --path $folder "
done
