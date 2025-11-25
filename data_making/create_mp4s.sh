#!/bin/bash

folder=$1

for fol in `find $folder -mindepth 1 -maxdepth  1  -type d | sort -V`;
do
    cd $fol
    last_folder=$(basename "$fol")  

    ffmpeg -y -framerate 30 -i %06d.png -vf "scale=2124:1528"  -c:v libx264 -pix_fmt yuv420p $last_folder.mp4 &
    echo "ffmpeg -framerate 30 -i %06d.png vf scale=2124:1528 -c:v libx264 -pix_fmt yuv420p $last_folder.mp4"

    
done


