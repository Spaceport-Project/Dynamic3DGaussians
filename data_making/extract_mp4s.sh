#!/bin/bash

folder=$1

for file in `find $folder -mindepth 1 -maxdepth 1  -type f -name "*.mp4" | sort -V`;
do 
    
    filename=$(basename "$file") 
    name="${filename%.*}" 

    echo $name 
    mkdir  ${folder}/${name}
    ffmpeg -i $file -vsync 0 -q:v 1 -qmin 1 -qmax 1  -start_number 0  ${folder}${name}/%06d.png 
    echo "ffmpeg -i $file -vsync 0 -q:v 1 -qmin 1 -qmax 1 ${folder}${name}/%06d.png"
    
done  