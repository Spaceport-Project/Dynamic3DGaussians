#!/bin/bash
src_folder="/home/hamit/Softwares/Dynamic3DGaussians/data/10-09-2024_data/pose_1_3"
number_timesteps=$(ls $src_folder/images/cam00/images/*.png |wc -l)
# output_undistorted_folder="/home/hamit/DATA/oguz_2_22-08-2024/processed_data/undistorted_images/"
# if [ ! -d "$output_undistorted_folder" ];then
#     mkdir -p output_undistorted_folder
# fi

run_colmap_undistort () {
    local file_name=$1
    local folder=$2
    colmap image_undistorter --image_path $folder  --input_path $src_folder/colmap_input/distorted/sparse/0/ \
         --output_path $folder  --output_type COLMAP
    
    # formatted_number=$(printf "%04d" $id)
    echo "Moving $file_name files!"
    for file in `find $folder/images/  -name "*.png" | sort -V`;
    do 
        fol=$(basename "$file" .png)
        if [ ! -d "$src_folder/ims/$fol/" ];then
            mkdir -p "$src_folder/ims/$fol/"
        fi
        
        mv  $file $src_folder/ims/$fol/$file_name

        
    done
    
}

b=0
number_timesteps=$((number_timesteps-1))
for id in $(seq 0 $number_timesteps);
do
	temp_folder="./tmps/tmp_$b"
    if [ -d "$temp_folder" ];then
        rm  -rf $temp_folder
    fi
    mkdir -p $temp_folder
    # cd $temp_folder
    formatted_number=$(printf "%06d" $id)
    file_name=$formatted_number.png
    # echo "$file_name"
    a=0
    for file in `find $src_folder/images -type f -name "$file_name" | sort -V`;
    do
            ln -s  $file $temp_folder/$a.png
            #cp $file seg_firstframes/$a.png
            a=$((a+1))
    done
    run_colmap_undistort $file_name $temp_folder &
    b=$((b+1))
    echo "b value $b"
    if [ "$b" -eq 5 ];then
        echo "Waiting for colmap undistort functions to be done!"
        wait
        b=0
        
    fi
    
    

done


