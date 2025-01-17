#!/bin/bash
src_folder="/home/hamit/Softwares/Dynamic3DGaussians/data/10-09-2024_data/pose_1_3"
# output_undistorted_folder="/home/hamit/DATA/oguz_2_22-08-2024/processed_data/undistorted_images/"
# if [ ! -d "$output_undistorted_folder" ];then
#     mkdir -p output_undistorted_folder
# fi



for id in {0..100};
do
	if [ -d "./tmp" ];then
        rm  -rf ./tmp
    fi
    mkdir ./tmp
    cd ./tmp
    formatted_number=$(printf "%06d" $id)
    file_name=$formatted_number.png
    # echo "$file_name"
    a=0
    for file in `find $src_folder/ims -type f -name "$file_name" | sort -V`;
    do
            ln -s  $file ./$a.png
            #cp $file seg_firstframes/$a.png
            a=$((a+1))
    done
    colmap image_undistorter --image_path ./  --input_path $src_folder/colmap_input/distorted/sparse/0/ \
         --output_path ./  --output_type COLMAP
    


    echo "Moving $file_name files!"
    for file in `find ./images/  -name "*.png" | sort -V`;
    do 
        fol=$(basename "$file" .png)

        mv  $file $src_folder/ims/$fol/$file_name

        
    done
    
    
    cd ../ 

done




# for file in `find ./seg -type f -name "000000.png" | sort -V`;
# do
#         cp $file seg_firstframes/$a.png
#         a=$((a+1))
# done


 
# colmap image_undistorter        --image_path /home/hamit/DATA/oguz_2_22-08-2024/processed_data/colmap_input/seg        --input_path /home/hamit/DATA/oguz_2_22-08-2024/processed_data/colmap_input/distorted/sparse/        --output_path /home/hamit/DATA/oguz_2_22-08-2024/processed_data/colmap_input/seg/undistorted        --output_type COLMAP