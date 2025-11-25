#!/bin/bash
a=0
# temp_folder="./tmp_006"
# output_folder_path="/media/hamit/Elements/05-02-2025_Data/processed_data/2025-02-05_14-16-46_gain_9/"
#  for file in `find $output_folder_path/ims/ -type f -name "000006.png" | sort -V`;
#         do
#                 ln -s $file $temp_folder/$a.png
#                 a=$((a+1))
#         done
ts=6
folder="/home/hamit/Dynamic3DGaussians/"
output_folder_path="/media/hamit/Elements/05-02-2025_Data/processed_data/2025-02-05_14-16-46_gain_9/"
 for file in `find $folder/masked_undistorted_images/  -name "*.png" | sort -V`;
        do 
            # python resize.py --input_image $file --output_image  $file  --dim $DIM
            #convert -resize $DIM $file $file
            
            temp_fol=$(basename "$file")
            IFS='_' read -ra array <<< "$temp_fol"
            fol=${array[0]}

            if [ ! -d "$output_folder_path/ims_black/$fol/" ];then
                mkdir -p "$output_folder_path/ims_black/$fol/"
            fi

            if [ ! -d "$output_folder_path/seg/$fol/" ];then
                mkdir -p "$output_folder_path/seg/$fol/" 
            fi



             formatted_number=$(printf "%06d" $ts)
             new_file_name=$formatted_number.png
            if [[ "$temp_fol" == *"_black.png"* ]]; then

                cp $file "$output_folder_path/ims_black/$fol/$new_file_name"
            elif [[ "$temp_fol" == *"_black_white.png"* ]]; then
                
                cp $file "$output_folder_path/seg/$fol/$new_file_name"

            fi

            

            
        done

