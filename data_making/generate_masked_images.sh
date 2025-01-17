#!/bin/bash

eval "$(conda shell.bash hook)"
conda init bash
# dataset_path="/mnt/Elements2/10-09-2024_data/pose_1_3_4096x3000/dataset_18cams/"
src_folder="/home/hamit/Softwares/Dynamic3DGaussians/data/10-09-2024_data/pose_1_3/"

run_masking_command(){
    local folder=$1
    local output_path=$2
    python /home/hamit/Softwares/spaceport-tools/4dgs-scripts/generate_masked_data.py images -n yolox-x -c \
            /home/hamit/Softwares/YOLOX/model_weights/yolox_x.pth \
            --sam_checkpoint /home/hamit/Softwares/segment-anything/model_weights/sam_vit_h_4b8939.pth \
            --sam_model_type vit_h \
            --path  $folder/ \
            --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
    mv -T  $folder/masked_undistorted_images $output_path
}



# if [ ! -d "$src_folder/masked_undistorted_images/" ];then
#     mkdir -p $src_folder/masked_undistorted_images/
# fi

conda activate nerfstudio

b=0
for folder_path  in `find $src_folder/ims -maxdepth 1 -mindepth 1 -type d  | sort -V`;
do
    
    
	
    parent_folder_name=$(basename "$folder_path")
    output_path="$src_folder/masked_undistorted_images/$parent_folder_name"
    if [ ! -d "$output_path" ];then
        mkdir -p $output_path
    fi
    run_masking_command $folder_path $output_path &
    
    b=$((b+1))
    if [ "$b" -eq 3 ];then
        echo "Waiting for masking functions to be done!"
        wait
        b=0
        
    fi

	
	# a=$((a+1))
done
conda deactivate
