#!/bin/bash

eval "$(conda shell.bash hook)"
conda init bash

conda activate nerfstudio
src_folder="/home/hamit/Softwares/Dynamic3DGaussians/data/10-09-2024_data/pose_1_3/"

# Number of concurrent jobs
max_jobs=3

# Array to hold background process IDs
pids=()

# Function to wait for any job to finish
wait_for_jobs() {
    while [ "${#pids[@]}" -ge "$max_jobs" ]; do
        for k in "${!pids[@]}"; do 
            echo "Running pid list ${pids[@]}"
            if ! kill -0 "${pids[k]}" 2>/dev/null ; then
                # Remove finished job from the array
                echo "${pids[k]} Killed"
                unset 'pids[k]'
             
            fi
        done
        sleep 3
    done
}
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



for folder_path  in `find $src_folder/ims -maxdepth 1 -mindepth 1 -type d  | sort -V`;
do
    
    wait_for_jobs
    parent_folder_name=$(basename "$folder_path")
    output_path="$src_folder/masked_undistorted_ims/$parent_folder_name"
    if [ ! -d "$output_path" ];then
        mkdir -p $output_path
    fi
    run_masking_command "$folder_path" "$output_path" &
    pids+=($!)
done

# Wait for all remaining jobs to finish
wait "${pids[@]}"

echo "All tasks are completed."

conda deactivate
