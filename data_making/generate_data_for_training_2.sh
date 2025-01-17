#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --dataset_path <dataset_path> --output_folder_path <output_folder_path> [--help]"
    echo "  --dataset_path <dataset_path>   Specify the dataset_path"
    echo "  --output_folder_path <output_folder_path>     Specify the output_folder_path"
    echo "  --help          Display this help message"
    exit 1
}


# Default values
dataset_path=""
output_folder_path=""
DIM="4220x3060"
# Parse command-line arguments

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_path)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                dataset_path="$2"
                shift 2
            else
                echo "Error: --dataset_path requires a non-empty argument."
                usage
            fi
            ;;
        --output_folder_path)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                output_folder_path="$2"
                shift 2
            else
                echo "Error: --output_folder_path requires a non-empty argument."
                usage
            fi
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Invalid option $1"
            usage
            ;;
    esac
done


# Check if dataset_path and output_folder_path are provided
if [ -z "$dataset_path" ] || [ -z "$output_folder_path" ]; then
    echo "Both dataset_path and output_folder_path are required."
    usage
fi





# list_and_check_frames_dataset() {
    
#     are_all_occurences_equal() {
#         local arr=("$@")
#         local first="${arr[0]}"

#         for number in "${arr[@]}"; do
#             if [ "$number" -ne "$first" ]; then
#                 return 1  # Return 1 if any number is not equal to the first
#             fi
#         done
#         return 0  # Return 0 if all numbers are equal
#     }


#     local list_number_occurences=($(find $dataset_path  -type f -name '*.png' | awk '{n=split($1,A,"_"); print A[n-1]}' | sort | uniq -c |  awk '{print $1}'))
#     # echo "${list_number_occurences[@]}"  >&2 
#     # if are_all_occurences_equal "${list_number_occurences[@]}"; then
#     #     echo "All numbers are equal." >&2 
        
#     # else
#     #     echo "Not all numbers are equal. Exiting" >&2 
#     #     exit
        
#     # fi
#     local list_time_stamps=($(find $dataset_path  -type f -name '*.png' | awk '{n=split($1,A,"_"); print A[n-1]}' | sort | uniq ))

#     echo "${list_time_stamps[@]}"

# }

organize_images_folder_by_cam(){
    local first_timestamp=$(find $dataset_path  -type f -name "*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq -c | awk '$1 == 24 {print $2; exit}')
    local list_timestamps=($(find $dataset_path  -type f -name "*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq))
    # local first_timestamp=${list_timestamps[0]}
    local list_serial_numbers=($(find $dataset_path  -type f -name "*_${first_timestamp}_*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n]}' | cut -f 1 -d '.' ))
    # echo "${list_timestamps[@]}" >&2
    echo "${list_serial_numbers[@]}" >&2
    local id=0
    local exit_flage="false"
    for idx in in "${!list_timestamps[@]}"; do
        # echo "id:$id"
        iters=(${list_timestamps[$idx+1]} ${list_timestamps[$idx+2]} ${list_timestamps[$idx+3]} ${list_timestamps[$idx+4]} ${list_timestamps[$idx+5]} ${list_timestamps[$idx+6]})
        # iters=(${list_timestamps[$idx+1]} ${list_timestamps[$idx+2]})
     
        if [ "${list_timestamps[$idx]}" -lt "${first_timestamp}" ];then
                continue
        fi
        # echo "$id"  >&2
        if [ "$id" -gt "299" ];then
            break
        fi
    
        local a=0
        for sn in "${list_serial_numbers[@]}"; do
            if [ ! -d $output_folder_path/ims/$a ];then
                mkdir -p $output_folder_path/ims/$a
            fi
            
            local file="$dataset_path/frame_${list_timestamps[$idx]}_$sn.png"
            if [ -e "$file" ]; then 
                formatted_number=$(printf "%06d" $id)
               cp $file  $output_folder_path/ims/$a/$formatted_number.png
            else
                local kk=0
                for it in "${!iters[@]}"; do
                    local next_file="$dataset_path/frame_${iters[$it]}_$sn.png"
                    if [ -e "$next_file" ]; then 
                       echo "Copying $next_file to $output_folder_path/ims/$a/$formatted_number.png" >&2
                       cp $next_file  $output_folder_path/ims/$a/$formatted_number.png
                       break 
                    fi
                    kk=$((kk+1))
                    if [ "$kk" -eq "${#iters[@]}" ]; then
                        echo "Cannot find a image within next 6 time stamps. It is beacuse either you have reached the end of the list or something wrong with the recordings. Make sure your recording does not have that many missing images. Exiting!" >&2 
                        exit_flag="true"
                        break
                         
                    fi

                done
              

                

            fi

            if [[ "$exit_flag" == "true" ]];then
                break;
            fi

         
            a=$((a+1))
        
        done
        if [[ "$exit_flag" == "true" ]];then
                    break;
        fi
        echo "Copying images from $a cameras for ${list_timestamps[$idx]} timestamp has finished!" >&2
        
        id=$((id+1))
    done
    echo $id

}
run_colmap(){
    
    local colmap_input="$output_folder_path/colmap_input"
    if [ ! -d "$colmap_input/input" ];then
        mkdir -p "$colmap_input/input"
    fi
    local a=0
    local ims_path="$output_folder_path/ims/"
    for file in `find $ims_path -type f -name "000000.png" | sort -V`;
    do
            cp $file  $colmap_input/input/$a.png
            a=$((a+1))
    done

    python convert.py -s $colmap_input # --no_gpu
    # python fix_intrinsics_and_convert.py  -s $colmap_input --no_gpu -i /home/hamit/basler_camera_calibrations/calibrations_list/

    


}


run_colmap_undistort_frames() {

    # Number of concurrent jobs
    local max_jobs=5

    # Array to hold background process IDs

    declare -A pid_folder_dict

    # Function to wait for any job to finish
    wait_for_jobs() {
        # while [ "${#pids[@]}" -ge "$max_jobs" ]; do
        while [ "${#pid_folder_dict[@]}" -ge "$max_jobs" ]; do 
            for key in "${!pid_folder_dict[@]}"; do
            
                if ! kill -0 "$key" 2>/dev/null ; then
                
                    echo "Pid $key finished"
                    echo ${pid_folder_dict[${key}]}
                    rm  -rf ${pid_folder_dict[${key}]}
                    unset pid_folder_dict[$key]
                fi
            done
            sleep 3
        done
    }


    colmap_undistort () {
        local file_name=$1
        local folder=$2
        colmap image_undistorter --image_path $folder  --input_path $output_folder_path/colmap_input/distorted_sparse_aligned/ \
            --output_path $folder  --output_type COLMAP
        
        
        # formatted_number=$(printf "%04d" $id)
        echo "Moving $file_name files!"
        for file in `find $folder/images/  -name "*.png" | sort -V`;
        do 
            # python resize.py --input_image $file --output_image  $file  --dim $DIM
            #convert -resize $DIM $file $file
            fol=$(basename "$file" .png)
            if [ ! -d "$output_folder_path/ims/$fol/" ];then
                mkdir -p "$output_folder_path/ims/$fol/"
            fi
            
            mv  $file $output_folder_path/ims/$fol/$file_name

            
        done
    
    }


    local b=0
    number_timesteps=$((number_timesteps-1))
    for id in $(seq 0 $number_timesteps);
    do
        local temp_folder="./tmps/tmp_$b"
        if [ -d "$temp_folder" ];then
            rm  -rf $temp_folder
        fi
        mkdir -p $temp_folder
        local formatted_number=$(printf "%06d" $id)
        local file_name=$formatted_number.png
        # echo "$file_name"
        wait_for_jobs 

        local a=0
        for file in `find $output_folder_path/ims/ -type f -name "$file_name" | sort -V`;
        do
                ln -s $file $temp_folder/$a.png
                #cp $file seg_firstframes/$a.png
                a=$((a+1))
        done
        colmap_undistort $file_name $temp_folder &
        local pid=$!
        pid_folder_dict["$pid"]=$temp_folder
        b=$((b+1))
    
    done
    for key in "${!pid_folder_dict[@]}"; do
        wait "$key"
        rm -rf ${pid_folder_dict[${key}]}
    done

}



run_masking_command() {

    # Number of concurrent jobs for masking
    local max_jobs=3

    local pids=()
    # Array to hold background process IDs
    local masked_folder="masked_undistorted_ims_test2"

    eval "$(conda shell.bash hook)"
    conda init bash

    conda activate nerfstudio

    # Function to wait for any job to finish
    wait_for_jobs() {
        while [ "${#pids[@]}" -ge "$max_jobs" ]; do
            for k in "${!pids[@]}"; do 
                if ! kill -0 "${pids[k]}" 2>/dev/null ; then
                    # Remove finished job from the array
                    echo "Pid ${pids[k]} finished"
                    unset 'pids[k]'
                
                fi
            done
            sleep 3
        done
    }



    masking_command(){
        local folder=$1
        local output_path=$2

        
        python /home/hamit/Softwares/spaceport-tools/4dgs-scripts/generate_masked_data.py images -n yolox-x -c \
                /home/hamit/Softwares/YOLOX/model_weights/yolox_x.pth \
                --sam_checkpoint /home/hamit/Softwares/segment-anything/model_weights/sam_vit_h_4b8939.pth \
                --sam_model_type vit_h \
                --path  $folder/ \
                --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        
        # rm -rf $output_path
        mv -T  $folder/masked_undistorted_images $output_path
    }

    for folder_path  in `find $output_folder_path/ims -maxdepth 1 -mindepth 1 -type d  | sort -V`;
    do
        
        wait_for_jobs
        local parent_folder_name=$(basename "$folder_path")
        local output_path="$output_folder_path/$masked_folder/$parent_folder_name"
        if [ ! -d "$output_path" ];then
            mkdir -p $output_path
        fi
        masking_command "$folder_path" "$output_path" &
        pids+=($!)
    
    done
    
    # Wait for all remaining jobs to finish
    wait "${pids[@]}"

    for folder_path  in `find $output_folder_path/$masked_folder -maxdepth 1 -mindepth 1 -type d | sort -V`;
    do
        local parent_folder_name=$(basename "$folder_path")
        local output_path_ims="$output_folder_path/ims_black/$parent_folder_name"
        local output_path_seg="$output_folder_path/seg/$parent_folder_name"

        if [ ! -d "$output_path_ims" ];then
            mkdir -p $output_path_ims/
        fi
        if [ ! -d "$output_path_seg" ];then
            mkdir -p $output_path_seg/
        fi
        for file in `find $folder_path -maxdepth 1 -mindepth 1 -type f -name "*.png" | sort -V`;
        do
            local file_name=$(basename "$file")
            local directory_path=$(dirname "$file_path")
            local new_file_name=""
            if [[ "$file_name" == *"_black.png"* ]]; then

                new_file_name="${file_name/_black.png/.png}"
                mv $file $output_path_ims/$new_file_name
            elif [[ "$file_name" == *"_black_white.png"* ]]; then

                new_file_name="${file_name/_black_white.png/.png}"
                mv $file $output_path_seg/$new_file_name

            fi

            

        done
    done
        

    conda deactivate


}




# list_time_stamps=($(list_and_check_frames_dataset))
# organize_images_folder_by_cam
# number_timesteps=$(organize_images_folder_by_cam)
number_timesteps=300
echo "Number of time stamps : $number_timesteps"


# run_colmap

run_colmap_undistort_frames


run_masking_command



echo "All tasks are completed."
















