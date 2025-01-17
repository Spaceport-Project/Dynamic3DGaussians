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
DIM="4096x2950"
cam_number=24

# declare -a 
# list_valid_time_stamps=()
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





calculate_number_valid_timesteps() {
    local temp_list=()
    local first_timestamp=$(find $dataset_path  -type f -name "*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq -c | awk -v cam_number="${cam_number}" '$1 == cam_number {print $2; exit}')
    local list_timestamps=($(find $dataset_path  -type f -name "*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq))
    local id=0
    for idx in  "${!list_timestamps[@]}"; do
        if [ "${list_timestamps[$idx]}" -lt "${first_timestamp}" ];then
                continue
        fi
      
        current_cam_number=$(find $dataset_path -maxdepth 1 -type f -name "*${list_timestamps[$idx]}*.png" |wc -l)
        # echo $current_cam_number >&2
        cam_number_minus_one=$((cam_number-1))

        if [ "${current_cam_number}" -lt $cam_number_minus_one ]; then
            echo -e "There must be at least $cam_number_minus_one images out of $cam_number for a timestamp, but there exist ${current_cam_number} images for the following time stamp '${list_timestamps[$idx]}'."\
            "\nExiting from the function of calculating number of valid timesteps!" >&2
            break
        fi


        if [ "$id" -ge 300 ];then
            break
        fi
        
        temp_list+=("${list_timestamps[$idx]}")
        id=$((id+1))
    done
    echo "${temp_list[@]}" 


}

organize_images_folder_by_cam(){
    local list_valid_ts=($@)
    local first_timestamp=$(find $dataset_path  -type f -name "*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq -c | awk -v cam_number="${cam_number}" '$1 == cam_number {print $2; exit}')
    local list_serial_numbers=($(find $dataset_path  -type f -name "*_${first_timestamp}_*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n]}' | cut -f 1 -d '.' ))
    
    local exit_flage="false"
    local id=0
    for idx in  "${!list_valid_ts[@]}"; do
        local iters=(${list_valid_ts[$idx+1]} ${list_valid_ts[$idx+2]} ${list_valid_ts[$idx+3]} ${list_valid_ts[$idx+4]} ${list_valid_ts[$idx+5]} ${list_valid_ts[$idx+6]})

        local a=0
        # echo ${list_serial_numbers[@]}
        for sn in "${list_serial_numbers[@]}"; do
            if [ ! -d $output_folder_path/ims/$a ];then
                mkdir -p $output_folder_path/ims/$a
            fi
            
            local file="$dataset_path/frame_${list_valid_ts[$idx]}_$sn.png"
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
        echo "Copying images from $a cameras for ${list_valid_ts[$idx]} timestamp has finished!" >&2
        id=$((id+1))
    done

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

    python convert.py -s $colmap_input  --no_gpu
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
        
        # return
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
    number_timesteps_minus_one=$((number_timesteps-1))
    for id in $(seq 0 $number_timesteps_minus_one);
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

    declare -A pid_folder_dict

    # Array to hold background process IDs
    local masked_folder="masked_undistorted_temp_folder"

    eval "$(conda shell.bash hook)"
    conda init bash

    conda activate dynamic3gs_38

    # Function to wait for any job to finish
    # wait_for_jobs() {
    #     while [ "${#pids[@]}" -ge "$max_jobs" ]; do
    #         for k in "${!pids[@]}"; do 
    #             if ! kill -0 "${pids[k]}" 2>/dev/null ; then
    #                 # Remove finished job from the array
    #                 echo "Pid ${pids[k]} finished"
    #                 unset 'pids[k]'
                
    #             fi
    #         done
    #         sleep 3
    #     done
    # }

    # Function to wait for any job to finish
    wait_for_jobs() {
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



    masking_command(){
        local folder=$1
        local ts=$2

        
        python /home/hamit/Softwares/spaceport-tools/4dgs-scripts/generate_window_masked_data.py images -n yolox-x -c \
                /home/hamit/Softwares/YOLOX/model_weights/yolox_x.pth \
                --sam_checkpoint /home/hamit/Softwares/segment-anything/model_weights/sam_vit_h_4b8939.pth \
                --sam_model_type vit_h \
                --path  $folder/ \
                --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        
       

        echo "Moving masked ${ts}. time step files!"
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



            local formatted_number=$(printf "%06d" $ts)
            local new_file_name=$formatted_number.png
            if [[ "$temp_fol" == *"_black.png"* ]]; then

                mv $file "$output_folder_path/ims_black/$fol/$new_file_name"
            elif [[ "$temp_fol" == *"_black_white.png"* ]]; then
                
                mv $file "$output_folder_path/seg/$fol/$new_file_name"

            fi

            

            
        done

    }

    local b=0
    number_timesteps_minus_one=$((number_timesteps-1))
    for id in $(seq 0 $number_timesteps_minus_one);
    do
        local temp_folder="./tmps/tmp_$b"
        if [ -d "$temp_folder" ];then
            rm  -rf $temp_folder
        fi
        mkdir -p $temp_folder
        local formatted_number=$(printf "%06d" $id)
        local file_name=$formatted_number.png
        
        wait_for_jobs

        local a=0
        for file in `find $output_folder_path/ims/ -type f -name "$file_name" | sort -V`;
        do
                ln -s $file $temp_folder/$a.png
                a=$((a+1))
        done
        sleep 5
        masking_command "$temp_folder" $b &
        local pid=$!
        pid_folder_dict["$pid"]=$temp_folder
        b=$((b+1))
    done



    for key in "${!pid_folder_dict[@]}"; do
        wait "$key"
        rm -rf ${pid_folder_dict[${key}]}
    done

  

    conda deactivate


}





# list_valid_time_stamps=($(calculate_number_valid_timesteps))
# # echo "${list_valid_time_stamps[@]}" 
# number_timesteps=${#list_valid_time_stamps[@]} 
number_timesteps=300
# echo "Number of time stamps : $number_timesteps"
# sleep 2
# # echo "list_valid_time_stamps: ${list_valid_time_stamps[@]}"
# organize_images_folder_by_cam ${list_valid_time_stamps[@]} &



# sleep 30

# echo "Starting colmap"
# run_colmap



# echo "Starting undistorting frames"
# run_colmap_undistort_frames


echo "Starting masking frames"
run_masking_command



echo "All tasks are completed."
















