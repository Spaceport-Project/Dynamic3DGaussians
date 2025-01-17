#!/bin/bash
src_folder="S1"
number_timesteps=$(ls $src_folder/ims/0/*.png |wc -l)



# Number of concurrent jobs
max_jobs=5

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

number_timesteps=$((number_timesteps-1))
for id in $(seq 0 $number_timesteps);
do
	temp_folder="./tmps/tmp_$b"
    if [ -d "$temp_folder" ];then
        rm  -rf $temp_folder
    fi
    mkdir -p $temp_folder
    formatted_number=$(printf "%06d" $id)
    file_name=$formatted_number.png
    # echo "$file_name"
    wait_for_jobs

    a=0
    for file in `find $src_folder/ims/ -type f -name "$file_name" | sort -V`;
    do
            ln -s  $file $temp_folder/$a.png
            #cp $file seg_firstframes/$a.png
            a=$((a+1))
    done
    run_colmap_undistort $file_name $temp_folder &
    pids+=($!)
   
    
    

done


