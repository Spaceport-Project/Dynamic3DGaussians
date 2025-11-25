   #!/bin/bash
   output_folder_path=$1
   colmap_input="$output_folder_path/colmap_input_fg"
    if [ ! -d "$colmap_input" ];then
        mkdir -p "$colmap_input/input"
    fi
     a=0
    ims_path="$output_folder_path/${folder_cam}/"
    for file in `find $ims_path   -name "000000.png" | sort -V`;
    do
            cp   $file  $colmap_input/input/$a.png
            a=$((a+1))
    done