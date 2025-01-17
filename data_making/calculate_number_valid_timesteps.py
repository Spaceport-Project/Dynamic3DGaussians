import os
import glob
import shutil
from typing import List
import sys  # Make sure to import sys for stderr output

def calculate_number_valid_timesteps(dataset_path: str, cam_number: int) -> List[str]:
    # Initialize empty list for valid timestamps
    temp_list = []

    # Get all PNG files and extract timestamps
    png_files = glob.glob(os.path.join(dataset_path, "*.png"))
    png_files.sort(key=lambda x: int(x.split('_')[-2]))
    png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[-2]))

    # Extract timestamps from filenames
    def get_timestamp(filename):
        return int(filename.split('_')[-2]) # Assuming timestamp is second-to-last part

    # Get all unique timestamps
    all_timestamps = sorted(set(get_timestamp(f) for f in png_files))

    # Find first timestamp that has correct number of cameras
    first_timestamp = None
    for timestamp in all_timestamps:
        matching_files = len(glob.glob(os.path.join(dataset_path, f"*{timestamp}*.png")))
        if matching_files == cam_number:
            first_timestamp = timestamp
            break

    if not first_timestamp:
        return []

    # Process timestamps
    id = 0
    for timestamp in all_timestamps:
        # Skip timestamps before first_timestamp
        if int(timestamp) < int(first_timestamp):
            continue

        # Count number of images for current timestamp
        current_cam_number = len(glob.glob(os.path.join(dataset_path, f"*{timestamp}*.png")))
        cam_number_minus_one = cam_number - 1

        # Check if we have enough images for this timestamp
        if current_cam_number < cam_number_minus_one:
            print(f"There must be at least {cam_number_minus_one} images out of {cam_number} "
                  f"for a timestamp, but there exist {current_cam_number} images for the "
                  f"following time stamp '{timestamp}'.\n"
                  f"Exiting from the function of calculating number of valid timesteps!",
                  file=sys.stderr)
            break

        # Break if we've processed 300 timestamps
        if id >= 300:
            break

        temp_list.append(timestamp)
        id += 1

    return temp_list

def organize_images_folder_by_cam(list_valid_ts:List[int], dataset_path: str, output_folder_path: str, cam_number: int):
    # Find first timestamp with correct number of cameras
    first_timestamp=list_valid_ts[0]
    png_files = glob.glob(os.path.join(dataset_path, "*.png"))
    png_files.sort()

  

    def get_serial_number(filename):
        return filename.split('_')[-1].split('.')[0]

    # Get list of serial numbers from first timestamp
    first_ts_files = glob.glob(os.path.join(dataset_path, f"*_{first_timestamp}_*.png"))
    list_serial_numbers = sorted([get_serial_number(f) for f in first_ts_files])

    exit_flag = False
    id = 0

    for idx in range(len(list_valid_ts)):
        # Get next 6 timestamps for potential replacements
        iters = list_valid_ts[idx+1:idx+7]

        for a, sn in enumerate(list_serial_numbers):
            # Create output directory if it doesn't exist
            cam_output_dir = os.path.join(output_folder_path, 'ims', str(a))
            os.makedirs(cam_output_dir, exist_ok=True)

            # Try to copy current timestamp image
            current_file = os.path.join(dataset_path, f"frame_{list_valid_ts[idx]}_{sn}.png")
            formatted_number = f"{id:06d}"
            output_file = os.path.join(cam_output_dir, f"{formatted_number}.png")

            if os.path.exists(current_file):
                shutil.copy2(current_file, output_file)
            else:
                # Try next 6 timestamps
                found_replacement = False
                for next_ts in iters:
                    next_file = os.path.join(dataset_path, f"frame_{next_ts}_{sn}.png")
                    if os.path.exists(next_file):
                        print(f"Copying {next_file} to {output_file}", file=sys.stderr)
                        shutil.copy2(next_file, output_file)
                        found_replacement = True
                        break

                if not found_replacement:
                    print("Cannot find a image within next 6 time stamps. "
                          "It is because either you have reached the end of the list "
                          "or something wrong with the recordings. "
                          "Make sure your recording does not have that many missing images. "
                          "Exiting!", file=sys.stderr)
                    exit_flag = True
                    break

            if exit_flag:
                break

        if exit_flag:
            break

        print(f"Copying images from {a+1} cameras for {list_valid_ts[idx]} timestamp has finished!", 
              file=sys.stderr)
        id += 1


if __name__ == "__main__":
    # Example usage
    dataset_path = "/mnt/Elements2/19-12-2024_Data/2024-12-19_19-12-14_4096/all_frames_300_650/"
    output_folder = "/home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_300-600_calib"
    cam_number = 24
    valid_timestamps = calculate_number_valid_timesteps(dataset_path, cam_number)
    organize_images_folder_by_cam(valid_timestamps, dataset_path, output_folder, cam_number)
    print(" ".join(valid_timestamps))
   