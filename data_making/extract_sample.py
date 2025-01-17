import ffmpeg
import os
def extract_frames_with_pts(cam_file_path, output_path, cam_serial):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

   # Get frame information using ffprobe
    # probe = ffmpeg.probe(cam_file_path, select_streams='v:0', show_frames=True)
    # frames_data = probe['frames']
    probe = ffmpeg.probe(
        cam_file_path,
        v='quiet',
        select_streams='v:0',
        show_frames=None,
        of='json'
    )  
    frames_data = probe['frames']  # List of frame information  

    

    try:
        # Extract frames using ffmpeg-python with chained operations
        (
            ffmpeg
            .input(cam_file_path)
            .filter('select', 'between(n,0,350)')
            .output(f'{output_path}/frame_%03d_{cam_serial}.png', 
                   vsync='0',
                   start_number=0)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

    # Rename files with PTS
    for i, frame in enumerate(frames_data, 0):
        pts_time = frame["pkt_pts"]
        old_name = f'{output_path}/frame_{i:03}_{cam_serial}.png'
        new_name = f'{output_path}/frame_{pts_time}_{cam_serial}.png'
        print(old_name, new_name)

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
# Example usage
if __name__ == "__main__":
    cam_file_path = "/media/hamit/HamitsKingston/19-12-2024_Data/2024-12-19_19-12-14_4096/Cam-Full_40455112.mp4"
    output_path = "/media/hamit/HamitsKingston/19-12-2024_Data/cam_112"
    cam_serial = "40455112"

    extract_frames_with_pts(cam_file_path, output_path, cam_serial)