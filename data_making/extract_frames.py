import subprocess
import json

def extract_frames_with_pts():
    # First, get frame information
    cmd = ['ffprobe',
           '-v', 'quiet',
           '-select_streams', 'v:0',
           '-show_frames',
           '-of', 'json',
           'Cam_40455120.mp4']

    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    
    frames_data = json.loads(result.stdout)

    # Now extract frames and rename them with their PTS
    subprocess.run(['ffmpeg', '-i', 'Cam_40455120.mp4', 
                   '-vsync', '0', 
                   'frame_%d.png'])

    # Rename files with PTS
    for i, frame in enumerate(frames_data['frames'], 1):
        pts_time = frame.get('pkt_pts', str(i))
        old_name = f'frame_{i}.png'
        new_name = f'frames_120/frame_{pts_time}_40455128.png'
        subprocess.run(['mv', old_name, new_name])

# Run the function
extract_frames_with_pts()
