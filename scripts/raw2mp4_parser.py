import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def main():
  # Initialize GStreamer
  Gst.init(None)

  # Define the pipeline description
  pipeline_description = (
      "filesrc blocksize=12288000 location=/home/hamit/Softwares/GrabberBaslerMultipleCameras/build/bayer8.bin ! "
      "video/x-bayer,width=4096,height=3000,framerate=9/1,format=rggb ! "
      "bayer2rgb ! nvh264enc bitrate=5000 ! h264parse ! "
      "filesink location=output2.mp4"
  )

  # Create the pipeline using parse_launch
  pipeline = Gst.parse_launch(pipeline_description)

  # Start the pipeline
  ret = pipeline.set_state(Gst.State.PLAYING)
  if ret == Gst.StateChangeReturn.FAILURE:
      print("Failed to start the pipeline.")
      pipeline.set_state(Gst.State.NULL)
      return

  # Wait until the pipeline finishes
  bus = pipeline.get_bus()
  msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

  # Handle messages
  if msg:
      if msg.type == Gst.MessageType.ERROR:
          err, debug = msg.parse_error()
          print(f"Error: {err}, {debug}")
      elif msg.type == Gst.MessageType.EOS:
          print("End-Of-Stream reached")

  # Stop the pipeline
  pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
  main()