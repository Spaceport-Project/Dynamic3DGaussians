import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def main():
    # Initialize GStreamer
    Gst.init(None)

    # Create the pipeline
    pipeline = Gst.Pipeline.new("pipeline")

    # Create the elements
    filesrc = Gst.ElementFactory.make("filesrc", "source")
    filesrc.set_property("blocksize", 12288000)
    filesrc.set_property("location", "/home/hamit/Softwares/GrabberBaslerMultipleCameras/build/bayer8.bin")

    capsfilter1 = Gst.ElementFactory.make("capsfilter", "capsfilter1")
    caps1 = Gst.Caps.from_string("video/x-bayer,width=4096,height=3000,framerate=9/1,format=rggb")
    capsfilter1.set_property("caps", caps1)

    bayer2rgb = Gst.ElementFactory.make("bayer2rgb", "bayer2rgb")
    # nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
    # nvvideoconvert.set_property("interpolation-method",4)
    # nvvideoconvert.set_property("copy-hw",2)
    # nvvideoconvert.set_property("contiguous-buffers",1)
    # nvvideoconvert.set_property("nvbuf-memory-type",0)


    capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
    caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420")
    capsfilter2.set_property("caps", caps2)

    nvh264enc = Gst.ElementFactory.make("nvh264enc", "encoder")
    h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
    filesink = Gst.ElementFactory.make("filesink", "sink")
    filesink.set_property("location", "output2.mp4")

    # Add the elements to the pipeline
    pipeline.add(filesrc)
    pipeline.add(capsfilter1)
    pipeline.add(bayer2rgb)
    # pipeline.add(nvvideoconvert)
    # pipeline.add(capsfilter2)
    pipeline.add(nvh264enc)
    pipeline.add(h264parse)
    pipeline.add(filesink)

    # Link the elements
    if not filesrc.link(capsfilter1):
        print("ERROR: Could not link filesrc to capsfilter")
        return

    if not capsfilter1.link(bayer2rgb):
        print("ERROR: Could not link capsfilter to bayer2rgb")
        return

    if not bayer2rgb.link(nvh264enc):
        print("ERROR: Could not link bayer2rgb to nvvideoconvert")
        return
    # if not nvvideoconvert.link(capsfilter2):
    #     print("ERROR: Could not link nvvideoconvert to capsfilter")
    #     return
    # if not capsfilter2.link(nvh264enc):
    #     print("ERROR: Could not link nvvideoconvert to nvh264enc")
    #     return

    if not nvh264enc.link(h264parse):
        print("ERROR: Could not link nvh264enc to h264parse")
        return

    if not h264parse.link(filesink):
        print("ERROR: Could not link h264parse to filesink")
        return

    # Start the pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start the pipeline.")
        pipeline.set_state(Gst.State.NULL)
        return

    # Wait until the pipeline finishes
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

    # Stop the pipeline
    pipeline.set_state(Gst.State.NULL)

    # Clean up
    #   if msg:
    #       msg.unref()
    #   bus.unref()
    #   pipeline.unref()

if __name__ == "__main__":
  main()
