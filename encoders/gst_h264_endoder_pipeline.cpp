#include <cstdio>
#include <cuda_runtime.h>
#include <functional> 

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudacontext.h>
#include <gst/cuda/gstcudamemory.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <pybind11/pybind11.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <pybind11/numpy.h>


namespace py = pybind11;



struct App {
    GMainLoop* loop=NULL;
    GstAppSrc* video_appsrc=NULL;
    GstElement *video_appsink;
    GstAppSrc* audio_appsrc=NULL;
    GstElement *audio_appsink;    

    GstCudaContext* context=NULL;
    GstElement* pipeline =NULL;
    unsigned int width ;
    unsigned int height ;
    unsigned int frame_rate ;

    
    std::mutex video_queue_mutex;
    std::condition_variable video_queue_cv;
    std::queue<GstBuffer*> video_buffer_queue;

    std::mutex audio_queue_mutex;
    std::condition_variable audio_queue_cv;
    std::queue<GstBuffer*> audio_buffer_queue;

    bool exit_flag = false;
 
};
std::mutex cudaMutex;


// App app;
std::vector<App*> apps;


gboolean push_cuda_tensor_frame( uintptr_t cuda_ptr, size_t  size, unsigned int frame_number, unsigned int frame_rate, int cnt) {

 


    GstCaps* caps = gst_caps_from_string("video/x-raw(memory:CUDAMemory),format=RGB,width=1920,height=1080,framerate=30/1");
    GstVideoInfo video_info;
    gst_video_info_from_caps(&video_info, caps);
    // std::cout<<"ptr:"<<(CUdeviceptr)gpu_ptr<<std::endl;
    gst_cuda_context_push(apps[0]->context);

    {
       
        GstBuffer* buf = gst_buffer_new();
        
        GstMemory* memory = gst_cuda_allocator_alloc(NULL, apps[0]->context, NULL, &video_info);
        if (memory != NULL && buf != NULL) {
        // Map the memory to access it
            GstMapInfo map;
            if (gst_memory_map(memory, &map, GST_MAP_WRITE)) {
                // Fill the memory with data (example: fill with zeros)
                cudaMemcpy(map.data, (void *)cuda_ptr, size, cudaMemcpyDeviceToDevice);
               
                gst_memory_unmap(memory, &map);
            // Step 4: Append the GstMemory to the GstBuffer
                gst_buffer_append_memory(buf, memory);
            } else {
                // Handle mapping error
                g_printerr("Failed to map memory.\n");
            }
        }




        GstClockTime duration = GST_SECOND / frame_rate;
        GST_BUFFER_PTS(buf) = frame_number * duration;
        GST_BUFFER_DURATION(buf) = duration;
        GstFlowReturn ret;
        g_signal_emit_by_name(apps[cnt]->video_appsrc, "push-buffer", buf, &ret);

        if (ret != GST_FLOW_OK) {
            std::cerr << "Error pushing buffer to appsrc" << std::endl;
            return FALSE;
        }
        gst_buffer_unref(buf);
    }
    gpointer ctx = gst_cuda_context_get_handle (apps[0]->context);
    gst_cuda_context_pop((CUctx_st**)ctx);
    
   
    return TRUE;
}

void push_tensor_frame(uintptr_t cuda_ptr, size_t  size, unsigned int frame_number, unsigned int frame_rate, int cnt) {
    // Create a GstBuffer that wraps the existing image data
    
    uint8_t *data_ptr = (uint8_t *)malloc(size);
    cudaMemcpy(data_ptr, (uint8_t *)cuda_ptr, size, cudaMemcpyDeviceToHost);
    // std::cout<<"push frame 0"<<std::endl;
    
    GstBuffer *buffer = gst_buffer_new_wrapped((uint8_t *)data_ptr, size);

    GstClockTime duration = GST_SECOND / frame_rate;
    GST_BUFFER_PTS(buffer) = frame_number; //* duration;
    GST_BUFFER_DURATION(buffer) = duration;
    
    // Create a GstFlowReturn variable to hold the flow return value

    GstFlowReturn ret = GST_FLOW_OK;

    // Push the buffer into the appsrc element
    g_signal_emit_by_name(apps[cnt]->video_appsrc, "push-buffer", buffer, &ret);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Error pushing buffer to appsrc: " << ret << std::endl;
    }

    // Unreference the buffer as it is now owned by GStreamer
    gst_buffer_unref(buffer);
}
void push_audio_packet(uintptr_t data_ptr, size_t  size, unsigned int frame_number, unsigned int frame_rate, int cnt) {

    GstBuffer *buffer = gst_buffer_new_wrapped((uint8_t *)data_ptr, size);


    GstClockTime duration = GST_SECOND / frame_rate;
    GST_BUFFER_PTS(buffer) = frame_number * duration;
    GST_BUFFER_DURATION(buffer) = duration;
    
    GstFlowReturn ret = GST_FLOW_OK;

    // Push the buffer into the appsrc element
    g_signal_emit_by_name(apps[cnt]->audio_appsrc, "push-buffer", buffer, &ret);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Error pushing buffer to appsrc: " << ret << std::endl;
    }
    gst_buffer_unref(buffer);


}

// Callback for bus messages
static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    GMainLoop* loop = (GMainLoop*)data;

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar* debug;
            GError* error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("Error: %s\n", error->message);
            g_error_free(error);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }

    return TRUE;
}

static GstFlowReturn on_new_audio_sample(GstElement *sink, gpointer data) {
    int cnt = GPOINTER_TO_INT(data);

    GstSample *sample = nullptr;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (sample) {
        GstBuffer *buffer = gst_sample_get_buffer(sample);

        if (buffer) {
            GstBuffer *buffer_copy = gst_buffer_ref(buffer);
            {
                std::lock_guard<std::mutex> lock( apps[cnt]->audio_queue_mutex);
                apps[cnt]->audio_buffer_queue.push(buffer_copy);
            }
            apps[cnt]->audio_queue_cv.notify_one();    

        }
        gst_sample_unref(sample);

    }

    return GST_FLOW_OK;
}


static GstFlowReturn on_new_video_sample(GstElement *sink, gpointer data) {

    int cnt = GPOINTER_TO_INT(data);

    GstSample *sample = nullptr;
    g_signal_emit_by_name(sink, "pull-sample", &sample);

    if (sample) {
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        if (buffer) {
           
            GstBuffer *buffer_copy = gst_buffer_ref(buffer);
            {
                std::lock_guard<std::mutex> lock( apps[cnt]->video_queue_mutex);
                apps[cnt]->video_buffer_queue.push(buffer_copy);
            }
            apps[cnt]->video_queue_cv.notify_one();
            
            // std::cout << "Received buffer of size: " << gst_buffer_get_size(buffer) << " "<<cnt<<std::endl;
            // cnt++;
            // gst_buffer_unref(buffer_copy);
        }
       
        // gst_buffer_unref(buffer);

        gst_sample_unref(sample);
    }
    return GST_FLOW_OK;
}

std::pair<int, uintptr_t> get_next_video_buffer_data(int cnt) {


    GstBuffer *buffer = nullptr;

    {
        std::unique_lock<std::mutex> lock( apps[cnt]->video_queue_mutex);
        apps[cnt]->video_queue_cv.wait(lock, [cnt] { 
            return !apps[cnt]->video_buffer_queue.empty() ||  apps[cnt]->exit_flag; });
       
        if ( apps[cnt]->exit_flag)
            return {0,0};
        buffer = apps[cnt]->video_buffer_queue.front();
        apps[cnt]->video_buffer_queue.pop();
        
    }   

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_buffer_unref(buffer);
        throw std::runtime_error("Failed to map buffer.");
    }



    gst_buffer_unmap(buffer, &map);
    gst_buffer_unref(buffer);

    return {map.size, (uintptr_t)map.data};
}

std::pair<int, uintptr_t> get_next_audio_buffer_data(int cnt) {


    GstBuffer *buffer = nullptr;
    std::cout<<"audio getting data 0 :"<<cnt<<std::endl;
    return {0, 0};

    {
        std::unique_lock<std::mutex> lock( apps[cnt]->audio_queue_mutex);
        apps[cnt]->audio_queue_cv.wait(lock, [cnt] { 
            return !apps[cnt]->audio_buffer_queue.empty() ||  apps[cnt]->exit_flag; });
       
        if ( apps[cnt]->exit_flag)
            return {0,0};
        buffer = apps[cnt]->audio_buffer_queue.front();
        apps[cnt]->audio_buffer_queue.pop();
        
    }   

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_buffer_unref(buffer);
        throw std::runtime_error("Failed to map buffer.");
    }



    gst_buffer_unmap(buffer, &map);
    gst_buffer_unref(buffer);

    return {map.size, (uintptr_t)map.data};
}



int main_fun() {
    
    static int i = 0;
    int cnt = i;

    apps.push_back(new (App){ .width=1920, .height=1080, .frame_rate=30});
    
    std::cout<<"Number of Clients connected:"<<cnt<<std::endl;
    // int argc = 0 ; 
    // char **argv = nullptr;
    // gst_init(&argc, &argv);

    apps[cnt]->loop = g_main_loop_new(NULL, FALSE);
    // Create pipeline
    // app.pipeline = gst_parse_launch(
    //     "appsrc name=source do-timestamp=true format=TIME ! "
    //     "video/x-raw(memory:CUDAMemory),format=RGB,width=1920,height=1080,framerate=30/1 ! " 
    //     " cudaconvert ! video/x-raw(memory:CUDAMemory),format=NV12 !  nvh264enc bitrate=5000 zerolatency=true  ! "
    //     " video/x-h264,stream-format=avc,alignment=au ! "
    //     " h264parse   ! "
    //     //  ! video/x-h264,stream-format=byte-stream,alignment=au  ! "
    //     //  config-interval=-1  ! video/x-h264,stream-format=byte-stream,alignment=au !"
    //     // "queue ! "
    //     "appsink name=sink sync=false",
    //     // " mp4mux ! "
    //     // " filesink location=output.mp4",
    //     // 
    //     NULL);
    // apps[cnt]->pipeline = gst_parse_launch(
    //     "appsrc name=source do-timestamp=true format=TIME ! "
    //     "video/x-raw(memory:CUDAMemory),format=RGB,width=1920,height=1080,framerate=30/1 ! " 
    //     " cudaconvert ! video/x-raw(memory:CUDAMemory),format=NV12 !  nvh264enc bitrate=5000 zerolatency=true  ! "
    //     " video/x-h264,stream-format=avc,alignment=au ! "
    //     " h264parse   ! "
    //     //  ! video/x-h264,stream-format=byte-stream,alignment=au  ! "
    //     //  config-interval=-1  ! video/x-h264,stream-format=byte-stream,alignment=au !"
    //     // "queue ! "
    //     "appsink name=sink sync=false",
    //     // " mp4mux ! "
    //     // " filesink location=output.mp4",
    //     // 
    // NULL);
    


    apps[cnt]->pipeline = gst_parse_launch(
        "appsrc name=video_source do-timestamp=true format=TIME ! "
        "video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1 ! " 
        " videoconvert ! video/x-raw,format=NV12 ! nvh264enc gop-size=50 bitrate=1500 !  "
        " video/x-h264,stream-format=avc,alignment=au ! "
        " h264parse   ! "
        // "  video/x-h264,stream-format=avc,alignment=au  ! "
        //  config-interval=-1  ! video/x-h264,stream-format=byte-stream,alignment=au !"
        // "queue ! "
        "appsink name=video_sink sync=false",
        // " mp4mux ! "
        // " filesink location=output.mp4",
        // 
    NULL);

    //  apps[cnt]->pipeline = gst_parse_launch(
    //     "appsrc name=video_source do-timestamp=true format=TIME ! "
    //     "video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1 ! "
    //     "videoconvert ! video/x-raw,format=NV12 ! nvh264enc bitrate=5000 ! "
    //     "h264parse ! "
    //     "queue ! appsink name=video_sink "

    //     "appsrc name=audio_source do-timestamp=true format=TIME ! "
    //     "audio/x-raw,format=S16LE,channels=2,rate=44100 ! "
    //      "audioconvert ! "
    //     //  audioresample ! avenc_aac bitrate=128000 ! aacparse ! "
    //     "queue ! appsink name=audio_sink",
    //     NULL);

    //  apps[cnt]->pipeline = gst_parse_launch(
    //     "appsrc name=source do-timestamp=true format=TIME ! "
    //     "video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1 ! " 
    //     " videoconvert  !  x264enc   ! "
    //     " video/x-h264,stream-format=avc,alignment=au ! "
    //     " h264parse   ! "
    //     // "  video/x-h264,stream-format=avs,alignment=au  ! "
    //     //  config-interval=-1  ! video/x-h264,stream-format=byte-stream,alignment=au !"
    //     // "queue ! "
    //     "appsink name=sink sync=false",
    //     // " mp4mux ! "
    //     // " filesink location=output.mp4",
    //     // 
    // NULL);

    if (!apps[cnt]->pipeline) {
        g_printerr("Failed to create pipeline\n");
        return -1;
    }

    // Get appsrc from pipeline
    GstElement* video_source = gst_bin_get_by_name(GST_BIN(apps[cnt]->pipeline), "video_source");
    apps[cnt]->video_appsrc = GST_APP_SRC(video_source);
    
    apps[cnt]->video_appsink = gst_bin_get_by_name(GST_BIN(apps[cnt]->pipeline), "video_sink");
    if (!apps[cnt]->video_appsink) {
        std::cerr << "Failed to get appsink element." << std::endl;
        gst_object_unref(apps[cnt]->pipeline);
        return -1;
    }

    g_object_set(apps[cnt]->video_appsink, "emit-signals", TRUE, nullptr);
        // Connect the new-sample signal
    g_signal_connect(apps[cnt]->video_appsink, "new-sample", G_CALLBACK(on_new_video_sample), GINT_TO_POINTER(cnt));






    // Set up bus
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(apps[cnt]->pipeline));
    gst_bus_add_watch(bus, bus_call, apps[cnt]->loop);
    gst_object_unref(bus);

    // Create CUDA context
    if (cnt == 0)
        apps[cnt]->context = gst_cuda_context_new(0);

    // Start playing
    gst_element_set_state(apps[cnt]->pipeline, GST_STATE_PLAYING);


  
    // // Run main loop
    // g_main_loop_run(loop);
    i++;
    return i-1;
 
}

void close_pipeline(int cnt){
    apps[cnt]->exit_flag = true;
    apps[cnt]->video_queue_cv.notify_one();
    gst_app_src_end_of_stream(apps[cnt]->video_appsrc);
    // gst_app_src_end_of_stream(apps[cnt]->audio_appsrc);


    // Run main loop
    g_main_loop_run(apps[cnt]->loop);
    

    // Clean up
    gst_element_set_state(apps[cnt]->pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(apps[cnt]->pipeline));
    while (! apps[cnt]->video_buffer_queue.empty()) {
       
         apps[cnt]->video_buffer_queue.front() ;
       
         apps[cnt]->video_buffer_queue.pop();
    }
    

}

PYBIND11_MODULE(gst_h264_endoder_pipeline, m)
{
    
    m.def("main_fun", &main_fun);
    m.def("push_cuda_tensor_frame", &push_cuda_tensor_frame,py::call_guard<py::gil_scoped_release>() );
    m.def("push_tensor_frame", &push_tensor_frame, py::call_guard<py::gil_scoped_release>());
    m.def("push_audio_packet", &push_audio_packet, py::call_guard<py::gil_scoped_release>());

    m.def("get_next_video_buffer_data", &get_next_video_buffer_data, "Get data from the next video buffer in the queue", py::call_guard<py::gil_scoped_release>());
    m.def("get_next_audio_buffer_data", &get_next_audio_buffer_data, "Get data from the next audio buffer in the queue", py::call_guard<py::gil_scoped_release>());

    m.def("close_pipeline", &close_pipeline);


   

} 



