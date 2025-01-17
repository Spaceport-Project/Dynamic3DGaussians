#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <turbojpeg.h>


// std::vector<uint8_t> *buffer =new std::vector<uint8_t>();

namespace py = pybind11;
std::tuple<std::string, int, uintptr_t > encodeImageBinary(
    py::array_t<uint8_t>& img,
    const std::string& format,
    int jpegQuality = 75
) {
     
    // auto buf = img.request(); 
    // py::array_t<uint8_t> img2 =  py::array_t<uint8_t>(buf);
    // std::cout << img2.data() << " " << img.data() << std::endl; // this won't return the same address twice
   

    // auto rows = img.shape(0);
    // auto cols = img.shape(1);
    // auto channels = img.shape(2);
    // // std::cout << "rows: " << rows << " cols: " << cols << " channels: " << channels << std::endl;
    // auto type = CV_8UC3;
    // cv::Mat image(rows, cols, type, (unsigned char*)img.data());

    py::buffer_info buf = img.request();
     // Get array dimensions
    int height = buf.shape[0];
    int width = buf.shape[1];
    cv::Mat image(height, width, CV_8UC3, buf.ptr);

    // uint8_t* input_ptr = static_cast<uint8_t*>(buf.ptr);
    
  

    // cv::imwrite("sample_cplusplus.jpeg", image);




    std::vector<uint8_t> *buffer =new std::vector<uint8_t>();
    std::vector<int> params;
    std::string mediaType;

    if (format == "png") {
        mediaType = "image/png";
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(3);  // Default compression level for PNG
    } else if (format == "jpeg") {
        mediaType = "image/jpeg";
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(jpegQuality);
    } else {
        throw std::invalid_argument("Unsupported format: " + format);
    }

    // Encode the image
    if (!cv::imencode("." + format, image, *buffer, params)) {
        throw std::runtime_error("Failed to encode image.");
    }

    return {mediaType, buffer->size(), (uintptr_t)buffer->data()};

}

std::tuple<std::string, int, uintptr_t > encodeImageBinary2(
    uintptr_t cuda_ptr ,
    int size,
    int height,
    int width,
    const std::string& format,
    int jpegQuality = 75
) {
   


    uint8_t *h_data = (uint8_t *)malloc(size);
    cudaMemcpy(h_data, (uint8_t *)cuda_ptr, size, cudaMemcpyDeviceToHost);
   
    // std::cout << "rows: " << height << " cols: " << width   << "size: " <<size<<std::endl;
    cv::Mat image(height, width, CV_8UC3, h_data);


    std::vector<uint8_t>  *buffer(new std::vector<uint8_t>());
    std::vector<int> params;
    std::string mediaType;

    if (format == "png") {
        mediaType = "image/png";
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(3);  // Default compression level for PNG
    } else if (format == "jpeg") {
        mediaType = "image/jpeg";
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(jpegQuality);
    } else {
        throw std::invalid_argument("Unsupported format: " + format);
    }

    // Encode the image
    if (!cv::imencode("." + format, image, *buffer, params)) {
        throw std::runtime_error("Failed to encode image.");
    }
    free(h_data);
    return {mediaType, buffer->size(), (uintptr_t)buffer->data()};


}


std::tuple<std::string, int, uintptr_t > encodeImageBinaryTurboJpeg(
    uintptr_t cuda_ptr ,
    int size,
    int height,
    int width,
    const std::string& format,
    int jpegQuality = 75
) {
   
    uint8_t *buffer = (uint8_t *)malloc(size);
    cudaMemcpy(buffer, (uint8_t *)cuda_ptr, size, cudaMemcpyDeviceToHost);
    unsigned char* compressedImage = NULL; //!< Memory is allocated by tjCompress2 if _jpegSize == 0
    long unsigned int jpegSize = 0;

    // std::shared_ptr<unsigned char *> compressedImage (NULL); //!< Memory is allocated by tjCompress2 if _jpegSize == 0

    std::string mediaType;


    tjhandle jpegCompressor = tjInitCompress();
    tjCompress2(jpegCompressor, buffer, width, 0, height, TJPF_RGB,
          &compressedImage, &jpegSize, TJSAMP_444, jpegQuality,
          TJFLAG_FASTDCT);

    tjDestroy(jpegCompressor);
    // tjFree(compressedImage);
    // std::cout<<jpegSize<<" "<<compressedImage[0]<<" "<<compressedImage[122]<<std::endl;

    free(buffer);
    // std::shared_ptr<uint8_t> shared_mem(compressedImage, [](uint8_t* p) { 
    //      std::this_thread::sleep_for(std::chrono::seconds(2));
    //     delete[] p; 
    // });
    // std::shared_ptr<uint8_t[]> copy_image(new uint8_t[jpegSize]);
    // std::memcpy(copy_image.get(), compressedImage, jpegSize);
    // tjFree(compressedImage);

    return {mediaType, jpegSize, (uintptr_t)compressedImage};


}


// Define the module
PYBIND11_MODULE(jpeg_encoder, m) {
    m.def("encode_image_binary", &encodeImageBinary, py::arg("image"), py::arg("format"), py::arg("jpeg_quality") = 75,
          "Encode an image to a binary format without GIL",
          py::call_guard<py::gil_scoped_release>());

    m.def("encode_image_binary2", &encodeImageBinary2, py::arg("image"), py::arg("size"), py::arg("height"), py::arg("width"), py::arg("format"), py::arg("jpeg_quality") = 75,
          "Encode an image to a binary format without GIL",
          py::call_guard<py::gil_scoped_release>());

    m.def("encode_image_binaryturbojpeg", &encodeImageBinaryTurboJpeg, py::arg("image"), py::arg("size"), py::arg("height"), py::arg("width"), py::arg("format"), py::arg("jpeg_quality") = 75,
          "Encode an image to a binary format without GIL",
          py::call_guard<py::gil_scoped_release>());
}