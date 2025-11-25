// #include <opencv2/opencv.hpp>
// #include <opencv2/photo.hpp>
// #include <iostream>
// #include <string>

// cv::Mat enhance_image(const cv::Mat& img, int brightness = 30, double contrast = 1.3, double sharpen_amount = 0.5) {
//     cv::Mat img_bilateral, denoised, brightened, gaussian, sharpened;
    
//     // Step 1: Bilateral filter (equivalent to the first line in Python)
//     // cv::bilateralFilter(img, img_bilateral, 2, 1, 1);
    
//     // Step 2: Denoising
//     cv::fastNlMeansDenoisingColored(img, denoised, 2, 10, 5, 15);
    
//     // Step 3: Brightness and contrast adjustment
//     denoised.convertTo(brightened, -1, contrast, brightness);
    
//     // Step 4: Sharpen using unsharp masking
//     cv::GaussianBlur(brightened, gaussian, cv::Size(0, 0), 2.0);
//     cv::addWeighted(brightened, 1 + sharpen_amount, gaussian, -sharpen_amount, 0, sharpened);
    
//     return sharpened;
// }

// void print_usage(const char* program_name) {
//     std::cout << "Usage: " << program_name << " <src_path> <dst_path> [brightness] [contrast] [sharpen_amount]\n";
//     std::cout << "  src_path: Input image path\n";
//     std::cout << "  dst_path: Output image path\n";
//     std::cout << "  brightness: Brightness adjustment (default: 30)\n";
//     std::cout << "  contrast: Contrast multiplier (default: 1.3)\n";
//     std::cout << "  sharpen_amount: Sharpening strength (default: 0.5)\n";
//     std::cout << "\nExample: " << program_name << " input.jpg output.jpg 25 1.2 0.7\n";
// }

// int main(int argc, char* argv[]) {
//     // Default values
//     int brightness = 30;
//     double contrast = 1.3;
//     double sharpen_amount = 0.5;
    
//     // Check minimum arguments
//     if (argc < 3) {
//         print_usage(argv[0]);
//         return -1;
//     }
    
//     std::string src_path = argv[1];
//     std::string dst_path = argv[2];
    
//     // Parse optional parameters
//     if (argc > 3) brightness = std::stoi(argv[3]);
//     if (argc > 4) contrast = std::stod(argv[4]);
//     if (argc > 5) sharpen_amount = std::stod(argv[5]);
    
//     // Load the image
//     cv::Mat img = cv::imread(src_path, cv::IMREAD_COLOR);
//     if (img.empty()) {
//         std::cerr << "Error: Could not load image from " << src_path << std::endl;
//         return -1;
//     }
    
//     std::cout << "Processing image: " << src_path << std::endl;
//     std::cout << "Parameters - Brightness: " << brightness 
//               << ", Contrast: " << contrast 
//               << ", Sharpen: " << sharpen_amount << std::endl;
    
//     // Enhance the image
//     cv::Mat enhanced = enhance_image(img, brightness, contrast, sharpen_amount);
    
//     // Save the result
//     if (cv::imwrite(dst_path, enhanced)) {
//         std::cout << "Enhanced image saved to: " << dst_path << std::endl;
//     } else {
//         std::cerr << "Error: Could not save image to " << dst_path << std::endl;
//         return -1;
//     }
    
//     return 0;
// }


#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <string>
#include <vector>

cv::Mat enhance_image(const cv::Mat& img, int brightness = 30, double contrast = 1.3, double sharpen_amount = 0.5) {
    cv::Mat result;
    
    // Check if image has alpha channel
    if (img.channels() == 4) {
        // Handle RGBA image
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        // Extract RGB channels and alpha channel
        cv::Mat rgb_img, alpha_channel = channels[3];
        cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, rgb_img);
        
        // Process RGB channels
        cv::Mat img_bilateral, denoised, brightened, gaussian, sharpened;
        
        // Step 1: Bilateral filter
        // cv::bilateralFilter(rgb_img, img_bilateral, 2, 1, 1);
        
        // Step 2: Denoising
        cv::fastNlMeansDenoisingColored(rgb_img, denoised, 2, 10, 5, 15);
        
        // Step 3: Brightness and contrast adjustment
        denoised.convertTo(brightened, -1, contrast, brightness);
        
        // Step 4: Sharpen using unsharp masking
        cv::GaussianBlur(brightened, gaussian, cv::Size(0, 0), 2.0);
        cv::addWeighted(brightened, 1 + sharpen_amount, gaussian, -sharpen_amount, 0, sharpened);
        
        // Split enhanced RGB back into channels
        std::vector<cv::Mat> enhanced_channels;
        cv::split(sharpened, enhanced_channels);
        
        // Add the original alpha channel back
        enhanced_channels.push_back(alpha_channel);
        
        // Merge all channels including alpha
        cv::merge(enhanced_channels, result);
        
    } else if (img.channels() == 3) {
        // Handle RGB image (original logic)
        cv::Mat img_bilateral, denoised, brightened, gaussian;
        
        // Step 1: Bilateral filter
        // cv::bilateralFilter(img, img_bilateral, 2, 1, 1);
        
        // Step 2: Denoising
        cv::fastNlMeansDenoisingColored(img, denoised, 2, 10, 5, 15);
        
        // Step 3: Brightness and contrast adjustment
        denoised.convertTo(brightened, -1, contrast, brightness);
        
        // Step 4: Sharpen using unsharp masking
        cv::GaussianBlur(brightened, gaussian, cv::Size(0, 0), 2.0);
        cv::addWeighted(brightened, 1 + sharpen_amount, gaussian, -sharpen_amount, 0, result);
        
    } else if (img.channels() == 1) {
        // Handle grayscale image
        cv::Mat img_bilateral, denoised, brightened, gaussian;
        
        // Step 1: Bilateral filter
        // cv::bilateralFilter(img, img_bilateral, 2, 1, 1);
        
        // Step 2: Denoising (use grayscale version)
        cv::fastNlMeansDenoising(img, denoised, 2, 5, 15);
        
        // Step 3: Brightness and contrast adjustment
        denoised.convertTo(brightened, -1, contrast, brightness);
        
        // Step 4: Sharpen using unsharp masking
        cv::GaussianBlur(brightened, gaussian, cv::Size(0, 0), 2.0);
        cv::addWeighted(brightened, 1 + sharpen_amount, gaussian, -sharpen_amount, 0, result);
        
    } else {
        std::cerr << "Error: Unsupported number of channels: " << img.channels() << std::endl;
        return img.clone();
    }
    
    return result;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <src_path> <dst_path> [brightness] [contrast] [sharpen_amount]\n";
    std::cout << "  src_path: Input image path\n";
    std::cout << "  dst_path: Output image path\n";
    std::cout << "  brightness: Brightness adjustment (default: 30)\n";
    std::cout << "  contrast: Contrast multiplier (default: 1.3)\n";
    std::cout << "  sharpen_amount: Sharpening strength (default: 0.5)\n";
    std::cout << "\nSupports RGB, RGBA, and grayscale images\n";
    std::cout << "Example: " << program_name << " input.png output.png 25 1.2 0.7\n";
}

int main(int argc, char* argv[]) {
    // Default values
    int brightness = 30;
    double contrast = 1.3;
    double sharpen_amount = 0.5;
    
    // Check minimum arguments
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }
    
    std::string src_path = argv[1];
    std::string dst_path = argv[2];
    
    // Parse optional parameters
    if (argc > 3) brightness = std::stoi(argv[3]);
    if (argc > 4) contrast = std::stod(argv[4]);
    if (argc > 5) sharpen_amount = std::stod(argv[5]);
    
    // Load the image with alpha channel support
    cv::Mat img = cv::imread(src_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << src_path << std::endl;
        return -1;
    }
    
    std::cout << "Processing image: " << src_path << std::endl;
    std::cout << "Image channels: " << img.channels() << std::endl;
    std::cout << "Parameters - Brightness: " << brightness 
              << ", Contrast: " << contrast 
              << ", Sharpen: " << sharpen_amount << std::endl;
    
    // Enhance the image
    cv::Mat enhanced = enhance_image(img, brightness, contrast, sharpen_amount);
    
    // Save the result
    if (cv::imwrite(dst_path, enhanced)) {
        std::cout << "Enhanced image saved to: " << dst_path << std::endl;
        std::cout << "Output channels: " << enhanced.channels() << std::endl;
    } else {
        std::cerr << "Error: Could not save image to " << dst_path << std::endl;
        return -1;
    }
    
    return 0;
}