#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

#include "extFrame.hpp"

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cerr << "Include the video, and second you want the frame, and name." << std::endl <<
        "./extFrame <VIDEO_FILE> <NAME> <NUM_OF_SEC>" << std::endl;
        return 1;
    }
    if(argc > 4) {
        std::cerr << "Too many arguments" << std::endl <<
        "./extFrame <VIDEO_FILE> <NAME> <NUM_OF_SEC>" << std::endl;
        return 1;
    }

    cv::Mat frame;
    cv::VideoCapture video;

    // Open video specified
    video.open(argv[1], 0);

    double fps = video.get(cv::CAP_PROP_FPS);
    int maxFrame = video.isOpened() ? 3 * (int)fps : (int)(fps * atoi(argv[3]));

    // Go through and cycle through
    for(int i = 0; i < maxFrame; i++) {
        // Read through the video
        video.read(frame);
        if(frame.empty()) {
            std::cerr << "Empty frame; input a valid time" << std::endl;
            return 1;
        }
    }

    // Write to a demo file
    std::string name = argv[2];
    saveImage(name, frame);
    cv::Mat gray = grayscale(frame);
    sobelFilter(gray);

    video.release();

    return 0;
}

cv::Mat grayscale(const cv::Mat &frame) {
    // Apply grayscale using BT.601 formula
    CV_Assert(frame.type() == CV_8UC3);
    cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
    // Y = 0.114B + 0.587G + 0.299R
    // Modify BT.2601 formula values to match 8-bit values (0-255)
    const int wB = 28;  // 0.114 * 256 = 29.184 ~ 29 - 1 = 28
    const int wG = 149; // 0.587 * 256 ~ 150.272 ~ 150 - 1 = 149
    const int wR = 76;  // 0.299 * 256 ~ 76.544 ~ 77 - 1 = 76

    for(int y = 0; y < frame.rows; y++) {
        const cv::Vec3b* src = frame.ptr<cv::Vec3b>(y);
        uint8_t* dst = gray.ptr<uint8_t>(y);

        for(int x = 0; x < frame.cols; x++) {
            const uint8_t B = src[x][0];
            const uint8_t G = src[x][1];
            const uint8_t R = src[x][2];

            // Calculate for new value per pixel and divide by 256
            dst[x] = static_cast<uchar>(((wB * B) + (wG * G) + (wR * R)) >> 8);
        }
    }
    // create output file
    saveImage("gray", gray);

    return gray;
}

void sobelFilter(const cv::Mat &frame) {
    CV_Assert(frame.type() == CV_8UC1);     // Check if provided img is a grayscale
    cv::Mat sobel(frame.rows, frame.cols, CV_8UC1); // Allocate memory for final output

    // Define matrices for sobel computation
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};    // For vertical
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};    // For horizontal

    for (int y = 1; y < frame.rows - 1; y++) {          // NOTE: Scans vertically
        for (int x = 1; x < frame.cols - 1; x++) {      // NOTE: Scans horizontally  
            int gxSum = 0, gySum = 0;

            // 3×3 neighborhood
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    int p = static_cast<int>(frame.at<uchar>(y + j, x + k));

                    gxSum += p * gx[j + 1][k + 1];
                    gySum += p * gy[j + 1][k + 1];
                }
            }

            // Compute magnitude and prevent numbers not within 0-255
            int magnitude = std::abs(gxSum) + std::abs(gySum);  // Rough equivalent of sqrt(gxSum^2 + gySum^2)
            if (magnitude > 255) magnitude = 255;

            sobel.at<uchar>(y,x) = static_cast<uchar>(magnitude);
        }
    }

    // Create output file
    saveImage("sobel", sobel);
    
    return;
}

void saveImage(std::string name, const cv::Mat &frame) {
    cv::imwrite(name + ".jpg", frame);
    return;
}
