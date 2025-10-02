#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

#include "extFrame.hpp"

int main(int argc, char* argv[]) {
    cv::Mat frame;
    cv::VideoCapture video;

    if(argc < 3) {
        std::cerr << "Define the variable and seconds.\n" <<
        "./extFrame <VIDEO_FILE> <NUM_OF_SEC>" << std::endl;
    }

    // Open video specified
    video.open("demo.mp4", 0);

    double fps = video.get(cv::CAP_PROP_FPS);
    int maxFrame = video.isOpened() ? 3 * (int)fps : (int)(fps * atoi(argv[2]));

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
    cv::imwrite("demo.jpg", frame);
    grayscale(frame);

    return 0;
}

void grayscale(const cv::Mat &frame) {
    // Apply grayscale using BT.601 formula
    CV_Assert(frame.type() == CV_8UC3);
    cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
    // Y = 0.114B + 0.587G + 0.299R
    // Modify BT.2601 formula values to match 8-bit values (0-255)
    const int wB = 28;  // 0.114 * 256 = 29.184 ~ 29 - 1 = 28
    const int wG = 149; // 0.587 * 256 ~ 150.272 ~ 150 - 1 = 149
    const int wR = 76;  // 0.299 * 256 ~ 76.544 ~ 77 - 1 = 76

    int x = 0;
    int y = 0;
    for(; y < frame.rows; y++) {
        const cv::Vec3b* src = frame.ptr<cv::Vec3b>(y);
        uint8_t* dst = gray.ptr<uint8_t>(y);

        for(; x < frame.cols; x++) {
            const uint8_t B = src[x][0];
            const uint8_t G = src[x][1];
            const uint8_t R = src[x][2];

            // Calculate for new value per pixel and divide by 256
            dst[x] = static_cast<uchar>(((wB * B) + (wG * G) + (wR * R)) >> 8);
        }
    }
    // create output file
    cv::imwrite("demo_gray.jpg", frame);
}
