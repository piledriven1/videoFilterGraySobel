#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

#include <iostream>
#include <string>

#include <pthread.h>

// For opencv
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

// Macros
#define NUMTHREADS 4

// Struct
struct threadArgs {
    const cv::Mat* src; // read-only input (e.g., BGR frame)
    cv::Mat* dst;       // output (e.g., gray), pre-allocated
    int height, width;           // Height, width
    int tid, totalThreads;            // thread id and total threads
};

// Function prototypes
void *grayThread(void *threadArgs);

void grayscale(const cv::Mat &frame, cv::Mat &dest, int start, int end);
void sobelFilter(const cv::Mat &frame, cv::Mat &dest);
void saveImage(const cv::Mat &frame, std::string name);

#endif