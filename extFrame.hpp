#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

#include <iostream>
#include <string>

// For threads
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
    const cv::Mat* src;                 // Input
    cv::Mat* dst;                       // Output (gray, sobel)
    int height, width;                  // Frame dimensions: Height, width
    int tid, totalThreads;              // Thread id and total threads
};

// Thread function prototypes
void *grayThread(void *threadArgs);
void *sobelThread(void *threadArgs);

// Function prototypes
void grayscale(const cv::Mat &frame, cv::Mat &dest, int start, int end);
void sobelFilter(const cv::Mat &frame, cv::Mat &dest);
void saveImage(const cv::Mat &frame, std::string name);

#endif