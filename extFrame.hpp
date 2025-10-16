#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

#include <iostream>
#include <string>

// For threading
#include <atomic>
#include <thread>
#include <vector>
#include <semaphore>

// For opencv
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#define NUMTHREADS 4

void grayscale(const cv::Mat &frame, cv::Mat &dest);
void sobelFilter(const cv::Mat &frame, cv::Mat &dest);
void saveImage(const cv::Mat &frame, std::string name);

#endif