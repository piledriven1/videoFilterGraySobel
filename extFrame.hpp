#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

#include <iostream>
#include <cstring>

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

cv::Mat grayscale(const cv::Mat &frame);
cv::Mat sobelFilter(const cv::Mat &frame);
void saveImage(const cv::Mat &frame, std::string name);

#endif