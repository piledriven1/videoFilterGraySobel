#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

cv::Mat grayscale(const cv::Mat &frame, std::string name);
void sobelFilter(const cv::Mat &frame, std::string name);
void saveImage(const cv::Mat &frame, std::string name);

#endif