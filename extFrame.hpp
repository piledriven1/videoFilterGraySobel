#ifndef EXTFRAME_HPP
#define EXTFRAME_HPP

cv::Mat grayscale(const cv::Mat &frame);
void sobelFilter(const cv::Mat &frame);
void saveImage(std::string name, const cv::Mat &frame);

#endif