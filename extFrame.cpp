#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    cv::Mat frame;
    cv::VideoCapture video;

    if(argc < 3) {
        std::cerr << "Define the variable and seconds.\n" <<
        "./extFrame <VIDEO_FILE> <NUM_OF_SEC>" << std::endl;
    }

    // Open video
    video.open(argv[1], 0);
    video.release();

    video.open("demo.mp4", 0);
    double fps = video.get(cv::CAP_PROP_FPS);

    // Go through and cycle through
    for(int i = 0; i < (int)(fps * atoi(argv[2])); i++) {
        // Read through the video
        video.read(frame);
        if(frame.empty()) {
            std::cerr << "Empty frame; input a valid time" << std::endl;
            return 1;
        }
    }

    // Write to a demo file
    cv::imwrite("demo.jpg", frame);
    video.release();

    return 0;
}
