#include "extFrame.hpp"

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cerr << "Not enough arguments" << std::endl <<
        "./extFrame <VIDEO_FILE> <VIDEO_TYPE> <NUM_OF_SEC>" << std::endl;
        return 1;
    }
    if(argc > 4) {
        std::cerr << "Too many arguments" << std::endl <<
        "./extFrame <VIDEO_FILE> <VIDEO_TYPE> <NUM_OF_SEC>" << std::endl;
        return 1;
    }
    std::string filter = argv[2];
    if(filter.compare("plain") != 0 && 
            filter.compare("gray") != 0 && 
            filter.compare("sobel") != 0) {
        std::cerr << "Invalid name" << std::endl << argv[2] <<
                    " != plain || gray || sobel" << std::endl;
        return 1;
    }

    cv::VideoCapture video;
    std::string name = argv[1];

    // Open video specified
    video.open(name, 0);

    // Find the FPS as an integer value
    int fps = (int)video.get(cv::CAP_PROP_FPS);
    int finalFrame = fps * atoi(argv[3]);

    cv::Mat plain, gray, sobel;

    // Resizes the GUI window 
    if (filter == "gray") {
        name = name + "_grayscale";
    } else if (filter == "sobel") {
        name = name + "_sobel";
    }
    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(name, 750, 500);             // Width, Height

    // Cycle through the video until the final frame is reached
    for(int i = 0; i < finalFrame; i++) {
        // Video must be read otherwise there will be no progress
        video.read(plain);
        if(plain.empty()) {
            std::cerr << "Empty frame; input a valid time" << std::endl;
            return 1;
        }

        // Reduce the amount of processing based on user argument
        if(filter.compare("plain") == 0) {
            cv::imshow(name, plain);
        }
        else {
            gray = grayscale(plain);
            if(filter.compare("gray") == 0) {
                cv::imshow(name, gray);
            }
            if(filter.compare("sobel") == 0) {
                sobel = sobelFilter(gray);
                cv::imshow(name, sobel);
            }
        }
        
        cv::waitKey(1000 / fps);
    }

    // Cleanup
    video.release();
    sobel.release();
    gray.release();
    plain.release();

    cv::destroyAllWindows();

    return 0;
}

cv::Mat grayscale(const cv::Mat &frame) {
    CV_Assert(frame.type() == CV_8UC3);
    cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
    // Apply grayscale using BT.709 formula
    // Y = 0.2126R + 0.7152G + 0.0722B
    // Modify BT.709 formula values to match 8-bit value range (0-255)
    const int wB = 18;  // 0.0722 * 256 = 18.4832 ~ 18
    const int wG = 183; // 0.7152 * 256 ~ 183.0912 ~ 183
    const int wR = 54;  // 0.2126 * 256 ~ 54.4256 ~ 54

    for(int y = 0; y < frame.rows; y++) {
        const cv::Vec3b* src = frame.ptr<cv::Vec3b>(y);
        uint8_t* dst = gray.ptr<uint8_t>(y);

        for(int x = 0; x < frame.cols; x++) {
            const uint8_t B = src[x][0];
            const uint8_t G = src[x][1];
            const uint8_t R = src[x][2];

            // Calculate for new value per pixel and divide by 256
            dst[x] =
                static_cast<uint8_t>(((wB * B) + (wG * G) + (wR * R)) >> 8);
        }
    }

    return gray;
}

cv::Mat sobelFilter(const cv::Mat &frame) {
    // Checks for a single channel 8-bit greyscale input
    CV_Assert(frame.type() == CV_8UC1);
    // Allocate memory for output image (8-bit)
    cv::Mat sobel(frame.rows, frame.cols, CV_8UC1);

    // Define matrices for sobel computation
    int gx[3][3] = {{-1, 0, 1}, 
                    {-2, 0, 2},
                     {-1, 0, 1}
                    };
    int gy[3][3] = {{-1, -2, -1},
                    {0, 0, 0},
                    {1, 2, 1}
                    };

    // Loop to process interior pixels only. Skips borders.
    for (int y = 1; y < frame.rows - 1; y++) {      // Scans vertically
        for (int x = 1; x < frame.cols - 1; x++) {  // Scans horizontal
            int gxSum = 0;
            int gySum = 0;

            // Compute 3x3 convolution with Sobel matrices and neighboring pixels
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    int p = static_cast<int>(frame.at<uint8_t>(y + j, x + k));

                    gxSum += p * gx[j + 1][k + 1];
                    gySum += p * gy[j + 1][k + 1];
                }
            }

            // Compute magnitude using (|Gx|+|Gy|) and clamp range to [0, 255]
            int magnitude = (std::abs(gxSum) + std::abs(gySum) > 255) ?
                            255 : std::abs(gxSum) + std::abs(gySum);

            sobel.at<uint8_t>(y,x) = static_cast<uint8_t>(magnitude);
        }
    }

    return sobel;
}

void saveImage(const cv::Mat &frame, std::string name) {
    cv::imwrite(name + ".jpg", frame);
    return;
}
