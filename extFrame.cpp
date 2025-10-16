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
    
    // Declare thread variables
    pthread_t thread[NUMTHREADS];
    threadArgs info[NUMTHREADS];

    // Declare video variables
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
        name += "_grayscale";
    } else if (filter == "sobel") {
        name += "_sobel";
    }
    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(name, 750, 500);               // Width, Height

    // Cycle through the video until the final frame is reached
    for(int i = 0; i < finalFrame; i++) {
        // Video must be read otherwise there will be no progress
        video.read(plain);
        if(plain.empty()) {
            std::cerr << "Empty frame; input a valid time" << std::endl;
            return 1;
        }

        // Allocate memory for output image (8-bit)
        if (i == 0) {                                     // Pre-allocates the memory once
            gray.create(plain.rows, plain.cols, CV_8UC1);
            if (filter == "sobel") {
                sobel.create(plain.rows, plain.cols, CV_8UC1);
            }
        }

        // Reduce the amount of processing based on user argument
        if(filter.compare("plain") == 0) {
            cv::imshow(name, plain);
        } else {
            // Create threads, fill struct and apply gray filter 
            for (int j = 0; j < NUMTHREADS; ++j) {
                info[j] = threadArgs{ &plain, &gray, plain.rows, plain.cols, j, NUMTHREADS};
                pthread_create(&thread[j], NULL, grayThread, &info[j]);
            }

            // Wait for all threads to finish
            for (int j = 0; j < NUMTHREADS; ++j) {
                pthread_join(thread[j], nullptr);
            }

            // Display video with a gray filter
            if(filter.compare("gray") == 0) {
                cv::imshow(name, gray);
            }

            // TODO: Integrate Sobel into threading!!!!!!!!!!!!!!!!!!!!!!!!!
            // if(filter.compare("sobel") == 0) {
            //     sobelFilter(gray, sobel);
            //     cv::imshow(name, sobel);
            // }
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

void *grayThread(void *args) {
    threadArgs* arg = static_cast<threadArgs*>(args);

    // Determines starting and ending rows (exclusive) for each thread
    int start = (arg->tid * arg->height) / arg->totalThreads;
    int end = ((arg->tid + 1) * arg->height) / arg->totalThreads;

    grayscale(*arg->src, *arg->dst, start, end);
    return nullptr;
}

void *sobelThread(void *threadArgs) {
    return nullptr;
}

void grayscale(const cv::Mat &frame, cv::Mat &dest, int start, int end) {
    CV_Assert(frame.type() == CV_8UC3);
    // Apply grayscale using BT.709 formula
    // Y = 0.2126R + 0.7152G + 0.0722B
    // Modify BT.709 formula values to match 8-bit value range (0-255)
    const int wB = 18;  // 0.0722 * 256 = 18.4832 ~ 18
    const int wG = 183; // 0.7152 * 256 ~ 183.0912 ~ 183
    const int wR = 54;  // 0.2126 * 256 ~ 54.4256 ~ 54

    for(int y = start; y < end; y++) {
        const cv::Vec3b* src = frame.ptr<cv::Vec3b>(y);
        uint8_t* dst = dest.ptr<uint8_t>(y);

        for(int x = 0; x < frame.cols; x++) {
            const uint8_t B = src[x][0];
            const uint8_t G = src[x][1];
            const uint8_t R = src[x][2];

            // Calculate for new value per pixel and divide by 256
            dst[x] =
                static_cast<uint8_t>(((wB * B) + (wG * G) + (wR * R)) >> 8);
        }
    }
}

void sobelFilter(const cv::Mat &frame, cv::Mat &dest) {
    // Checks for a single channel 8-bit greyscale input
    CV_Assert(frame.type() == CV_8UC1);

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

            dest.at<uint8_t>(y,x) = static_cast<uint8_t>(magnitude);
        }
    }
}

void saveImage(const cv::Mat &frame, std::string name) {
    cv::imwrite(name + ".jpg", frame);
    return;
}
