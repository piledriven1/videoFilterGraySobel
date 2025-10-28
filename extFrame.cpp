#include "extFrame.hpp"

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cerr << "Not enough arguments" << std::endl <<
        "./extFrame <VIDEO_FILE> <FILTER_TYPE> <NUM_OF_SEC>" << std::endl;
        return 1;
    }
    if(argc > 4) {
        std::cerr << "Too many arguments" << std::endl <<
        "./extFrame <VIDEO_FILE> <FILTER_TYPE> <NUM_OF_SEC>" << std::endl;
        return 1;
    }
    std::string filter = argv[2];
    if(filter.compare("plain") != 0 && 
            filter.compare("gray") != 0 && 
            filter.compare("sobel") != 0) {
        std::cerr << "Invalid filter" << std::endl << argv[2] <<
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
    if (fps <= 0) fps = 30;
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
        // Pre-allocates the memory once
        if (i == 0 && (filter == "gray" || filter == "sobel")) {  // (fixed compare logic)
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
                info[j] = threadArgs{&plain, &gray, plain.rows, plain.cols, j, NUMTHREADS};
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

            // Display video with a sobel filter
            if(filter.compare("sobel") == 0) {
                // Create threads, fill struct and apply sobel filter 
                for (int j = 0; j < NUMTHREADS; ++j) {
                    info[j] = threadArgs{
                                    &gray, &sobel,
                                    plain.rows, plain.cols,
                                    j, NUMTHREADS
                        };
                    pthread_create(&thread[j], NULL, sobelThread, &info[j]);
                }

                // Wait for all threads to finish
                for (int j = 0; j < NUMTHREADS; ++j) {
                    pthread_join(thread[j], nullptr);
                }

                cv::imshow(name, sobel);                // Display frame 
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

void *grayThread(void *args) {
    threadArgs* arg = static_cast<threadArgs*>(args);

    // Determines starting and ending rows (exclusive) for each thread
    int start = (arg->tid * arg->height) / arg->totalThreads;
    int end = ((arg->tid + 1) * arg->height) / arg->totalThreads;

    grayscale(*arg->src, *arg->dst, start, end);
    return nullptr;
}

void *sobelThread(void *args) {
    threadArgs* arg = static_cast<threadArgs*>(args);

    // Determines starting and ending rows (exclusive) for each thread
    int start = (arg->tid * arg->height) / arg->totalThreads;
    int end = ((arg->tid + 1) * arg->height) / arg->totalThreads;

    sobelFilter(*arg->src, *arg->dst, start, end);
    return nullptr;
}

void grayscale(const cv::Mat &frame, cv::Mat &dest, int start, int end) {
    CV_Assert(frame.type() == CV_8UC3);
    // Apply grayscale using BT.709 formula
    // Y = 0.2126R + 0.7152G + 0.0722B
    // Modify BT.709 formula values to match 8-bit value range (0-255)
    const uint16_t wB = 18;   // 0.0722 * 256 = 18.4832 ~ 18
    const uint16_t wG = 183;  // 0.7152 * 256 ~ 183.0912 ~ 183
    const uint16_t wR = 54;   // 0.2126 * 256 ~ 54.4256 ~ 54

    const int cols = frame.cols;

    for(int y = start; y < end; y++) {
        const uint8_t* srow = frame.ptr<uint8_t>(y);
        uint8_t* dst = dest.ptr<uint8_t>(y);

        int x = 0;
        // NEON path: process 16 pixels per iteration
        for (; x <= cols - 16; x += 16) {
            // Load 16 interleaved BGR pixels, deinterleave into 3 vectors
            uint8x16x3_t bgr = vld3q_u8(srow + 3 * x);

            // Widen to 16-bit
            uint16x8_t b_lo = vmovl_u8(vget_low_u8 (bgr.val[0]));
            uint16x8_t b_hi = vmovl_u8(vget_high_u8(bgr.val[0]));
            uint16x8_t g_lo = vmovl_u8(vget_low_u8 (bgr.val[1]));
            uint16x8_t g_hi = vmovl_u8(vget_high_u8(bgr.val[1]));
            uint16x8_t r_lo = vmovl_u8(vget_low_u8 (bgr.val[2]));
            uint16x8_t r_hi = vmovl_u8(vget_high_u8(bgr.val[2]));

            // Calculate for new value per pixel and divide by 256
            // dst[x] = ((wB*B + wG*G + wR*R) >> 8) for 16 pixels
            uint16x8_t y_lo = vmulq_n_u16(r_lo, wR);
            y_lo = vmlaq_n_u16(y_lo, g_lo, wG);
            y_lo = vmlaq_n_u16(y_lo, b_lo, wB);

            uint16x8_t y_hi = vmulq_n_u16(r_hi, wR);
            y_hi = vmlaq_n_u16(y_hi, g_hi, wG);
            y_hi = vmlaq_n_u16(y_hi, b_hi, wB);

            y_lo = vshrq_n_u16(y_lo, 8);
            y_hi = vshrq_n_u16(y_hi, 8);
            uint8x16_t y8 = vcombine_u8(vmovn_u16(y_lo), vmovn_u16(y_hi));
            vst1q_u8(dst + x, y8);
        }

        // Scalar tail for leftover <16 pixels (simple, safe)
        for(; x < cols; x++) {
            const uint8_t B = srow[3*x + 0];
            const uint8_t G = srow[3*x + 1];
            const uint8_t R = srow[3*x + 2];
            dst[x] = static_cast<uint8_t>(((wB * B) + (wG * G) + (wR * R)) >> 8);
        }
    }
}

void sobelFilter(const cv::Mat &frame, cv::Mat &dest, int start, int end) {
    // Checks for a single channel 8-bit greyscale input
    CV_Assert(frame.type() == CV_8UC1);

    // Define matrices for sobel computation (reference)
    // int gx[3][3] = {{-1, 0, 1}, 
    //                 {-2, 0, 2},
    //                 {-1, 0, 1}};
    // int gy[3][3] = {{-1, -2, -1},
    //                 { 0,  0,  0},
    //                 { 1,  2,  1}};
    // We implement these kernels directly with NEON differences.

    const int cols = frame.cols;

    // Loop to process interior pixels only. Skips borders.
    int ys = std::max(start, 1);
    int ye = std::min(end, frame.rows - 1);  // Scans vertically
    for (int y = ys; y < ye; y++) {
        const uint8_t* t = frame.ptr<uint8_t>(y - 1);
        const uint8_t* m = frame.ptr<uint8_t>(y + 0);
        const uint8_t* b = frame.ptr<uint8_t>(y + 1);
        uint8_t* d = dest.ptr<uint8_t>(y);

        int x = 1;                            // Scans horizontal
        // NEON path: 16-pixel wide blocks, computing |Gx|+|Gy|
        for (; x <= cols - 17; x += 16) {
            uint8x16_t tL8 = vld1q_u8(t + (x - 1));
            uint8x16_t tC8 = vld1q_u8(t + (x + 0));
            uint8x16_t tR8 = vld1q_u8(t + (x + 1));

            uint8x16_t mL8 = vld1q_u8(m + (x - 1));
            uint8x16_t mR8 = vld1q_u8(m + (x + 1));

            uint8x16_t bL8 = vld1q_u8(b + (x - 1));
            uint8x16_t bC8 = vld1q_u8(b + (x + 0));
            uint8x16_t bR8 = vld1q_u8(b + (x + 1));

            auto WLO = [](uint8x16_t v){ return vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))); };
            auto WHI = [](uint8x16_t v){ return vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))); };

            int16x8_t tL_lo = WLO(tL8), tC_lo = WLO(tC8), tR_lo = WLO(tR8);
            int16x8_t mL_lo = WLO(mL8),            mR_lo = WLO(mR8);
            int16x8_t bL_lo = WLO(bL8), bC_lo = WLO(bC8), bR_lo = WLO(bR8);

            int16x8_t tL_hi = WHI(tL8), tC_hi = WHI(tC8), tR_hi = WHI(tR8);
            int16x8_t mL_hi = WHI(mL8),            mR_hi = WHI(mR8);
            int16x8_t bL_hi = WHI(bL8), bC_hi = WHI(bC8), bR_hi = WHI(bR8);

            // Compute 3x3 convolution with Sobel matrices and neighboring pixels
            auto sobel_gx = [](int16x8_t tR, int16x8_t tL, int16x8_t mR, int16x8_t mL, int16x8_t bR, int16x8_t bL){
                int16x8_t v = vsubq_s16(tR, tL);
                int16x8_t md = vsubq_s16(mR, mL);
                v = vaddq_s16(v, vmulq_n_s16(md, 2));
                v = vaddq_s16(v, vsubq_s16(bR, bL));
                return v;
            };
            auto sobel_gy = [](int16x8_t tL, int16x8_t tC, int16x8_t tR, int16x8_t bL, int16x8_t bC, int16x8_t bR){
                int16x8_t v = vsubq_s16(bL, tL);
                int16x8_t cd = vsubq_s16(bC, tC);
                v = vaddq_s16(v, vmulq_n_s16(cd, 2));
                v = vaddq_s16(v, vsubq_s16(bR, tR));
                return v;
            };

            int16x8_t gx_lo = sobel_gx(tR_lo, tL_lo, mR_lo, mL_lo, bR_lo, bL_lo);
            int16x8_t gy_lo = sobel_gy(tL_lo, tC_lo, tR_lo, bL_lo, bC_lo, bR_lo);

            int16x8_t gx_hi = sobel_gx(tR_hi, tL_hi, mR_hi, mL_hi, bR_hi, bL_hi);
            int16x8_t gy_hi = sobel_gy(tL_hi, tC_hi, tR_hi, bL_hi, bC_hi, bR_hi);

            // Compute magnitude using (|Gx|+|Gy|) and clamp range to [0, 255]
            uint16x8_t mag_lo = vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(gx_lo)),
                                          vreinterpretq_u16_s16(vabsq_s16(gy_lo)));
            uint16x8_t mag_hi = vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(gx_hi)),
                                          vreinterpretq_u16_s16(vabsq_s16(gy_hi)));

            uint8x16_t out = vcombine_u8(vqmovn_u16(mag_lo), vqmovn_u16(mag_hi));
            vst1q_u8(d + x, out);
        }

        // Scalar tail (and edges) for remaining pixels
        for (; x < cols - 1; ++x) {
            int gxSum = 0;
            int gySum = 0;

            // Compute 3x3 convolution with Sobel matrices and neighboring pixels
            gxSum += -1* t[x-1] +  1* t[x+1];
            gxSum += -2* m[x-1] +  2* m[x+1];
            gxSum += -1* b[x-1] +  1* b[x+1];

            gySum += -1* t[x-1] + -2* t[x] + -1* t[x+1];
            gySum +=  1* b[x-1] +  2* b[x] +  1* b[x+1];

            // Compute magnitude using (|Gx|+|Gy|) and clamp range to [0, 255]
            int magnitude = std::abs(gxSum) + std::abs(gySum);
            d[x] = static_cast<uint8_t>(magnitude > 255 ? 255 : magnitude);
        }
        // (Leftmost/rightmost columns and top/bottom rows are skipped, as in your original loop.)
    }
}

void saveImage(const cv::Mat &frame, std::string name) {
    cv::imwrite(name + ".jpg", frame);
    return;
}